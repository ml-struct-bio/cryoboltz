# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

from __future__ import annotations

from math import sqrt
import random

from einops import rearrange
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from boltz.data import const
import boltz.model.layers.initialize as init
from boltz.model.loss.diffusion import (
    smooth_lddt_loss,
    weighted_rigid_align,
)
from boltz.model.modules.encoders import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    FourierEmbedding,
    PairwiseConditioning,
    SingleConditioning,
)
from boltz.model.modules.transformers import (
    ConditionedTransitionBlock,
    DiffusionTransformer,
)
from boltz.model.modules.utils import (
    LinearNoBias,
    center_random_augmentation,
    apply_transform,
    default,
    log,
)
from boltz.model.modules.guidance import (
    init_guidance,
    init_multi_guidance,
    compute_guidance,
    schedule_guidance,
)
from boltz.model.modules.coupling import (
    estimate_flexibility_from_coords,
    build_coupling_strength,
    compute_coupling_gradient,
)
from boltz.model.modules.hierarchy import (
    HierarchicalCouplingSchedule,
    build_default_groups,
)

class DiffusionModule(Module):
    """Diffusion module"""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: int = 16,
        dim_fourier: int = 256,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        activation_checkpointing: bool = False,
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the diffusion module.

        Parameters
        ----------
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        atoms_per_window_queries : int, optional
            The number of atoms per window for queries, by default 32.
        atoms_per_window_keys : int, optional
            The number of atoms per window for keys, by default 128.
        sigma_data : int, optional
            The standard deviation of the data distribution, by default 16.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.
        atom_encoder_depth : int, optional
            The depth of the atom encoder, by default 3.
        atom_encoder_heads : int, optional
            The number of heads in the atom encoder, by default 4.
        token_transformer_depth : int, optional
            The depth of the token transformer, by default 24.
        token_transformer_heads : int, optional
            The number of heads in the token transformer, by default 8.
        atom_decoder_depth : int, optional
            The depth of the atom decoder, by default 3.
        atom_decoder_heads : int, optional
            The number of heads in the atom decoder, by default 4.
        atom_feature_dim : int, optional
            The atom feature dimension, by default 128.
        conditioning_transition_layers : int, optional
            The number of transition layers for conditioning, by default 2.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        offload_to_cpu : bool, optional
            Whether to offload the activations to CPU, by default False.

        """

        super().__init__()

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_attention_encoder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            atom_encoder_depth=atom_encoder_depth,
            atom_encoder_heads=atom_encoder_heads,
            structure_prediction=True,
            activation_checkpointing=activation_checkpointing,
        )

        self.s_to_a_linear = nn.Sequential(
            nn.LayerNorm(2 * token_s), LinearNoBias(2 * token_s, 2 * token_s)
        )
        init.final_init_(self.s_to_a_linear[1].weight)

        self.token_transformer = DiffusionTransformer(
            dim=2 * token_s,
            dim_single_cond=2 * token_s,
            dim_pairwise=token_z,
            depth=token_transformer_depth,
            heads=token_transformer_heads,
            activation_checkpointing=activation_checkpointing,
            offload_to_cpu=offload_to_cpu,
        )

        self.a_norm = nn.LayerNorm(2 * token_s)

        self.atom_attention_decoder = AtomAttentionDecoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            attn_window_queries=atoms_per_window_queries,
            attn_window_keys=atoms_per_window_keys,
            atom_decoder_depth=atom_decoder_depth,
            atom_decoder_heads=atom_decoder_heads,
            activation_checkpointing=activation_checkpointing,
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        r_noisy,
        times,
        relative_position_encoding,
        feats,
        multiplicity=1,
        model_cache=None,
    ):
        s, normed_fourier = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
        )

        if model_cache is None or len(model_cache) == 0:
            z = self.pairwise_conditioner(
                z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
            )
        else:
            z = None

        # Compute Atom Attention Encoder and aggregation to coarse-grained tokens
        a, q_skip, c_skip, p_skip, to_keys = self.atom_attention_encoder(
            feats=feats,
            s_trunk=s_trunk,
            z=z,
            r=r_noisy,
            multiplicity=multiplicity,
            model_cache=model_cache,
        )

        # Full self-attention on token level
        a = a + self.s_to_a_linear(s)

        mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0)
        a = self.token_transformer(
            a,
            mask=mask.float(),
            s=s,
            z=z,  # note z is not expanded with multiplicity until after bias is computed
            multiplicity=multiplicity,
            model_cache=model_cache,
        )
        a = self.a_norm(a)

        # Broadcast token activations to atoms and run Sequence-local Atom Attention
        r_update = self.atom_attention_decoder(
            a=a,
            q=q_skip,
            c=c_skip,
            p=p_skip,
            feats=feats,
            multiplicity=multiplicity,
            to_keys=to_keys,
            model_cache=model_cache,
        )

        return {"r_update": r_update, "token_a": a.detach()}


class OutTokenFeatUpdate(Module):
    """Output token feature update"""

    def __init__(
        self,
        sigma_data: float,
        token_s=384,
        dim_fourier=256,
    ):
        """Initialize the Output token feature update for confidence model.

        Parameters
        ----------
        sigma_data : float
            The standard deviation of the data distribution.
        token_s : int, optional
            The token dimension, by default 384.
        dim_fourier : int, optional
            The dimension of the fourier embedding, by default 256.

        """

        super().__init__()
        self.sigma_data = sigma_data

        self.norm_next = nn.LayerNorm(2 * token_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.transition_block = ConditionedTransitionBlock(
            2 * token_s, 2 * token_s + dim_fourier
        )

    def forward(
        self,
        times,
        acc_a,
        next_a,
    ):
        next_a = self.norm_next(next_a)
        fourier_embed = self.fourier_embed(times)
        normed_fourier = (
            self.norm_fourier(fourier_embed)
            .unsqueeze(1)
            .expand(-1, next_a.shape[1], -1)
        )
        cond_a = torch.cat((acc_a, normed_fourier), dim=-1)

        acc_a = acc_a + self.transition_block(next_a, cond_a)

        return acc_a


class AtomDiffusion(Module):
    """Atom diffusion module"""

    def __init__(
        self,
        score_model_args,
        num_sampling_steps=5,
        sigma_min=0.0004,
        sigma_max=160.0,
        sigma_data=16.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.5,
        gamma_0=0.8,
        gamma_min=1.0,
        noise_scale=1.003,
        step_scale=1.5,
        coordinate_augmentation=True,
        compile_score=False,
        alignment_reverse_diff=False,
        synchronize_sigmas=False,
        use_inference_model_cache=False,
        accumulate_token_repr=False,
        density_map=None,
        aligned_model=None,
        global_scale=(0.25, 0.05),
        global_steps=(101, 150),
        local_scale=(0.5, 0.5),
        local_steps=(151, 175),
        res=2.0,
        thresh=0.0,
        dust=5,
        cloud_size=0.25,
        voxel_batch=32768,
        # Multi-conformation parameters
        density_maps=None,
        aligned_models=None,
        coupling_base_strength=1.0,
        coupling_min_strength=0.01,
        coupling_schedule='cosine',
        hierarchy_transitions=None,
        hierarchy_groups=None,
        hierarchy_group_strategy='pairs',
        n_path_intermediates=10,
        path_refinement_steps=5,
        path_blend_weight=0.7,
        **kwargs,
    ):
        """Initialize the atom diffusion module.

        Parameters
        ----------
        score_model_args : dict
            The arguments for the score model.
        num_sampling_steps : int, optional
            The number of sampling steps, by default 5.
        sigma_min : float, optional
            The minimum sigma value, by default 0.0004.
        sigma_max : float, optional
            The maximum sigma value, by default 160.0.
        sigma_data : float, optional
            The standard deviation of the data distribution, by default 16.0.
        rho : int, optional
            The rho value, by default 7.
        P_mean : float, optional
            The mean value of P, by default -1.2.
        P_std : float, optional
            The standard deviation of P, by default 1.5.
        gamma_0 : float, optional
            The gamma value, by default 0.8.
        gamma_min : float, optional
            The minimum gamma value, by default 1.0.
        noise_scale : float, optional
            The noise scale, by default 1.003.
        step_scale : float, optional
            The step scale, by default 1.5.
        coordinate_augmentation : bool, optional
            Whether to use coordinate augmentation, by default True.
        compile_score : bool, optional
            Whether to compile the score model, by default False.
        alignment_reverse_diff : bool, optional
            Whether to use alignment reverse diff, by default False.
        synchronize_sigmas : bool, optional
            Whether to synchronize the sigmas, by default False.
        use_inference_model_cache : bool, optional
            Whether to use the inference model cache, by default False.
        accumulate_token_repr : bool, optional
            Whether to accumulate the token representation, by default False.
        density_map : str, optional
            MRC file of cryo-EM map for guidance (single-map mode).
        aligned_model : str, optional
            PDB/CIF file of a map-aligned structure for sample alignment
            during guidance.
        global_scale : tuple(float, float), optional
            Starting and ending guidance strength during global guidance.
            By default 0.25 to 0.05.
        global_steps : tuple(int, int), optional
            1-indexed diffusion timesteps for global guidance.
            By default 101 to 150.
        local_scale : tuple(float, float), optional
            Starting and ending guidance strength during local guidance.
            By default 0.5 to 0.5.
        local_steps : tuple(int, int), optional
            1-indexed diffusion timesteps for local guidance.
            By default 151 to 175.
        res : float, optional
            The nominal resolution of the input map. By default, 2.0.
        thresh : float, optional
            Map values below this value will be zeroed in global guidance.
            By default, 0.0.
        dust : int, optional
            Connected components of the map below this size will be removed
            during global guidance. By default, 5.
        cloud_size: float, optional
            Scaling factor for size of point cloud during global guidance.
        voxel_batch: int, optional
            Number of voxels to process simultaneously during local guidance.
        density_maps : list[str], optional
            List of MRC files for multi-conformation joint sampling.
        aligned_models : list[str], optional
            List of aligned CIF models, one per density map.
        coupling_base_strength : float, optional
            Maximum coupling strength for rigid regions. By default, 1.0.
        coupling_min_strength : float, optional
            Minimum coupling strength for flexible regions. By default, 0.01.
        coupling_schedule : str, optional
            How coupling decays over time: 'cosine', 'linear', 'constant'.
        hierarchy_transitions : list[int], optional
            Step indices for hierarchy level transitions.
        hierarchy_groups : list[list[int]], optional
            Conformation groupings for Level 1 coupling.
        hierarchy_group_strategy : str, optional
            Default grouping strategy if groups not specified.
        n_path_intermediates : int, optional
            Number of path intermediates for path inference.
        path_refinement_steps : int, optional
            Number of refinement iterations for path inference.
        path_blend_weight : float, optional
            Score network trust weight for path inference.
        """
        super().__init__()
        self.score_model = DiffusionModule(
            **score_model_args,
        )
        if compile_score:
            self.score_model = torch.compile(
                self.score_model, dynamic=False, fullgraph=False
            )

        # parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampling_steps = num_sampling_steps
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.noise_scale = noise_scale
        self.step_scale = step_scale
        self.coordinate_augmentation = coordinate_augmentation
        self.alignment_reverse_diff = alignment_reverse_diff
        self.synchronize_sigmas = synchronize_sigmas
        self.use_inference_model_cache = use_inference_model_cache

        self.accumulate_token_repr = accumulate_token_repr
        self.token_s = score_model_args["token_s"]
        if self.accumulate_token_repr:
            self.out_token_feat_update = OutTokenFeatUpdate(
                sigma_data=sigma_data,
                token_s=score_model_args["token_s"],
                dim_fourier=score_model_args["dim_fourier"],
            )

        self.density_map = density_map
        self.aligned_model = aligned_model
        self.guidance_scales = {
            'global': global_scale,
            'local': local_scale
        }
        self.guidance_steps = {
            'global': global_steps,
            'local': local_steps
        }
        self.map_params = {
            'res': res,
            'thresh': thresh,
            'dust': dust,
            'cloud_size': cloud_size,
            'voxel_batch': voxel_batch
        }

        # Multi-conformation parameters
        self.density_maps = density_maps
        self.aligned_models = aligned_models
        self.coupling_base_strength = coupling_base_strength
        self.coupling_min_strength = coupling_min_strength
        self.coupling_schedule = coupling_schedule
        self.hierarchy_transitions = hierarchy_transitions
        self.hierarchy_groups = hierarchy_groups
        self.hierarchy_group_strategy = hierarchy_group_strategy
        self.n_path_intermediates = n_path_intermediates
        self.path_refinement_steps = path_refinement_steps
        self.path_blend_weight = path_blend_weight

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return next(self.score_model.parameters()).device

    def c_skip(self, sigma):
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        return sigma * self.sigma_data / torch.sqrt(self.sigma_data**2 + sigma**2)

    def c_in(self, sigma):
        return 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        return log(sigma / self.sigma_data) * 0.25

    def preconditioned_network_forward(
        self,
        noised_atom_coords,
        sigma,
        network_condition_kwargs: dict,
        training: bool = True,
    ):
        batch, device = noised_atom_coords.shape[0], noised_atom_coords.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1")

        net_out = self.score_model(
            r_noisy=self.c_in(padded_sigma) * noised_atom_coords,
            times=self.c_noise(sigma),
            **network_condition_kwargs,
        )

        denoised_coords = (
            self.c_skip(padded_sigma) * noised_atom_coords
            + self.c_out(padded_sigma) * net_out["r_update"]
        )
        return denoised_coords, net_out["token_a"]

    def sample_schedule(self, num_sampling_steps=None):
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        inv_rho = 1 / self.rho

        steps = torch.arange(
            num_sampling_steps, device=self.device, dtype=torch.float32
        )
        sigmas = (
            self.sigma_max**inv_rho
            + steps
            / (num_sampling_steps - 1)
            * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = sigmas * self.sigma_data

        sigmas = F.pad(sigmas, (0, 1), value=0.0)  # last step is sigma value of 0.
        return sigmas

    def sample(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        train_accumulate_token_repr=False,
        **network_condition_kwargs,
    ):
        print("Started structure sampling")
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        shape = (*atom_mask.shape, 3)

        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # initialize guidance
        guidance = (self.density_map is not None)
        if guidance:
            guidance_inputs = {'density_map': self.density_map, 'aligned_model': self.aligned_model, 
                                'map_params': self.map_params}
            guidance_inputs['n_atoms'] = (atom_mask[0] > 0).sum().item()
            guidance_inputs['pad'] = len(atom_mask[0]) - guidance_inputs['n_atoms']
            guidance_inputs['sequence'] = network_condition_kwargs['feats']['res_type'][0].argmax(1).tolist()
            guidance_types = [k for k, v in self.guidance_steps.items() if v is not None]
            guidance_losses = {k: -torch.ones((num_sampling_steps, multiplicity), dtype=torch.float32) for k in guidance_types}
            guidance_params = init_guidance(guidance_types, guidance_inputs, self.device)
            guidance_start = min([self.guidance_steps[k][0] for k in guidance_types])
            guidance_schedules = schedule_guidance(self.guidance_steps, self.guidance_scales)

        # atom position is noise at the beginning
        init_sigma = sigmas[0]
        atom_coords = init_sigma * torch.randn(shape, device=self.device)
        atom_coords_denoised = None
        model_cache = {} if self.use_inference_model_cache else None

        token_repr = None
        token_a = None
        denoised_traj = torch.zeros((num_sampling_steps, *shape), dtype=torch.float32)

        # gradually denoise
        for i, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            # random rotation and translation of input coords to denoiser
            atom_coords, augment_params = center_random_augmentation(
                atom_coords,
                atom_mask,
                augmentation=True,
                centering=True,
                ret_transform=True
            )

            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()

            # add noise to sample
            t_hat = sigma_tm * (1 + gamma)
            eps = (
                self.noise_scale
                * sqrt(t_hat**2 - sigma_tm**2)
                * torch.randn(shape, device=self.device)
            )
            atom_coords_noisy = atom_coords + eps

            # activate guidance terms for this step
            if guidance:
                guidance_current = []
                for k in guidance_types:
                    if self.guidance_steps[k][0] <= i+1 <= self.guidance_steps[k][1]:
                        guidance_current.append(k)
                    if self.guidance_steps[k][0] == i+1:
                        print(f"Started {k} guidance")
            guidance_this_step = (len(guidance_current) > 0) if guidance else False

            with torch.set_grad_enabled(guidance_this_step):
                # denoise the noisy coordinates
                if guidance_this_step: atom_coords_noisy.requires_grad_(True)
                atom_coords_denoised, token_a = self.preconditioned_network_forward(
                    atom_coords_noisy,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        multiplicity=multiplicity,
                        model_cache=model_cache,
                        **network_condition_kwargs,
                    ),
                )

                # aligned noisy and denoised coordinates
                if self.alignment_reverse_diff:
                    with torch.autocast("cuda", enabled=False):
                        atom_coords_denoised = weighted_rigid_align(
                            atom_coords_denoised.float(),
                            atom_coords_noisy.clone().float(),
                            atom_mask.float(),
                            atom_mask.float(),
                            differentiable=guidance_this_step
                        )

            # compute unguided score term
            with torch.no_grad():
                denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat

            # compute inputs to confidence module
            if self.accumulate_token_repr:
                if token_repr is None:
                    token_repr = torch.zeros_like(token_a)

                with torch.set_grad_enabled(train_accumulate_token_repr):
                    sigma = torch.full(
                        (atom_coords_denoised.shape[0],),
                        t_hat,
                        device=atom_coords_denoised.device,
                    )
                    token_repr = self.out_token_feat_update(
                        times=self.c_noise(sigma), acc_a=token_repr, next_a=token_a
                    )

            # guidance step
            with torch.set_grad_enabled(guidance_this_step):
                # align denoised sample to the input structure when guidance starts, 
                # afterwards undo the random transformations
                if guidance:
                    if i+1 == guidance_start:
                        atom_coords_denoised, align_params = weighted_rigid_align(
                            atom_coords_denoised.float(),
                            guidance_params['ref_coords'].expand(*shape),
                            atom_mask.float(),
                            guidance_params['ref_mask'].expand(multiplicity, -1).int(),
                            ret_transform=True,
                            differentiable=True
                        )
                    elif i+1 > guidance_start:
                        atom_coords_denoised = apply_transform(atom_coords_denoised, augment_params, invert=True)

                # compute guidance term, rescale magnitude, and add to the diffusion update
                if guidance_this_step:
                    guidance_params['curr_step'] = i 
                    unguided_norm = torch.linalg.matrix_norm(denoised_over_sigma, keepdim=True).expand(*shape)
                    for guidance_type in guidance_current:
                        loss = compute_guidance(atom_coords_denoised, atom_mask, guidance_type, guidance_params)
                        score = atom_coords_noisy.grad
                        guided_norm = torch.linalg.matrix_norm(score, keepdim=True).expand(*shape)
                        update_mask = (guided_norm > 0.)
                        score[update_mask] *= unguided_norm[update_mask] / guided_norm[update_mask]
                        guidance_scale = guidance_schedules[guidance_type][i]
                        denoised_over_sigma += guidance_scale * score
                        guidance_losses[guidance_type][i] = loss.detach().cpu()
                    atom_coords_noisy.requires_grad_(False)

            # take diffusion step on samples
            atom_coords = (
                atom_coords_noisy
                + self.step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )

            # bring noisy samples into alignment with map
            if guidance:
                if i+1 == guidance_start:    
                    atom_coords = apply_transform(atom_coords, align_params)
                elif i+1 > guidance_start:
                    atom_coords = apply_transform(atom_coords, augment_params, invert=True)
            
            denoised_traj[i, ...] = atom_coords_denoised.detach().cpu()

        # move samples back to original centering
        if guidance and 'ref_center' in guidance_params:
            atom_coords += guidance_params['ref_center'].to(atom_coords)
            denoised_traj += guidance_params['ref_center'].to(denoised_traj)
        
        print("Finished structure sampling")
        outputs = {
            'sample_atom_coords': atom_coords, 
            'diff_token_repr': token_repr,
            'denoised_traj': denoised_traj,
        }
        if guidance: outputs['guidance_loss'] = guidance_losses
        return outputs

    def sample_joint(
        self,
        atom_mask,
        num_sampling_steps=None,
        multiplicity=1,
        **network_condition_kwargs,
    ):
        """Joint sampling of N conformations with coupling and hierarchical guidance.

        Implements the Bayesian joint posterior sampling framework:
        For each conformation i, the score update includes:
          1. Boltz-1 prior score (denoiser)
          2. Density guidance for map y_i (CryoBoltz)
          3. Coupling term: sum_{j!=i} Lambda_{ij} (x_j - x_i)

        The coupling strength varies over time via the hierarchical schedule:
          Level 0 (early): all conformations coupled
          Level 1 (middle): group-based coupling
          Level 2 (late): independent sampling

        Parameters
        ----------
        atom_mask : torch.Tensor
            Atom mask, shape (batch, n_atoms).
        num_sampling_steps : int, optional
            Number of diffusion steps.
        multiplicity : int, optional
            Number of samples per conformation.
        **network_condition_kwargs : dict
            Arguments for the score network (s_trunk, z_trunk, etc.).

        Returns
        -------
        dict
            'sample_atom_coords': list of N tensors, each (multiplicity, n_atoms, 3)
            'diff_token_repr': token representation (from last conformation)
            'denoised_traj': list of N trajectory tensors
            'guidance_loss': dict of per-conformation guidance losses
            'coupling_history': per-step coupling strengths
        """
        assert self.density_maps is not None, \
            "sample_joint requires density_maps (list of MRC paths)"
        assert self.aligned_models is not None, \
            "sample_joint requires aligned_models (list of CIF paths)"

        n_conf = len(self.density_maps)
        assert len(self.aligned_models) == n_conf, \
            "Number of density maps and aligned models must match"

        print(f"Started joint structure sampling for {n_conf} conformations")
        num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
        atom_mask_expanded = atom_mask.repeat_interleave(multiplicity, 0)
        shape = (*atom_mask_expanded.shape, 3)

        # Build sigma schedule (shared across all conformations)
        sigmas = self.sample_schedule(num_sampling_steps)
        gammas = torch.where(sigmas > self.gamma_min, self.gamma_0, 0.0)
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[1:]))

        # Initialize per-conformation guidance
        guidance_types = [k for k, v in self.guidance_steps.items() if v is not None]
        guidance_schedules = schedule_guidance(self.guidance_steps, self.guidance_scales)
        guidance_start = min([self.guidance_steps[k][0] for k in guidance_types])

        # Build per-conformation guidance inputs
        n_atoms = (atom_mask_expanded[0] > 0).sum().item()
        pad = len(atom_mask_expanded[0]) - n_atoms
        sequence = network_condition_kwargs['feats']['res_type'][0].argmax(1).tolist()

        inputs_list = []
        for idx in range(n_conf):
            inputs = {
                'density_map': self.density_maps[idx],
                'aligned_model': self.aligned_models[idx],
                'map_params': self.map_params,
                'n_atoms': n_atoms,
                'pad': pad,
                'sequence': sequence,
            }
            inputs_list.append(inputs)

        # Initialize multi-conformation guidance
        all_guidance_params, shared_params = init_multi_guidance(
            guidance_types, inputs_list, self.device
        )

        # Estimate flexibility from reference structures for coupling
        flexibility = estimate_flexibility_from_coords(
            shared_params['ref_coords_list'],
            shared_params['ref_masks_list'],
        )
        coupling_strength = build_coupling_strength(
            flexibility,
            base_strength=self.coupling_base_strength,
            min_strength=self.coupling_min_strength,
        )

        # Build hierarchical coupling schedule
        groups = self.hierarchy_groups
        if groups is None and n_conf > 2:
            groups = build_default_groups(n_conf, self.hierarchy_group_strategy)

        hierarchy = HierarchicalCouplingSchedule(
            n_conformations=n_conf,
            num_steps=num_sampling_steps,
            level_transitions=self.hierarchy_transitions,
            groups=groups,
            strength_schedule=self.coupling_schedule,
            base_strength=self.coupling_base_strength,
        )

        # Initialize per-conformation state
        init_sigma = sigmas[0]
        all_coords = [
            init_sigma * torch.randn(shape, device=self.device)
            for _ in range(n_conf)
        ]
        all_denoised_traj = [
            torch.zeros((num_sampling_steps, *shape), dtype=torch.float32)
            for _ in range(n_conf)
        ]
        all_guidance_losses = [
            {k: -torch.ones((num_sampling_steps, multiplicity), dtype=torch.float32)
             for k in guidance_types}
            for _ in range(n_conf)
        ]
        coupling_history = torch.zeros(num_sampling_steps)

        token_repr = None

        # Main diffusion loop
        for i, (sigma_tm, sigma_t, gamma) in enumerate(sigmas_and_gammas):
            sigma_tm, sigma_t, gamma = sigma_tm.item(), sigma_t.item(), gamma.item()
            t_hat = sigma_tm * (1 + gamma)

            # Get coupling info for this step
            partners_info = []
            for c in range(n_conf):
                p, s = hierarchy.get_active_partners(i, c)
                partners_info.append((p, s))
            coupling_history[i] = partners_info[0][1] if partners_info else 0.0

            # Determine if any guidance is active
            guidance_current = []
            for k in guidance_types:
                if self.guidance_steps[k][0] <= i + 1 <= self.guidance_steps[k][1]:
                    guidance_current.append(k)
                    if self.guidance_steps[k][0] == i + 1:
                        print(f"Started {k} guidance at step {i+1}")
            guidance_this_step = len(guidance_current) > 0

            # Process each conformation
            all_denoised_over_sigma = []
            all_augment_params = []
            all_align_params = [None] * n_conf

            for c in range(n_conf):
                atom_coords = all_coords[c]

                # Random rotation and translation
                atom_coords, augment_params = center_random_augmentation(
                    atom_coords, atom_mask_expanded,
                    augmentation=True, centering=True, ret_transform=True
                )
                all_augment_params.append(augment_params)

                # Add noise
                eps = (
                    self.noise_scale
                    * sqrt(t_hat ** 2 - sigma_tm ** 2)
                    * torch.randn(shape, device=self.device)
                )
                atom_coords_noisy = atom_coords + eps

                with torch.set_grad_enabled(guidance_this_step):
                    if guidance_this_step:
                        atom_coords_noisy.requires_grad_(True)

                    # Denoise via score network (shared Boltz-1 model)
                    model_cache = {} if self.use_inference_model_cache else None
                    atom_coords_denoised, token_a = self.preconditioned_network_forward(
                        atom_coords_noisy, t_hat,
                        training=False,
                        network_condition_kwargs=dict(
                            multiplicity=multiplicity,
                            model_cache=model_cache,
                            **network_condition_kwargs,
                        ),
                    )

                    # Alignment
                    if self.alignment_reverse_diff:
                        with torch.autocast("cuda", enabled=False):
                            atom_coords_denoised = weighted_rigid_align(
                                atom_coords_denoised.float(),
                                atom_coords_noisy.clone().float(),
                                atom_mask_expanded.float(),
                                atom_mask_expanded.float(),
                                differentiable=guidance_this_step
                            )

                # Compute unguided score
                with torch.no_grad():
                    denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat

                # Guidance alignment
                gp = all_guidance_params[c]
                with torch.set_grad_enabled(guidance_this_step):
                    if i + 1 == guidance_start:
                        atom_coords_denoised, align_p = weighted_rigid_align(
                            atom_coords_denoised.float(),
                            gp['ref_coords'].expand(*shape),
                            atom_mask_expanded.float(),
                            gp['ref_mask'].expand(multiplicity, -1).int(),
                            ret_transform=True, differentiable=True
                        )
                        all_align_params[c] = align_p
                    elif i + 1 > guidance_start:
                        atom_coords_denoised = apply_transform(
                            atom_coords_denoised, augment_params, invert=True
                        )

                    # Density guidance (per-conformation map)
                    if guidance_this_step:
                        gp['curr_step'] = i
                        unguided_norm = torch.linalg.matrix_norm(
                            denoised_over_sigma, keepdim=True
                        ).expand(*shape)
                        for guidance_type in guidance_current:
                            loss = compute_guidance(
                                atom_coords_denoised, atom_mask_expanded,
                                guidance_type, gp
                            )
                            score = atom_coords_noisy.grad
                            guided_norm = torch.linalg.matrix_norm(
                                score, keepdim=True
                            ).expand(*shape)
                            update_mask = (guided_norm > 0.)
                            score[update_mask] *= (
                                unguided_norm[update_mask] / guided_norm[update_mask]
                            )
                            guidance_scale = guidance_schedules[guidance_type][i]
                            denoised_over_sigma += guidance_scale * score
                            all_guidance_losses[c][guidance_type][i] = loss.detach().cpu()
                        atom_coords_noisy.requires_grad_(False)

                all_denoised_over_sigma.append(denoised_over_sigma)
                all_coords[c] = atom_coords_noisy  # temporarily store noisy coords

                # Store denoised trajectory
                all_denoised_traj[c][i, ...] = atom_coords_denoised.detach().cpu()

            # Apply coupling gradients (Contribution 1 & 2)
            # Uses all_denoised_traj at step i for the coupling targets
            for c in range(n_conf):
                active_partners, coupling_mult = partners_info[c]
                if coupling_mult > 0 and len(active_partners) > 0:
                    # Use denoised coordinates for coupling
                    denoised_list = [
                        all_denoised_traj[j][i].to(self.device)
                        for j in range(n_conf)
                    ]
                    coupling_grad = compute_coupling_gradient(
                        denoised_list, coupling_strength.to(self.device),
                        c, atom_mask_expanded, active_indices=active_partners
                    )
                    # Scale coupling gradient to match score magnitude
                    coupling_norm = torch.linalg.matrix_norm(
                        coupling_grad, keepdim=True
                    ).expand(*shape)
                    score_norm = torch.linalg.matrix_norm(
                        all_denoised_over_sigma[c], keepdim=True
                    ).expand(*shape)
                    norm_mask = (coupling_norm > 0.)
                    if norm_mask.any():
                        coupling_grad[norm_mask] *= (
                            score_norm[norm_mask] / coupling_norm[norm_mask]
                        )
                    all_denoised_over_sigma[c] += coupling_mult * coupling_grad

            # Take diffusion step for all conformations
            for c in range(n_conf):
                atom_coords_noisy = all_coords[c]
                all_coords[c] = (
                    atom_coords_noisy
                    + self.step_scale * (sigma_t - t_hat) * all_denoised_over_sigma[c]
                )

                # Bring samples into map alignment
                if i + 1 == guidance_start and all_align_params[c] is not None:
                    all_coords[c] = apply_transform(all_coords[c], all_align_params[c])
                elif i + 1 > guidance_start:
                    all_coords[c] = apply_transform(
                        all_coords[c], all_augment_params[c], invert=True
                    )

        # Move back to original centering
        for c in range(n_conf):
            ref_center = shared_params['ref_center'].to(all_coords[c])
            all_coords[c] += ref_center
            all_denoised_traj[c] += ref_center.to(all_denoised_traj[c])

        print("Finished joint structure sampling")

        outputs = {
            'sample_atom_coords': all_coords[0],  # primary for compatibility
            'all_sample_atom_coords': all_coords,
            'diff_token_repr': token_repr,
            'denoised_traj': all_denoised_traj[0],
            'all_denoised_traj': all_denoised_traj,
            'guidance_loss': {
                f'conf_{c}': all_guidance_losses[c] for c in range(n_conf)
            },
            'coupling_history': coupling_history,
            'n_conformations': n_conf,
        }
        return outputs

    def loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / ((sigma * self.sigma_data) ** 2)

    def noise_distribution(self, batch_size):
        return (
            self.sigma_data
            * (
                self.P_mean
                + self.P_std * torch.randn((batch_size,), device=self.device)
            ).exp()
        )

    def forward(
        self,
        s_inputs,
        s_trunk,
        z_trunk,
        relative_position_encoding,
        feats,
        multiplicity=1,
    ):
        # training diffusion step
        batch_size = feats["coords"].shape[0]

        if self.synchronize_sigmas:
            sigmas = self.noise_distribution(batch_size).repeat_interleave(
                multiplicity, 0
            )
        else:
            sigmas = self.noise_distribution(batch_size * multiplicity)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1")

        atom_coords = feats["coords"]
        B, N, L = atom_coords.shape[0:3]
        atom_coords = atom_coords.reshape(B * N, L, 3)
        atom_coords = atom_coords.repeat_interleave(multiplicity // N, 0)
        feats["coords"] = atom_coords

        atom_mask = feats["atom_pad_mask"]
        atom_mask = atom_mask.repeat_interleave(multiplicity, 0)

        atom_coords = center_random_augmentation(
            atom_coords, atom_mask, augmentation=self.coordinate_augmentation
        )

        noise = torch.randn_like(atom_coords)
        noised_atom_coords = atom_coords + padded_sigmas * noise

        denoised_atom_coords, _ = self.preconditioned_network_forward(
            noised_atom_coords,
            sigmas,
            training=True,
            network_condition_kwargs=dict(
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                relative_position_encoding=relative_position_encoding,
                feats=feats,
                multiplicity=multiplicity,
            ),
        )

        return dict(
            noised_atom_coords=noised_atom_coords,
            denoised_atom_coords=denoised_atom_coords,
            sigmas=sigmas,
            aligned_true_atom_coords=atom_coords,
        )

    def compute_loss(
        self,
        feats,
        out_dict,
        add_smooth_lddt_loss=True,
        nucleotide_loss_weight=5.0,
        ligand_loss_weight=10.0,
        multiplicity=1,
    ):
        denoised_atom_coords = out_dict["denoised_atom_coords"]
        noised_atom_coords = out_dict["noised_atom_coords"]
        sigmas = out_dict["sigmas"]

        resolved_atom_mask = feats["atom_resolved_mask"]
        resolved_atom_mask = resolved_atom_mask.repeat_interleave(multiplicity, 0)

        align_weights = noised_atom_coords.new_ones(noised_atom_coords.shape[:2])
        atom_type = (
            torch.bmm(
                feats["atom_to_token"].float(), feats["mol_type"].unsqueeze(-1).float()
            )
            .squeeze(-1)
            .long()
        )
        atom_type_mult = atom_type.repeat_interleave(multiplicity, 0)

        align_weights = align_weights * (
            1
            + nucleotide_loss_weight
            * (
                torch.eq(atom_type_mult, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type_mult, const.chain_type_ids["RNA"]).float()
            )
            + ligand_loss_weight
            * torch.eq(atom_type_mult, const.chain_type_ids["NONPOLYMER"]).float()
        )

        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            atom_coords = out_dict["aligned_true_atom_coords"]
            atom_coords_aligned_ground_truth = weighted_rigid_align(
                atom_coords.detach().float(),
                denoised_atom_coords.detach().float(),
                align_weights.detach().float(),
                mask=resolved_atom_mask.detach().float(),
            )

        # Cast back
        atom_coords_aligned_ground_truth = atom_coords_aligned_ground_truth.to(
            denoised_atom_coords
        )

        # weighted MSE loss of denoised atom positions
        mse_loss = ((denoised_atom_coords - atom_coords_aligned_ground_truth) ** 2).sum(
            dim=-1
        )
        mse_loss = torch.sum(
            mse_loss * align_weights * resolved_atom_mask, dim=-1
        ) / torch.sum(3 * align_weights * resolved_atom_mask, dim=-1)

        # weight by sigma factor
        loss_weights = self.loss_weight(sigmas)
        mse_loss = (mse_loss * loss_weights).mean()

        total_loss = mse_loss

        # proposed auxiliary smooth lddt loss
        lddt_loss = self.zero
        if add_smooth_lddt_loss:
            lddt_loss = smooth_lddt_loss(
                denoised_atom_coords,
                feats["coords"],
                torch.eq(atom_type, const.chain_type_ids["DNA"]).float()
                + torch.eq(atom_type, const.chain_type_ids["RNA"]).float(),
                coords_mask=feats["atom_resolved_mask"],
                multiplicity=multiplicity,
            )

            total_loss = total_loss + lddt_loss

        loss_breakdown = dict(
            mse_loss=mse_loss,
            smooth_lddt_loss=lddt_loss,
        )

        return dict(loss=total_loss, loss_breakdown=loss_breakdown)
