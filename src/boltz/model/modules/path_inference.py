"""Physics-constrained path inference between conformations.

After joint sampling produces N structures, this module infers physically
plausible transition paths between them using the Boltz-1 score function
as a manifold constraint.

Key idea: Instead of linear interpolation (which produces steric clashes),
we use iterative noise-denoise steps with the Boltz-1 score network to
project intermediate structures onto the "manifold of plausible protein
structures".

Algorithm:
1. Initialize path as linear interpolation between endpoints
2. For each intermediate:
   a. Add noise at level sigma
   b. Denoise using Boltz-1 score network
   c. Blend with interpolated target to maintain path continuity
3. Iterate with decreasing sigma for progressive refinement
"""

import torch
import numpy as np
from math import sqrt
from typing import Optional

from boltz.model.loss.diffusion import weighted_rigid_align


def linear_interpolation(coords_start, coords_end, n_intermediates):
    """Generate linearly interpolated intermediates between two structures.

    Parameters
    ----------
    coords_start : torch.Tensor
        Starting structure coordinates, shape (n_atoms, 3).
    coords_end : torch.Tensor
        Ending structure coordinates, shape (n_atoms, 3).
    n_intermediates : int
        Number of intermediate structures (excluding endpoints).

    Returns
    -------
    torch.Tensor
        Interpolated coordinates, shape (n_intermediates, n_atoms, 3).
    """
    alphas = torch.linspace(0, 1, n_intermediates + 2, device=coords_start.device)[1:-1]
    intermediates = []
    for alpha in alphas:
        interp = (1 - alpha) * coords_start + alpha * coords_end
        intermediates.append(interp)
    return torch.stack(intermediates, dim=0)


def score_denoise_step(coords, sigma, score_model, network_kwargs, sigma_data=16.0):
    """Perform one noise-then-denoise step to project onto protein manifold.

    Parameters
    ----------
    coords : torch.Tensor
        Current coordinates, shape (batch, n_atoms, 3).
    sigma : float
        Noise level.
    score_model : callable
        The preconditioned score network forward function.
    network_kwargs : dict
        Additional arguments for the score network.
    sigma_data : float
        Data distribution standard deviation.

    Returns
    -------
    torch.Tensor
        Denoised coordinates, shape (batch, n_atoms, 3).
    """
    device = coords.device
    noise = sigma * torch.randn_like(coords)
    noised_coords = coords + noise

    with torch.no_grad():
        denoised, _ = score_model(
            noised_coords,
            sigma,
            training=False,
            network_condition_kwargs=network_kwargs,
        )

    return denoised


def infer_path(
    structures,
    atom_mask,
    score_model,
    network_kwargs,
    n_intermediates=10,
    refinement_steps=5,
    sigma_schedule=None,
    blend_weight=0.7,
    sigma_data=16.0,
):
    """Infer physically plausible transition paths between structures.

    Parameters
    ----------
    structures : list[torch.Tensor]
        List of N structure coordinates, each shape (n_atoms, 3).
        These are the endpoints from joint sampling.
    atom_mask : torch.Tensor
        Atom mask, shape (n_atoms,).
    score_model : callable
        The preconditioned score network.
    network_kwargs : dict
        Arguments for the score network (trunk representations, etc.).
    n_intermediates : int
        Number of intermediate structures per pair.
    refinement_steps : int
        Number of noise-denoise refinement iterations.
    sigma_schedule : list[float], optional
        Noise levels for each refinement step (decreasing).
        If None, uses geometric schedule from sigma_data/4 to sigma_data/64.
    blend_weight : float
        Weight for blending denoised result with interpolation target.
        Higher values trust the score network more.
    sigma_data : float
        Data distribution standard deviation.

    Returns
    -------
    dict with keys:
        'paths': list of torch.Tensor, one per consecutive pair
            Each shape (n_intermediates + 2, n_atoms, 3) including endpoints.
        'full_path': torch.Tensor
            Complete path through all structures, shape (total_points, n_atoms, 3).
    """
    n_structures = len(structures)
    device = structures[0].device

    if sigma_schedule is None:
        # Geometric schedule: decreasing noise levels
        sigma_start = sigma_data / 4
        sigma_end = sigma_data / 64
        sigma_schedule = [
            sigma_start * (sigma_end / sigma_start) ** (i / max(refinement_steps - 1, 1))
            for i in range(refinement_steps)
        ]

    all_paths = []
    full_path_coords = [structures[0]]

    for pair_idx in range(n_structures - 1):
        start = structures[pair_idx]  # (n_atoms, 3)
        end = structures[pair_idx + 1]  # (n_atoms, 3)

        # Align end to start for smoother interpolation
        start_batch = start.unsqueeze(0)  # (1, n_atoms, 3)
        end_batch = end.unsqueeze(0)  # (1, n_atoms, 3)
        mask_batch = atom_mask.unsqueeze(0).float()
        end_aligned = weighted_rigid_align(
            end_batch, start_batch, mask_batch, mask_batch
        ).squeeze(0)

        # Initialize with linear interpolation
        intermediates = linear_interpolation(start, end_aligned, n_intermediates)
        # intermediates: (n_intermediates, n_atoms, 3)

        # Iterative refinement via noise-denoise
        for step, sigma in enumerate(sigma_schedule):
            # Add noise and denoise each intermediate
            denoised = score_denoise_step(
                intermediates, sigma, score_model, network_kwargs, sigma_data
            )

            # Blend denoised result with interpolation target
            # This ensures path continuity while allowing physical refinement
            alphas = torch.linspace(0, 1, n_intermediates + 2, device=device)[1:-1]
            for k in range(n_intermediates):
                target = (1 - alphas[k]) * start + alphas[k] * end_aligned
                intermediates[k] = blend_weight * denoised[k] + (1 - blend_weight) * target

        # Build complete path for this pair
        path = torch.cat([start.unsqueeze(0), intermediates, end_aligned.unsqueeze(0)], dim=0)
        all_paths.append(path)

        # Append intermediates and endpoint to full path
        full_path_coords.append(intermediates)
        if pair_idx < n_structures - 2:
            full_path_coords.append(end_aligned.unsqueeze(0))
        else:
            full_path_coords.append(end_aligned.unsqueeze(0))

    # Concatenate full path
    full_path = torch.cat(full_path_coords, dim=0)

    return {
        'paths': all_paths,
        'full_path': full_path,
    }


def compute_path_energy(path_coords, atom_mask, score_model, network_kwargs,
                        sigma=1.0, sigma_data=16.0):
    """Compute an energy-like score for each point along a path.

    Uses the denoising score magnitude as a proxy for how well each
    intermediate structure lies on the protein manifold.

    Parameters
    ----------
    path_coords : torch.Tensor
        Path coordinates, shape (n_points, n_atoms, 3).
    atom_mask : torch.Tensor
        Atom mask, shape (n_atoms,).
    score_model : callable
        The preconditioned score network.
    network_kwargs : dict
        Arguments for the score network.
    sigma : float
        Noise level for score evaluation.
    sigma_data : float
        Data distribution standard deviation.

    Returns
    -------
    torch.Tensor
        Per-point energy scores, shape (n_points,).
        Lower values indicate more physically plausible structures.
    """
    n_points = path_coords.shape[0]
    device = path_coords.device

    with torch.no_grad():
        # Denoise each point and measure reconstruction error
        noise = sigma * torch.randn_like(path_coords)
        noised = path_coords + noise

        denoised, _ = score_model(
            noised,
            sigma,
            training=False,
            network_condition_kwargs=network_kwargs,
        )

        # Reconstruction error as energy proxy
        mask = atom_mask.float().unsqueeze(-1)  # (n_atoms, 1)
        diff = ((path_coords - denoised) ** 2 * mask).sum(dim=(-1, -2))
        n_valid = mask.sum()
        energies = diff / n_valid

    return energies
