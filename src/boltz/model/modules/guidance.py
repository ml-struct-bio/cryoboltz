import torch
import numpy as np
from geomloss import SamplesLoss
from boltz.model.modules.density_map import MapGenerator
from boltz.model.modules.cloud import kmeans_cloud
from boltz.model.modules.struct_tools import load_cif


# Initialize objects needed for map guidance
def init_guidance(types, inputs, device):
    params = {}

    # load aligned input structure
    coords, mask = load_cif(inputs['aligned_model'], inputs['sequence'], ca_only=True)
    params['ref_coords'] = torch.nn.functional.pad(coords, (0, 0, 0, inputs['pad'])).to(device)
    params['ref_mask'] = torch.nn.functional.pad(mask, (0, inputs['pad'])).to(device)
    params['ref_center'] = params['ref_coords'][params['ref_mask']].mean(0)
    params['ref_coords'][params['ref_mask']] -= params['ref_center']

    # load map
    params['map_generator'] = MapGenerator(
        inputs['density_map'],
        inputs['sequence'],
        nominal_res=inputs['map_params']['res'],
        ref_center=params['ref_center'],
        device=device
    )
    params['density_map'] = torch.tensor(params['map_generator'].density).to(device)
    params['voxel_size'] = params['map_generator'].voxel_size
    params['origin'] = params['map_generator'].origin.to(device)

    # convert map to point cloud
    if 'global' in types:
        cloud_points = int(inputs['map_params']['cloud_size'] * inputs['n_atoms'] // params['voxel_size']**3)
        params['density_cloud'] = kmeans_cloud(
            params['density_map'],
            params['voxel_size'],
            params['origin'],
            params['ref_center'],
            cloud_points,
            weighted=True,
            thresh=inputs['map_params']['thresh'],
            dust=inputs['map_params']['dust'],
        )
        params['cloud_loss'] = SamplesLoss(reach=10)

    if 'local' in types:
        params['vox_batch_size'] = inputs['map_params']['voxel_batch']

    return params


def init_multi_guidance(types, inputs_list, device):
    """Initialize guidance for multiple density maps (multi-conformation mode).

    Parameters
    ----------
    types : list[str]
        Guidance types to initialize ('global', 'local').
    inputs_list : list[dict]
        List of input dicts, one per conformation. Each dict has the same
        structure as the single-map inputs dict.
    device : torch.device
        Device to use.

    Returns
    -------
    list[dict]
        List of guidance params, one per conformation.
    dict
        Shared parameters (common ref_center computed as mean across maps).
    """
    n_conf = len(inputs_list)
    all_params = []

    # First pass: load all reference structures to compute shared center
    ref_coords_list = []
    ref_masks_list = []
    for inputs in inputs_list:
        coords, mask = load_cif(inputs['aligned_model'], inputs['sequence'], ca_only=True)
        coords_padded = torch.nn.functional.pad(coords, (0, 0, 0, inputs['pad'])).to(device)
        mask_padded = torch.nn.functional.pad(mask, (0, inputs['pad'])).to(device)
        ref_coords_list.append(coords_padded)
        ref_masks_list.append(mask_padded)

    # Compute shared reference center (mean of all reference structures)
    all_masked_coords = []
    for coords, mask in zip(ref_coords_list, ref_masks_list):
        all_masked_coords.append(coords[mask].mean(0))
    shared_center = torch.stack(all_masked_coords).mean(0)

    # Second pass: initialize per-conformation guidance with shared center
    for idx, inputs in enumerate(inputs_list):
        params = {}
        params['ref_coords'] = ref_coords_list[idx]
        params['ref_mask'] = ref_masks_list[idx]
        params['ref_center'] = shared_center
        params['ref_coords'][params['ref_mask']] -= shared_center

        # Load map with shared center
        params['map_generator'] = MapGenerator(
            inputs['density_map'],
            inputs['sequence'],
            nominal_res=inputs['map_params']['res'],
            ref_center=shared_center,
            device=device
        )
        params['density_map'] = torch.tensor(params['map_generator'].density).to(device)
        params['voxel_size'] = params['map_generator'].voxel_size
        params['origin'] = params['map_generator'].origin.to(device)

        if 'global' in types:
            cloud_points = int(
                inputs['map_params']['cloud_size'] * inputs['n_atoms']
                // params['voxel_size'] ** 3
            )
            params['density_cloud'] = kmeans_cloud(
                params['density_map'],
                params['voxel_size'],
                params['origin'],
                shared_center,
                cloud_points,
                weighted=True,
                thresh=inputs['map_params']['thresh'],
                dust=inputs['map_params']['dust'],
            )
            params['cloud_loss'] = SamplesLoss(reach=10)

        if 'local' in types:
            params['vox_batch_size'] = inputs['map_params']['voxel_batch']

        all_params.append(params)

    shared_params = {
        'ref_center': shared_center,
        'ref_coords_list': ref_coords_list,
        'ref_masks_list': ref_masks_list,
    }

    return all_params, shared_params


# Cosine annealing guidance strength
def schedule_guidance(steps, scales):
    schedules = {}
    for t in steps:
        if steps[t] is None: continue
        start, end = steps[t]
        schedule = torch.zeros(end)
        if scales[t][0] == scales[t][1]:
            schedule[start-1:] = scales[t][0]
        else:
            assert scales[t][0] >= scales[t][1], "Only decreasing scale is supported"
            step_idxs = torch.arange(end - start + 1)
            schedule[start-1:] = scales[t][1] + 0.5 * (scales[t][0]-scales[t][1]) \
                * (1+torch.cos(np.pi*step_idxs/(end-start)))
        schedules[t] = schedule
    return schedules


# Populate guidance gradients and return guidance loss
def compute_guidance(
        atom_coords,
        atom_mask,
        guidance_type,
        guidance_params,
    ):
    if guidance_type=='local':
        nll = density_mse_grad(
            atom_coords,
            guidance_params['density_map'],
            guidance_params['map_generator'], (atom_mask > 0),
            guidance_params['map_generator'].nominal_res,
            guidance_params['vox_batch_size']
        )
    elif guidance_type=='global':
        nll = density_cloud_grad(
            atom_coords,
            guidance_params['density_cloud'],
            (atom_mask > 0),
            guidance_params['cloud_loss']
        )
    return nll


# Global guidance
def density_cloud_grad(pred_coords, ref_cloud, mask, loss_fn):
    nll = torch.zeros(len(pred_coords)).to(pred_coords)
    for i, pred_coord in enumerate(pred_coords):
        nll[i] = loss_fn(pred_coord[mask[i]], ref_cloud)
    nll.mean().backward(retain_graph=True)
    return nll


# Local guidance
def density_mse_grad(pred_coords, ref_map, map_generator, coord_mask, resolution, vox_batch_size):
    # estimate normalization statistics from one predicted map only
    with torch.no_grad():
        pred_map =  map_generator.get_full_density_per_batch(pred_coords[0, coord_mask[0]], vox_batch_size, resolution)[0]
        mean, std = pred_map.mean(), pred_map.std()
        ref_map_flat = ref_map.flatten()
        ref_map_flat = (ref_map_flat - ref_map_flat.mean()) / ref_map_flat.std()

    total_nll = torch.zeros(len(pred_coords)).to(pred_coords).detach()
    total_grad = torch.zeros_like(pred_coords).detach()
    n_voxels = len(map_generator.pix_coords_3d)
    for i, pred_coord in enumerate(pred_coords):
        for b in range(0, n_voxels, vox_batch_size):
            subset = slice(b, b+vox_batch_size)
            pix_coords_subset = map_generator.pix_coords_3d[subset, :]
            pred_submap = map_generator.calculate_potential_one_batch(pred_coord[coord_mask[i]], pix_coords_subset, resolution)
            pred_submap = (pred_submap - mean) / std
            nll = torch.sum((ref_map_flat[subset] - pred_submap)**2)
            total_nll[i] += nll / n_voxels
            total_grad[i, ...] += torch.autograd.grad(nll, pred_coord)[0]
    pred_coords.backward(gradient=total_grad, retain_graph=True)
    return total_nll
