"""Residue-specific coupling for multi-conformation joint sampling.

Implements the coupling term from the Bayesian joint posterior:
    nabla_{x_i} log p(x_1,...,x_N | y_1,...,y_N, s) includes
    sum_{j!=i} Lambda_{ij} (x_j - x_i)

where Lambda_{ij} is a residue-specific coupling strength matrix:
strong for rigid regions, weak for flexible regions.
"""

import torch
import numpy as np


def estimate_flexibility_from_bfactors(bfactors, normalize=True):
    """Estimate per-residue flexibility from B-factors.

    Higher B-factors indicate more flexible regions that should
    have weaker coupling between conformations.

    Parameters
    ----------
    bfactors : torch.Tensor
        Per-residue B-factors, shape (n_residues,).
    normalize : bool
        Whether to normalize to [0, 1] range.

    Returns
    -------
    torch.Tensor
        Per-residue flexibility scores in [0, 1], shape (n_residues,).
    """
    flex = bfactors.clone().float()
    if normalize and flex.max() > flex.min():
        flex = (flex - flex.min()) / (flex.max() - flex.min())
    return flex


def estimate_flexibility_from_coords(ref_coords_list, ref_masks, ca_indices=None):
    """Estimate per-residue flexibility from coordinate variance across conformations.

    Given multiple reference structures (e.g., aligned initial models),
    compute the per-residue RMSF as a proxy for flexibility.

    Parameters
    ----------
    ref_coords_list : list[torch.Tensor]
        List of N reference coordinate tensors, each shape (n_atoms, 3).
    ref_masks : list[torch.Tensor]
        List of N masks, each shape (n_atoms,).
    ca_indices : torch.Tensor, optional
        Indices of CA atoms for residue-level aggregation.

    Returns
    -------
    torch.Tensor
        Per-atom flexibility scores in [0, 1], shape (n_atoms,).
    """
    # Stack all coords: (N, n_atoms, 3)
    coords = torch.stack(ref_coords_list, dim=0)
    masks = torch.stack(ref_masks, dim=0)

    # Compute per-atom mean position across conformations
    valid = masks.unsqueeze(-1).float()  # (N, n_atoms, 1)
    n_valid = valid.sum(0).clamp(min=1)  # (n_atoms, 1)
    mean_pos = (coords * valid).sum(0) / n_valid  # (n_atoms, 3)

    # Compute RMSF (root mean square fluctuation)
    deviations = ((coords - mean_pos.unsqueeze(0)) ** 2).sum(-1)  # (N, n_atoms)
    rmsf = torch.sqrt((deviations * masks.float()).sum(0) / n_valid.squeeze(-1).clamp(min=1))

    # Normalize to [0, 1]
    if rmsf.max() > rmsf.min():
        rmsf = (rmsf - rmsf.min()) / (rmsf.max() - rmsf.min())

    return rmsf


def build_coupling_strength(flexibility, base_strength=1.0, min_strength=0.01):
    """Convert flexibility scores to coupling strengths.

    Rigid regions (low flexibility) get strong coupling.
    Flexible regions (high flexibility) get weak coupling.

    Parameters
    ----------
    flexibility : torch.Tensor
        Per-residue flexibility in [0, 1], shape (n_atoms,).
    base_strength : float
        Maximum coupling strength for rigid regions.
    min_strength : float
        Minimum coupling strength for flexible regions.

    Returns
    -------
    torch.Tensor
        Per-atom coupling strength, shape (n_atoms,).
    """
    # Lambda = base_strength * (1 - flexibility) + min_strength
    strength = base_strength * (1.0 - flexibility) + min_strength
    return strength


def compute_coupling_gradient(all_coords, coupling_strength, conf_idx, atom_mask,
                              active_indices=None):
    """Compute the coupling gradient for conformation i.

    Coupling term: sum_{j!=i} Lambda * (x_j - x_i)
    This pushes conformation i toward the mean of other conformations,
    weighted by residue-specific coupling strength.

    Parameters
    ----------
    all_coords : list[torch.Tensor]
        List of N coordinate tensors, each shape (multiplicity, n_atoms, 3).
    coupling_strength : torch.Tensor
        Per-atom coupling strength, shape (n_atoms,).
    conf_idx : int
        Index of the current conformation.
    atom_mask : torch.Tensor
        Atom mask, shape (multiplicity, n_atoms).
    active_indices : list[int], optional
        Indices of conformations to couple with. If None, couple with all.

    Returns
    -------
    torch.Tensor
        Coupling gradient for conformation i, shape (multiplicity, n_atoms, 3).
    """
    n_conf = len(all_coords)
    x_i = all_coords[conf_idx]  # (multiplicity, n_atoms, 3)
    device = x_i.device

    if active_indices is None:
        active_indices = [j for j in range(n_conf) if j != conf_idx]
    else:
        active_indices = [j for j in active_indices if j != conf_idx]

    if len(active_indices) == 0:
        return torch.zeros_like(x_i)

    # Lambda per atom: (n_atoms,) -> (1, n_atoms, 1) for broadcasting
    lam = coupling_strength.to(device).unsqueeze(0).unsqueeze(-1)
    mask = atom_mask.unsqueeze(-1).float()  # (multiplicity, n_atoms, 1)

    # Compute mean displacement from all coupled conformations
    grad = torch.zeros_like(x_i)
    for j in active_indices:
        x_j = all_coords[j].to(device)
        grad += lam * (x_j - x_i) * mask

    return grad


def compute_pairwise_rmsd(all_coords, atom_mask):
    """Compute pairwise RMSD between all conformations.

    Parameters
    ----------
    all_coords : list[torch.Tensor]
        List of N coordinate tensors, each shape (multiplicity, n_atoms, 3).
    atom_mask : torch.Tensor
        Atom mask, shape (multiplicity, n_atoms).

    Returns
    -------
    torch.Tensor
        Pairwise RMSD matrix, shape (N, N). Uses first sample in multiplicity.
    """
    n_conf = len(all_coords)
    rmsd_matrix = torch.zeros(n_conf, n_conf)
    mask = atom_mask[0].bool()

    for i in range(n_conf):
        for j in range(i + 1, n_conf):
            diff = all_coords[i][0, mask] - all_coords[j][0, mask]
            rmsd = torch.sqrt((diff ** 2).sum(-1).mean())
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    return rmsd_matrix
