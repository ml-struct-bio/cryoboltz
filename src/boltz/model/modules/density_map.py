import torch.nn as nn
import numpy as np
import torch
import mrcfile
from boltz.data import const

ATOM_TO_Z = {'H': 1.0, 'C': 6.0, 'N': 7.0, 'O': 8.0, 'S':16.0}

class MapGenerator(nn.Module):
    def __init__(self, mrc_path, seq, nominal_res=2.0, ref_center=None, device="cuda"):
        super(MapGenerator, self).__init__()
        self.nominal_res = nominal_res

        self.n_residues = len(seq)
        if type(seq)==str: tricodes = [const.prot_letter_to_token[s] for s in seq]
        else: tricodes = [const.tokens[int(s)] for s in seq]
        self.indices_X_full_to_coords = []
        for i, tricode in enumerate(tricodes):
            n_atoms = len(const.ref_atoms[tricode])
            for j in range(n_atoms):
                self.indices_X_full_to_coords.append(i * 14 + j)

        self.n_atoms = len(self.indices_X_full_to_coords)
        amplitudes = []
        for tricode in tricodes:
            atoms = const.ref_atoms[tricode]
            for atom in atoms:
                amplitudes.append(ATOM_TO_Z[atom[0]])
        self.amplitudes = torch.tensor(amplitudes).float().to(device)

        with mrcfile.open(mrc_path) as mrc:
            voxel_size = mrc.voxel_size['x'].item()
            self.density = mrc.data
            self.mrc_header = mrc.header
            nx = mrc.header['nx'].item()
            ny = mrc.header['ny'].item()
            nz = mrc.header['nz'].item()
        
        self.origin = torch.tensor([self.mrc_header['origin'][i].item() for i in ['x', 'y', 'z']]).to(device)
        ax_x = torch.arange(nx).float()
        ax_y = torch.arange(ny).float()
        ax_z = torch.arange(nz).float()
        pix_coords_Z, pix_coords_Y, pix_coords_X = torch.meshgrid(ax_z, ax_y, ax_x, indexing='ij')
        self.pix_coords_3d = torch.stack([pix_coords_X, pix_coords_Y, pix_coords_Z], dim=-1)
        self.pix_coords_3d = self.pix_coords_3d.reshape(-1, 3).to(device) * voxel_size
        self.pix_coords_3d += self.origin
        if ref_center is not None: self.pix_coords_3d -= ref_center

        self.n_pix = nx * ny * nz
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.voxel_size = voxel_size

    def get_full_density_per_batch(self, coords, n_sampled_voxels, resolution):
        potential_full = torch.zeros(self.nx * self.ny * self.nz).to(coords.device)
        for i in range(((self.n_pix - 1) // n_sampled_voxels) + 1):
            i_min = i * n_sampled_voxels
            i_max = min((i + 1) * n_sampled_voxels, self.n_pix)
            sampled_indices = torch.arange(i_min, i_max).to(coords.device)
            pix_coords_3d_sampled = self.pix_coords_3d[sampled_indices]
            assert (torch.linalg.norm(coords, dim=-1) < 1-6).sum() <= 1, f"{(torch.linalg.norm(coords, dim=-1) < 1-6).sum()} atoms at the origin"
            potential_sampled = self.calculate_potential_one_batch(coords, pix_coords_3d_sampled, resolution).reshape(-1)
            potential_full[i_min: i_max] = potential_sampled 
        potential_full = potential_full.reshape(1, self.nz, self.ny, self.nx)
        return potential_full

    # coords (N, 3), pix_coords_3d_sampled (V, 3)
    def calculate_potential_one_batch(self, coords, pix_coords_3d_sampled, resolution):
        sigma = resolution / (np.sqrt(2.) * np.pi)
        dist_sq = ((coords[:, None, :] - pix_coords_3d_sampled[None, :, :])**2).sum(-1) # (N, V)
        potential = (self.amplitudes[:, None] * torch.exp(-(dist_sq / (2*sigma**2)))).sum(0)
        return potential
