import numpy as np
import torch
import cc3d

# Takes in (V, V, V) density map and returns (K, 3) point cloud
def kmeans_cloud(density, voxel_size, origin, ref_center, K, weighted=True, thresh=0.0, dust=0):
    device = density.device
    density_new = density.clone()
    density_new[density_new <= thresh] = 0.0
    if dust > 0:
        mask = np.array((density_new.cpu() > 0))
        cleaned_mask = cc3d.dust(mask, threshold=dust, connectivity=26, in_place=False)
        density_new[~torch.tensor(cleaned_mask).to(device)] = 0.0
    cloud = torch.argwhere(density_new > 0)
    intensities = (density_new[density_new > 0] if weighted else torch.ones(len(cloud))).to(device)
    cloud = voxel_size * cloud.float()
    cloud = cloud[:, [2, 1, 0]]
    K = min(K, len(cloud))
    cloud, _ = weighted_kmeans(cloud, intensities, K)
    cloud = cloud + origin - ref_center
    return cloud


def weighted_kmeans(points, weights, k, max_iters=100, tol=1e-4):
    N = points.shape[0]

    # Initialize centroids
    indices = torch.randperm(N)[:k]
    centroids = points[indices].clone()

    for _ in range(max_iters):
        # Assign points to nearest centroid
        dists = torch.cdist(points, centroids)
        assignments = torch.argmin(dists, dim=1)

        # Update centroids using weighted average
        new_centroids = torch.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.any():
                w = weights[mask]
                p = points[mask]
                new_centroids[i] = (w[:, None] * p).sum(dim=0) / w.sum()
            else:
                # Reinitialize empty cluster to a random point
                new_centroids[i] = points[torch.randint(0, N, (1,))]

        # Check convergence
        shift = torch.norm(new_centroids - centroids, dim=1).sum()
        if shift < tol:
            break
        centroids = new_centroids

    return centroids, assignments


def save_cloud(points, outpdb, weights=None):
    element = "C"
    with open(outpdb, 'w') as f:
        for i, xyz in enumerate(points):
            x, y, z = xyz
            w = f"{weights[i]:6.2f}" if weights is not None else 1.0
            f.write(
                f"HETATM{i:5d}  {element:<2s}  UNL     1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  {w}  {w}          {element:>2s}\n"
            )
        f.write("END\n")
