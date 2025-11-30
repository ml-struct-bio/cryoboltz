import argparse
import numpy as np
import mrcfile

parser = argparse.ArgumentParser(description="Crop map for input to CryoBoltz")
parser.add_argument('map', help='Path to MRC file')
parser.add_argument('--thresh', type=float, 
                    help="Crop map to a tight box at this map threshold")
parser.add_argument('--pad', type=int, 
                    help="If cropping based on a threshold, how many voxels to add to each side after cropping")
parser.add_argument('--dim', type=int, help="Crop map to a cube of this dimension")
parser.add_argument('-o', required=True, help='Path to output MRC file')
args = parser.parse_args()

old_mrc = mrcfile.open(args.map, mode='r')
old_data = old_mrc.data
nz, ny, nx = old_data.shape

if args.thresh is not None:
    z, y, x = np.nonzero(old_data >= args.thresh)
    zs, ys, xs = z.min(), y.min(), x.min()
    ze, ye, xe = z.max() + 1, y.max() + 1, x.max() + 1
    if args.pad is not None:
        p = args.pad
        zs, ys, xs = max(zs - p, 0), max(ys - p, 0), max(xs - p, 0)
        
        ze, ye, xe = min(ze + p, nz), min(ye + p, ny), min(xe + p, nx)
elif args.dim is not None:
    d = args.dim
    zs, ys, xs = (nz - d)//2, (ny - d)//2, (nx - d)//2
    ze, ye, xe = zs + d, ys + d, xs + d

new_data = old_data[zs:ze, ys:ye, xs:xe]
new_mrc = mrcfile.new(args.o, overwrite=True)
new_mrc.set_data(new_data)
new_mrc.voxel_size = old_mrc.voxel_size
xorg = old_mrc.header.origin['x'] + old_mrc.voxel_size['x'] * xs
yorg = old_mrc.header.origin['y'] + old_mrc.voxel_size['y'] * ys
zorg = old_mrc.header.origin['z'] + old_mrc.voxel_size['z'] * zs
new_mrc.header.origin = (xorg, yorg, zorg)

old_mrc.close()
new_mrc.close()
