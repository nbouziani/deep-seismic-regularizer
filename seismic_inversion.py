import os
import argparse
import matplotlib.pyplot as plt

from firedrake import *

from velocity_models.velocity import load_velocity_model


parser = argparse.ArgumentParser(description="Run seismic inversion")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="device id",
)
parser.add_argument(
        "--resources_dir",
        default=os.path.join(os.environ['HOME'], 'deep-seismic-regularizer'),
        type=str,
        required=False,
        help="Resources dir",
)
parser.add_argument(
    "--mesh_dir",
    default='meshes',
    type=str,
    required=False,
    help="Mesh dir",
)
parser.add_argument(
    "--velocity_dir",
    default='velocity_models',
    type=str,
    required=False,
    help="Velocity model dir",
)

args = parser.parse_args()
mesh_dir = os.path.join(args.resources_dir, args.mesh_dir)

mesh = Mesh(os.path.join(mesh_dir, 'marmousi_hmin_40_32k.msh'))
V = FunctionSpace(mesh, 'DG', 2)
vp = load_velocity_model('marmousi', V)

fig, axes = plt.subplots(figsize=(17, 5))
collection = tripcolor(vp, axes=axes)
plt.colorbar(collection)

plt.show()
