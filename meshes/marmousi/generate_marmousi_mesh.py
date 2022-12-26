# mpiexec -n 2 python generate_marmousi_mesh.py

import os
import argparse
import meshio

from mpi4py import MPI

from SeismicMesh import get_sizing_function_from_segy, generate_mesh, Rectangle


comm = MPI.COMM_WORLD

"""
Build a mesh of the Marmousi benchmark velocity model in serial or parallel
Takes roughly 1 minute with 2 processors and less than 1 GB of RAM.
"""


parser = argparse.ArgumentParser(description="Generate Marmousi mesh")
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
parser.add_argument(
    "--hmin",
    default=30.,
    type=float,
    help="hmin: minimum mesh size",
)

args = parser.parse_args()

# Name of SEG-Y file containg velocity model.
# path = '/home/nacime/firedrake/src/firedrake/firedrake/external_operators/neural_networks/notebooks/'
# fname = os.path.join(path, "MODEL_S-WAVE_VELOCITY_1.25m.segy")
resources_dir = os.path.join(os.environ['HOME'], 'deep-seismic-regularizer')
velocity_dir = os.path.join(resources_dir, 'velocity_models/marmousi')
mesh_dir = os.path.join(resources_dir, 'meshes/marmousi')
fname = os.path.join(velocity_dir, "marmvel.segy")

# Depth in km
Lz = 3000.
# Width in km
Lx = 9200.  # 17000.
# Bounding box describing domain extents (corner coordinates)
bbox = (-Lz, 0.0, 0.0, Lx)
rectangle = Rectangle(bbox)

# Desired minimum mesh size in domain
hmin = args.hmin
# hmin = 1429.0/(M*frequency)

# Number of cells per wavelength for a given 𝑓𝑚𝑎𝑥 (default==0 cells)
wl = 15
# 𝑓𝑚𝑎𝑥 in hertz for which to estimate `wl` (default==2 Hertz)
freq = 5
# Theoretical maximum stable timestep in seconds given Courant number Cr (default==0.0 s)
dt = 0.001 
# Maximum allowable variation in mesh size in decimal percent (default==0.0)
grade = 0.15

# Construct mesh sizing object from velocity model
ef = get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=wl,
    freq=freq,
    dt=dt,
    grade=grade
    # The width of the domain pad in -z, +x, -x, +y, -y directions (default==0.0 m).
    #domain_pad=1e3,
    # The method (`edge`, `linear_ramp`, `constant`) to pad velocity in the domain pad region (default==None)
    # pad_style=pad,
)

points, cells = generate_mesh(domain=rectangle, edge_length=ef, mesh_improvement=False )

# import ipdb ; ipdb.set_trace()
if comm.rank == 0:
# Write the mesh in a vtk format for visualization in ParaView
# NOTE: SeismicMesh outputs assumes the domain is (z,x) so for visualization
# in ParaView, we swap the axes so it appears as in the (x,z) plane.
	meshio.write_points_cells(
	    os.path.join(mesh_dir, "marmvel_hmin_%s.msh" % int(hmin)),
	    # points/1000,
	    points[:, [1, 0]]/ 1000,
	    # points/1000,
	    {"triangle": cells},
	    file_format="gmsh22"
	)
