import os
import h5py
import gzip
import numpy as np

from scipy.interpolate import RegularGridInterpolator

from firedrake.function import Function
from firedrake.functionspace import VectorFunctionSpace
from firedrake.interpolation import interpolate


def interpolate_2d_model(fname, V, Lz, Lx):
    mesh = V.ufl_domain()
    minz = - Lz
    maxz = 0.
    minx = 0
    maxx = Lx

    W = VectorFunctionSpace(mesh, V.ufl_element())
    coords = interpolate(mesh.coordinates, W)
    # (z,x)
    qp_x, qp_z = coords.dat.data[:, 0], coords.dat.data[:, 1]

    with h5py.File(fname, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])

        n, m = Z.shape
        z = np.linspace(minz, maxz, n)
        x = np.linspace(minx, maxx, m)

        # Sanitize out-of-bounds values
        qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
        qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]

        interpolant = RegularGridInterpolator((z, x), Z)
        v = interpolant((qp_z2, qp_x2))

    c = Function(V)
    # Converting from m/s to km/s
    c.dat.data[:] = v / 1000.
    return c


def load_velocity_model(name, V, model_dir=''):
    if name == 'circle':
        pass
    elif name == 'waveguide':
        pass
    elif name == 'marmousi':
        # Load velocity model file
        name_file = os.path.join(model_dir, 'marmvel.hdf5')
        # Depth (km)
        Lz = 3.
        # Width (km)
        Lx = 9.2  # 17.
        c = interpolate_2d_model(name_file, V, Lz, Lx)
    elif name == 'marmousi2':
        # Load and decompress velocity model file
        name_file = os.path.join(model_dir, 'marmousi2', name + '.hdf5')
        with open(name_file, 'wb') as f:
            with gzip.open(name_file + '.gz', 'rb') as g:
                f.write(g.read())
        # Depth (km)
        Lz = 3.
        # Width (km)
        Lx = 17.
        c = interpolate_2d_model(name_file, V, Lz, Lx)
    else:
        raise ValueError

    return c
