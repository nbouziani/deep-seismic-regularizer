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
    maxz = 0.5
    minx = 0.0
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

        # make sure no out-of-bounds
        qp_z2 = [minz if z < minz else maxz if z > maxz else z for z in qp_z]
        qp_x2 = [minx if x < minx else maxx if x > maxx else x for x in qp_x]

        interpolant = RegularGridInterpolator((z, x), Z)
        v = interpolant((qp_z2, qp_x2))

    c = Function(V)
    c.dat.data[:] = v
    # Check units ?
    return c
