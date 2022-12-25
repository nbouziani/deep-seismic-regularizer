import matplotlib.pyplot as plt

from firedrake import *

from velocity_models.velocity import load_velocity_model


mesh = Mesh('meshes/marmousi_hmin_40_32k.msh')
V = FunctionSpace(mesh, 'DG', 1)
vp = load_velocity_model('marmousi', V)

fig, axes = plt.subplots(figsize=(17, 5))
collection = tripcolor(vp, axes=axes)
plt.colorbar(collection)

plt.show()
