from pythtb import TBModel, WFArray, Mesh, Lattice
import numpy as np 
import matplotlib.pyplot as plt 
from HamiltonianModel.hamiltonian import set_model as sm 

my_model = sm(1.0, 0.24, 0.05, 1.0, 0.0)

mesh = Mesh(["k", "k", "k"])
mesh.build_grid(shape=(10, 10, 10), gamma_centered=True, k_endpoints=[True, True, True])
print(mesh)

wfa = WFArray(my_model.lattice, mesh, spinful=True)
wfa.solve_model(my_model)

bflux = wfa.berry_flux(state_idx=[1], plane=(0, 1))

mesh_cart = mesh.points @ my_model.recip_lat_vecs
KX, KY = mesh_cart[..., 0], mesh_cart[..., 1]

kz_index = 0

X_2d = KX[:-1, :-1, kz_index]
Y_2d = KY[:-1, :-1, kz_index]
Z_2d = bflux[:, :, kz_index]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X_2d, Y_2d, Z_2d, cmap="plasma")

ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
ax.set_zlabel(r"$\Omega(\mathbf{k})$")

fig.colorbar(surf, label=r"$\Omega(\mathbf{k})$", pad=0.1)

plt.savefig("result/berry_curv.pdf")
plt.show()
