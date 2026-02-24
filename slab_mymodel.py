from pythtb import TBModel, Lattice
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from HamiltonianModel.hamiltonian import set_model
delta = 0.0
t = 1.0
soc_val = 0.1
rashba = 0.1
W = 0 * soc_val

# Matriks Pauli

sigma_z = np.array([0., 0., 0., 1.])
sigma_x = np.array([0., 1., 0., 0])
sigma_y = np.array([0., 0., 1., 0])

r3h = np.sqrt(3.0) / 2.0

sigma_a = 0.5 * sigma_x - r3h * sigma_y
sigma_b = 0.5 * sigma_x + r3h * sigma_y
sigma_c = -1.0 * sigma_x

my_model = set_model(t, soc_val, rashba, delta, W)
print(my_model)
my_model.info()

fin_model = my_model.make_finite(periodic_dirs=[1], num_cells=[20])

k_nodes = [
    [0.0, 0.0],  # Gamma
    [0.5, 0.0],  # Mx
    [0.5, 0.5],  # Lx
    [0.0, 0.5],  # A
    [0.0, 0.0]   # Gamma
]
k_labels = (r"$\Gamma $",r"$M_x$", r"$L_x$", r"$A$", r"$\Gamma $")

#fin_model.visualize_3d(draw_hoppings=True)

fig, ax = fin_model.plot_bands(
    nk=500, k_nodes=k_nodes, k_node_labels=k_labels, proj_orb_idx=[1]
)
plt.savefig('result/ribbon_band.pdf')
plt.show()
