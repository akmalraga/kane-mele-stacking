from pythtb import TBModel, Lattice, Mesh
import numpy as np 
import matplotlib.pyplot as plt 
from HamiltonianModel.hamiltonian import set_model as sm 

delta = np.array([0.0, 0.5, 1.0, 2.5, 3.0])
t = 1.0
soc_val = 0.25
rashba = np.array([0.0, 0.05, 0.1, 0.2, 0.3])


k_nodes = [
    [0.0, 0.0, 0.0],    # Gamma
    [2./3., 1./3., 0.0],# K
    [0.5, 0.5, 0.0],    # M
    [1./3., 2./3., 0.0],# K'
    [0.0, 0.0, 0.0],    # Gamma
]
label_k = (r"$\Gamma$", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma$")


fig, ax = plt.subplots(len(delta), len(rashba), figsize=(12, 12))
ax = np.atleast_2d(ax)


for i, d in enumerate(delta):
    for j, r in enumerate(rashba):
        my_model = sm(t, soc_val, r, d, 0)
        (k_vec, k_dist, k_node) = my_model.k_path(k_nodes, 101)

        evals = my_model.solve_ham(k_vec)

        mesh = Mesh(dim_k=3, axis_types=["k", "k", "k"])
        mesh.build_grid(shape=(20, 20, 20))
        kpts = mesh.flat

        energies = my_model.solve_ham(kpts)
        energies = energies.flatten()

        fin_model_False = my_model.make_finite([0, 1, 2], [10, 10, 2], glue_edges=[False, False, False])
        evals_false = fin_model_False.solve_ham()

        evals_false = evals_false.flatten()

        ax[i,j].hist(evals_false, bins=100, density=True, alpha=0.6, range=(-4.0, 4.0), color='red')

        if i == 0:
            ax[i, j].set_title(f"Rashba = {r}")
        if j == 0:
            ax[i, j].set_ylabel(f"$\Delta$ = {d}\nDOS")
        if i == len(delta) - 1:
            ax[i, j].set_xlabel("Energy")

        if i == 0 and j == 0:
            ax[i, j].legend()

plt.tight_layout()
plt.savefig("result/Peta_Kerapatan_Energi.pdf")
plt.show()
