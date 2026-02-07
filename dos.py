from pythtb import TBModel, Lattice, Mesh
import numpy as np 
import matplotlib.pyplot as plt 
from HamiltonianModel.hamiltonian import set_model as sm 

t = 1.0
soc = 1/4
rashba = 0.05
delta = 1.0 

model = sm(t, soc, rashba, delta, 0)

k_nodes = [
    [0.0, 0.0, 0.0],    # Gamma
    [2./3., 1./3., 0.0],# K
    [0.5, 0.5, 0.0],    # M
    [1./3., 2./3., 0.0],# K'
    [0.0, 0.0, 0.0],    # Gamma
]
label_k = (r"$\Gamma$", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma$")
(k_vec, k_dist, k_node) = model.k_path(k_nodes, 101)

evals = model.solve_ham(k_vec)

mesh = Mesh(dim_k=3, axis_types=["k", "k", "k"])
mesh.build_grid(shape=(20, 20, 20))
kpts = mesh.flat

energies = model.solve_ham(kpts)
energies = energies.flatten()

fin_model_true = model.make_finite([0, 1, 2], [10, 10, 10], glue_edges=[True, True, True])
evals_true = fin_model_true.solve_ham()

fin_model_False = model.make_finite([0, 1, 2], [10, 10, 10], glue_edges=[False, False, False])
evals_false = fin_model_False.solve_ham()

# flatten eigenvalue arrays
evals_false = evals_false.flatten()
evals_true = evals_true.flatten()

# now plot density of states
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].hist(evals_false, 50, range=(-4.0, 4.0))
ax[0].set_ylim(0.0, 80.0)
ax[0].set_title("Finite model without PBC")
ax[0].set_xlabel("Band energy")
ax[0].set_ylabel("Number of states")

ax[1].hist(evals_true, 50, range=(-4.0, 4.0))
ax[1].set_ylim(0.0, 80.0)
ax[1].set_title("Finite model with PBC")
ax[1].set_xlabel("Band energy")
ax[1].set_ylabel("Number of states")

plt.hist(energies, 100)
plt.savefig("result/dos.pdf")
plt.show()
