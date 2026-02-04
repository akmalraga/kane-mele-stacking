from pythtb import TBModel, Lattice
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from HamiltonianModel.hamiltonian import set_model as sm


delta = np.array([0.0, 0.5, 1.0, 2.5, 3.0])
t = 1.0
soc_val = 0.25
rashba = np.array([0.0, 0.05, 0.1, 0.2, 0.3])

fig, ax = plt.subplots(5, 5, figsize=(12, 5))

k_nodes = [
    [0.0, 0.0, 0.0],    # Gamma
    [2./3., 1./3., 0.0],# K
    [0.5, 0.5, 0.0],    # M
    [1./3., 2./3., 0.0],# K'
    [0.0, 0.0, 0.0],    # Gamma
]
label_k = (r"$\Gamma$", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma$")

for i, d in enumerate(delta):
    for j, r in enumerate(rashba):
        my_model = sm(t, soc_val, r, d, 0)

        my_model.plot_bands(
            k_nodes=k_nodes,
            nk=201,
            k_node_labels=label_k,
            fig=fig,
            ax=ax[i, j],
            proj_orb_idx=[0],
            cmap='inferno',
            cbar=True
        )

for i in range(5):
    for j in range(5):
        ax[i, j].grid(True, alpha=0.3)

        # Hanya baris terakhir yang punya label k
        if i != 4:
            ax[i, j].set_xticklabels([])
            ax[i, j].set_xlabel("")
        
        # Hanya kolom pertama yang punya label energi
        if j != 0:
            ax[i, j].set_yticklabels([])
            ax[i, j].set_ylabel("")

        # Judul kecil tiap panel
        ax[i, j].set_title(
            rf"$\delta={delta[i]}$, $\lambda_R={rashba[j]}$",
            fontsize=8
        )

# Label global
fig.text(0.5, 0.04, "k-path", ha='center', fontsize=12)
fig.text(0.04, 0.5, "Energy", va='center', rotation='vertical', fontsize=12)

# Spasi antar subplot (lebih rapi dari tight_layout)
plt.subplots_adjust(
    left=0.08,
    right=0.98,
    bottom=0.08,
    top=0.95,
    wspace=0.25,
    hspace=0.35
)


plt.tight_layout()
plt.savefig('peta_bulk.pdf', dpi=600)
plt.show()

