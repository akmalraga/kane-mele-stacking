from pythtb import tb_model
import numpy as np
import matplotlib.pyplot as plt
from HamiltonianModel.hamiltonian import set_model as sm

def get_my_model(phase):
    ## --- Parameters ---
    t = 1.0
    soc_val = 0.25
    rashba = 0.05
    
    if phase == "trivial":
        delta = 2.5  
    else:
        delta = 1.0

    ## --- Pauli Matrices ---
    sigma_0 = np.eye(2, dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    r3h = np.sqrt(3.0) / 2.0
    sigma_a = 0.5 * sigma_x - r3h * sigma_y
    sigma_b = 0.5 * sigma_x + r3h * sigma_y
    sigma_c = -1.0 * sigma_x

    ## --- Lattice Definition ---
    lat_vecs = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
    orb_vecs = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
    my_model = tb_model(3, 3, lat_vecs, orb_vecs, nspin=2)

    ## --- Onsite Energies ---
    my_model.set_onsite([delta, -delta])

    ## --- Hopping Terms ---
    # Nearest Neighbor
    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):
        my_model.set_hop(t * sigma_0, 0, 1, lvec)

    # Next-Nearest Neighbor (Kane-Mele SOC)
    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):
        my_model.set_hop(soc_val * 1j * sigma_z, 0, 0, lvec)
    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):
        my_model.set_hop(soc_val * 1j * sigma_z, 1, 1, lvec)

    # Interlayer Hopping
    my_model.set_hop(0.1 * soc_val * 1j * sigma_z, 1, 1, [0, 0, 1])
    my_model.set_hop(-0.1 * soc_val * 1j * sigma_z, 0, 0, [0, 0, 1])

    # Rashba Coupling
    my_model.set_hop(1j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")
    my_model.set_hop(1j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")
    my_model.set_hop(1j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")

    return my_model

## --- Band Structure Calculation & Plotting ---
model_triv = get_my_model("trivial")
model_topo = get_my_model("topological")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# K-Space Path
k_nodes = [
    [0.0, 0.0, 0.0],    # Gamma
    [2./3., 1./3., 0.0],# K
    [0.5, 0.5, 0.0],    # M
    [1./3., 2./3., 0.0],# K'
    [0.0, 0.0, 0.0],    # Gamma
]
label_k = (r"$\Gamma$", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma$")

# Plotting
model_triv.plot_bands(k_nodes=k_nodes, nk=201, k_node_labels=label_k, fig=fig, ax=ax1, proj_orb_idx=[0],cmap='inferno')
model_topo.plot_bands(k_nodes=k_nodes, nk=201, k_node_labels=label_k, fig=fig, ax=ax2, proj_orb_idx=[0], cmap='inferno')

# Formatting
ax1.set_title(r"Trivial Phase ($\Delta = 3.0$)")
ax2.set_title(r"Topological Phase ($\Delta = 0.7$)")
ax1.set_ylim(-6, 6)
ax2.set_ylim(-6, 6)

#model_topo.visualize_3d()

plt.tight_layout()
plt.savefig('result/topology_vs_trivial.pdf')
plt.show()
