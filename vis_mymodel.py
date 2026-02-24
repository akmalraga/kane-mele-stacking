from pythtb import TBModel, Lattice
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import plotly.express as px
from HamiltonianModel.hamiltonian import set_model

delta = 0.0
t = 1.0
soc_list = np.array([0.06, 0.24])
soc_val = 0.125
rashba = 0.05
W = 0 * soc_list

# Matriks Pauli
sigma_z = np.array([0., 0., 0., 1.])
sigma_x = np.array([0., 1., 0., 0])
sigma_y = np.array([0., 0., 1., 0])
r3h = np.sqrt(3.0) / 2.0
sigma_a = 0.5 * sigma_x - r3h * sigma_y
sigma_b = 0.5 * sigma_x + r3h * sigma_y
sigma_c = -1.0 * sigma_x

data_csv =[]
my_model = set_model(t, soc_val, rashba, delta, 0)

print(my_model)
my_model.info()

sc_model = my_model.make_supercell([[2, 1, 0], [-1, 2, 0], [0, 0, 2]], to_home=True)
slab_model = sc_model.cut_piece(3,1, glue_edges=False)
pos = slab_model.get_orb_vecs()
z_vals = pos[:,2]
z_unique = np.unique(np.round(z_vals, 5))
colors_layer = ["red", "green", "blue", "orange", "purple"]
color_site = []

for z in z_vals:
    idx = np.where(z_unique == np.round(z,5))[0][0]
    color_site.append(colors_layer[idx % len(colors_layer)])

fig = slab_model.visualize_3d(show_model_info=False, site_colors=color_site)

k_nodes = [[0, 0], [2 / 3, 1 / 3], [0.5, 0.5], [1 / 3, 2 / 3], [0, 0], [0.5, 0.5]]
k_label = (r"$\Gamma $", r"$K$", r"$M$", r"$K^\prime$", r"$\Gamma $", r"$M$")
fig, ax = slab_model.plot_bands(
        nk=500, k_nodes=k_nodes, k_node_labels=k_label, proj_orb_idx=[0], lw=1
       )
fig.show()
