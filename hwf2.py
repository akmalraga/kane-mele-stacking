from pythtb import WFArray, Mesh, Lattice, TBModel
import numpy as np 
import matplotlib.pyplot as plt 
from HamiltonianModel.hamiltonian import set_model as sm 

t = 1.0
soc = 0.3
rashba = 0.25
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

#model.plot_bands(       k_nodes=k_nodes, nk=201, k_node_labels=label_k, proj_orb_idx=[0] )

mesh = Mesh(["k", "k", "k"])
mesh.build_grid(shape=(41, 41, 41), gamma_centered=True)
print(mesh)

wf_array = WFArray(model.lattice, mesh, spinful=True)
wf_array.solve_model(model=model)

wan_cent = wf_array.berry_phase(
        axis_idx=1, state_idx=[0,1], contin=True, berry_evals=True
        )
wan_cent /= 2 * np.pi

nky = wan_cent.shape[0]
ky = np.linspace(0, 1, nky)

#print(wan_cent)

for shift in range(-2, 3):
    plt.plot(ky, wan_cent[:, 0] + float(shift), "k")
    plt.plot(ky, wan_cent[:, 1] + float(shift), "k")
    plt.plot(ky, wan_cent[:, 2] + float(shift), "k")

plt.savefig("result/hwf.pdf")
plt.show()
