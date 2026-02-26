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

wan_cent_inter = wf_array.berry_phase(
        axis_idx=1, state_idx=[1,2], contin=True, berry_evals=True
        )

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

#intralayer case
nky = wan_cent.shape[0]
ky = np.linspace(0, 1, nky)

flux = wf_array.berry_flux(plane=(0, 1), state_idx=[0,1], non_abelian=False)
chern_kz = np.sum(flux, axis=(0,1)) / (2 * np.pi)
print(chern_kz)

for shift in range(-2, 3):
    ax[0].plot(ky, wan_cent[:, 0] + float(shift), "k")
    ax[0].plot(ky, wan_cent[:, 1] + float(shift), "k")

ax[0].set_ylabel("Wannier center along x")
ax[0].set_xlabel(r"$k_y$")
ax[0].set_xticks([0.0, 0.5, 1.0])
ax[0].set_xlim(0.0, 1.0)
ax[0].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax[0].axvline(x=0.5, linewidth=0.5, color="k")
ax[0].set_title("1D Wannier centers: Intralayer Case")

#interlayer case
nky2 = wan_cent_inter.shape[0]
ky2 = np.linspace(0, 1, nky2)

for shift in range(-2, 3):
    ax[1].plot(ky2, wan_cent_inter[:, 2] + float(shift), "k")
    ax[1].plot(ky2, wan_cent_inter[:, 1] + float(shift), "k")

ax[1].set_ylabel("Wannier center along z")
ax[1].set_xlabel(r"$k_y$")
ax[1].set_xticks([0.0, 0.5, 1.0])
ax[1].set_xlim(0.0, 1.0)

ax[1].set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax[1].axvline(x=0.5, linewidth=0.5, color="k")

ax[1].set_title("1D Wannier centers: Interlayer Case")

plt.savefig("result/hwf.pdf")
plt.show()
