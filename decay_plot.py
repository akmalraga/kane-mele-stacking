from pythtb import Mesh, Wannier, WFArray 
from HamiltonianModel.hamiltonian import set_model 
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

prim_model = set_model(1.0, 0.3, 0.1, 1.0, 0.0)

n_super_cell = 1 
model = prim_model.make_finite(periodic_dirs=[2], num_cells=[2])
model.info(show=True, short=False)

nks = 20, 20 
mesh = Mesh(dim_k=2, axis_types=["k", "k"])
mesh.build_grid(shape=nks)
print(mesh)
wfa = WFArray(model.lattice, mesh, spinful=True)
wfa.solve_model(model)

n_orb = model.norb
n_occ = int(n_orb / 2)

low_E_sites = np.arange(
        0, n_orb, 2
        )
high_E_sites = np.arange(
        1, n_orb, 2
        )

omite_site = 6
sites = list(np.setdiff1d(low_E_sites, [omite_site]))

tf_list = [
    [(int(orb), 0, 1.0), (int(orb), 1, 1.0)]
    for orb in low_E_sites
]

n_tfs = len(tf_list)

print(f"Trial Wavefunction: {tf_list}")
print(f"# of Wannier function: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {n_tfs / n_occ}")

WF = Wannier(wfa)
WF.project(tf_list, band_idxs=list(range(n_occ)))

WF.project(use_tilde=True)

WF.maxloc(alpha=1 / 2, max_iter=1000, tol=1e-10, grad_min=1e-10, verbose=True)

fig, ax = WF.plot_decay(0, show=True)
fig, ax = WF.plot_density(0, show=True)
#fig, ax = WF.plot_centers(
#    color_home_cell=True, center_scale=15, legend=True, pmx=4, pmy=4, show=True
#)

