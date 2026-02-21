from pythtb import Mesh, Wannier, WFArray 
from HamiltonianModel.hamiltonian import set_model 
import numpy as np 

prim_model = set_model(1.0, 0.3, 0.1, 1.0, 0.0)

n_super_cell = 2 
model = prim_model.make_supercell([[n_super_cell, 0, 0],[0, n_super_cell, 0],[0, 0, n_super_cell]])
model.info(show=True, short=False)

nks = 10, 10, 10 
mesh = Mesh(dim_k=3, axis_types=["k", "k", "k"])
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
        [(orb, 1)] for orb in low_E_sites
        ]

n_tfs = len(tf_list)

print(f"Trial Wavefunction: {tf_list}")
print(f"# of Wannier function: {n_tfs}")
print(f"# of occupied bands: {n_occ}")
print(f"Wannier fraction: {n_tfs / n_occ}")
