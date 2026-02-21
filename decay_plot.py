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
wfa = WFArray(model.lattice, mesh)
wfa.solve_model(model)

