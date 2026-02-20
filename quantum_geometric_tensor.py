import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from HamiltonianModel.hamiltonian import set_model 

my_model = set_model(1.0, 0.3, 0.25, 1.0, 0)

nkx, nky, nkz = 10, 10, 10 
k_pts = my_model.k_uniform_mesh([nkx, nky, nkz])

g = my_model.quantum_geometric_tensor(k_pts=k_pts, occ_idxs=[0, 1], cartesian=True)
print(g.shape)


