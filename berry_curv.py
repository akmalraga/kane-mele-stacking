from pythtb import TBModel, WFArray, Mesh, Lattice
import numpy as np 
import matplotlib.pyplot as plt 
from HamiltonianModel.hamiltonian import set_model 

my_model = set_model(1.0, 0.3, 0.05, 1.0, 0)

nkx, nky = 40, 40 # Pakai 40 agar lebih halus (resolusi lebih tinggi)
ky_val = 0.0      # Kita ambil slice di bidang k_z = 0

kx = np.linspace(0, 1, nkx, endpoint=False)
kz = np.linspace(0, 1, nky, endpoint=False)
KX_mesh, KZ_mesh = np.meshgrid(kx, kz)

k_pts = np.zeros((1600, 3))
k_pts[:, 0] = KX_mesh.flatten()
k_pts[:, 1] = ky_val
k_pts[:, 2] = KZ_mesh.flatten() 


b_curv = my_model.berry_curvature(k_pts=k_pts, plane=(0, 2), cartesian=True)

k_pts_sq = k_pts.reshape((nkx, nky, 3))
b_curv_sq = b_curv.reshape((nkx, nky))

recip_lat_vecs = my_model.recip_lat_vecs
mesh_Cart = k_pts_sq @ recip_lat_vecs
KX = mesh_Cart[:, :, 0]
KZ = mesh_Cart[:, :, 2]

# XY-Plane
kz_val = 0.0      # Kita ambil slice di bidang k_z = 0

kx = np.linspace(0, 1, nkx, endpoint=False)
ky = np.linspace(0, 1, nky, endpoint=False)
KX_mesh, KY_mesh = np.meshgrid(kx, ky)

k_pts = np.zeros((1600, 3))
k_pts[:, 0] = KX_mesh.flatten()
k_pts[:, 1] = KY_mesh.flatten()
k_pts[:, 2] = kz_val


b_curv_xy = my_model.berry_curvature(k_pts=k_pts, plane=(0, 1), cartesian=True)

k_pts_sq_xy = k_pts.reshape((nkx, nky, 3))
b_curv_sq_xy = b_curv_xy.reshape((nkx, nky))

recip_lat_vecs = my_model.recip_lat_vecs
mesh_Cart = k_pts_sq_xy @ recip_lat_vecs
KX = mesh_Cart[:, :, 0]
KY = mesh_Cart[:, :, 1]

# 3D Plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=plt.figaspect(0.5))
fig.suptitle('Berry Curvature of XY-Plane and XZ-Plane', fontsize=16)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(KX, KZ, b_curv_sq.real, cmap='inferno')
ax1.set_xlabel(r'$k_x$')
ax1.set_ylabel(r'$k_z$')
ax1.set_zlabel(r'$\Omega_{xz}(\mathbf{k})$')
ax1.set_title('Berry Curvature in XZ-Plane', fontsize=14)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(KX, KY, b_curv_sq_xy.real, cmap='inferno')
ax2.set_xlabel(r'$k_x$')
ax2.set_ylabel(r'$k_y$')
ax2.set_zlabel(r'$\Omega_{xy}(\mathbf{k})$')
ax2.set_title('Berry Curvature in XY-Plane', fontsize=14)

data_to_save = np.column_stack((KX.flatten(), KZ.flatten(), b_curv_sq.flatten().real))
np.savetxt("result/berry_curvature_data.txt", data_to_save, header="kx ky BerryCurvature", comments="")

plt.savefig("result/berry_curvature_3D.pdf")
plt.show()
