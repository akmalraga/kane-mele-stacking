from pythtb import TBModel, WFArray, Mesh, Lattice
import numpy as np 
import matplotlib.pyplot as plt 
from HamiltonianModel.hamiltonian import set_model 

my_model = set_model(1.0, 0.3, 0.05, 1.0, 0)

# 1. Tentukan jumlah grid 2D yang kamu inginkan
nkx, nky = 40, 40 # Pakai 40 agar lebih halus (resolusi lebih tinggi)
kz_val = 0.0      # Kita ambil slice di bidang k_z = 0

# 2. Buat grid 2D manual untuk kx dan ky, tapi kz tetap konstan
# Ini trik agar tutorial 2D kamu tetap jalan di model 3D
kx = np.linspace(0, 1, nkx, endpoint=False)
ky = np.linspace(0, 1, nky, endpoint=False)
KX_mesh, KY_mesh = np.meshgrid(kx, ky)

# Susun k_pts menjadi (N, 3) karena modelmu minta 3 koordinat
k_pts = np.zeros((nkx * nky, 3))
k_pts[:, 0] = KX_mesh.flatten()
k_pts[:, 1] = KY_mesh.flatten()
k_pts[:, 2] = kz_val # Ini kunci agar model 3D tidak error

# 3. Hitung Berry Curvature pada k_pts buatan kita
b_curv = my_model.berry_curvature(k_pts=k_pts, plane=(0, 1), cartesian=True)

# 4. Sekarang tutorial 2D kamu akan jalan!
# Reshape k_pts ke (nkx, nky, 3) karena kita punya 3 koordinat k
k_pts_sq = k_pts.reshape((nkx, nky, 3))
b_curv_sq = b_curv.reshape((nkx, nky))

# Transformasi ke koordinat Kartesian untuk plotting yang benar
recip_lat_vecs = my_model.recip_lat_vecs
# Kita hanya butuh komponen x dan y dari hasil perkalian matriks
mesh_Cart = k_pts_sq @ recip_lat_vecs

KX = mesh_Cart[:, :, 0]
KY = mesh_Cart[:, :, 1]

# Plot menggunakan Z-axis (b_curv_sq)
plt.figure(figsize=(8,6))
im = plt.pcolormesh(KX, KY, b_curv_sq.real, cmap="plasma", shading="gouraud")
plt.colorbar(label=r"$\Omega_{xy}(\mathbf{k})$")
plt.savefig("result/berry_curvature.pdf")
plt.show()
