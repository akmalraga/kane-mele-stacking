from pythtb import TBModel, Lattice
import matplotlib.pyplot as plt
import numpy as np
from HamiltonianModel.hamiltonian import set_model as sm

sigma_z = np.array([0., 0., 0., 1.], dtype=complex)
sigma_x = np.array([0., 1., 0., 0], dtype=complex)
sigma_y = np.array([0., 0., 1., 0], dtype=complex)

r3h = np.sqrt(3.0) / 2.0
sigma_a = 0.5 * sigma_x - r3h * sigma_y
sigma_b = 0.5 * sigma_x + r3h * sigma_y
sigma_c = -1.0 * sigma_x

lat = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]
orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]
lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=...)

model = TBModel(lattice=lattice, spinful=True)

t = 1.0
soc = 1/4
m = 0.5
rashba = 0.05

model.set_onsite(
        lambda beta: [0, m * np.sin(beta), m * np.sin(beta), m * np.sin(beta)], ind_i=0,
        )
model.set_onsite(
        lambda beta: [0, -m * np.sin(beta), -m * np.sin(beta), -m * np.sin(beta)], ind_i=1,
        )

for lvec in ([-1, 0, 0], [0, -1, 0], [0, 0, -1]):
    model.set_hop(t, 0, 1, lvec)

model.set_hop(lambda beta: 3 * t + m * np.cos(beta), 0, 1, [0, 0, 0], mode="set")

lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])

for j in range(6):
    spin = np.array([0.0] + dir_list[j])
    model.set_hop(1j * soc * spin, 0, 0, lvec_list[j])
    model.set_hop(-1j * soc * spin, 1, 1, lvec_list[j])

model.set_hop(1.j * rashba * spin, 0, 1, [0, 0, 0], mode="add")
model.set_hop(1.j * rashba * spin, 0, 1, [-1, 0, 0], mode="add")
model.set_hop(1.j * rashba * spin, 0, 1, [0, -1, 0], mode="add")

print(model)

nks = 30, 30, 30
n_beta = 21
betas = np.linspace(0, 2 * np.pi, n_beta, endpoint=True)
param_periods = {"beta": 2 * np.pi}

print(f"Total number of points: {nks[0] * nks[1] * nks[2] * n_beta}")

betas, axion, c2 = model.axion_angle(
    nks=nks,
    param_periods=param_periods,
    return_second_chern=True,
    use_tensorflow=True,
    diff_scheme="central",
    diff_order=8,
    beta=betas,
)

print(f"Second Chern number C2 = {c2}")

fig, ax = plt.subplots()

ax.set_xlabel(r"$\beta$", size=15)
ax.set_ylabel(r"$\theta$", size=15)

tick_positions = np.arange(0, 2 * np.pi + np.pi / 4, np.pi / 4)
tick_labels = [
    r"$0$",
    r"$\frac{\pi}{4}$",
    r"$\frac{\pi}{2}$",
    r"$\frac{3\pi}{4}$",
    r"$\pi$",
    r"$\frac{5\pi}{4}$",
    r"$\frac{6\pi}{4}$",
    r"$\frac{7\pi}{4}$",
    r"$2\pi$",
]

# Set the ticks and labels for both axes
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.set_yticks(tick_positions)
ax.set_yticklabels(tick_labels)

## Riemann sum
ax.plot(betas, axion, lw=1, zorder=2, c="k")
ax.scatter(betas, axion, s=6, zorder=2, c="r")

ax.grid()
ax.set_title("Axion angle vs adiabatic parameter", size=12)
plt.savefig("result/axion_angle.pdf")
plt.show()
