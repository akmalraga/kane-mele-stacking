from pythtb import TBModel, Lattice
import numpy as np

def set_model(t, soc, rashba, delta, W):
    sigma_z = np.array([0., 0., 0., 1.])
    sigma_x = np.array([0., 1., 0., 0])
    sigma_y = np.array([0., 0., 1., 0])

    r3h = np.sqrt(3.0) / 2.0
    sigma_a = 0.5 * sigma_x - r3h * sigma_y
    sigma_b = 0.5 * sigma_x + r3h * sigma_y
    sigma_c = -1.0 * sigma_x


    lat = [[1, 0, 0], [0.5, np.sqrt(3.0)/2.0, 0.0], [0.0, 0.0, 1.0]]

    orb = [[1./3., 1./3., 0.0], [2./3., 2./3., 0.0]]

    lattice = Lattice(lat_vecs=lat, orb_vecs=orb, periodic_dirs=...)

    model = TBModel(lattice=lattice, spinful=True)


    disorder_values = np.random.uniform(-W/2, W/2, size=len(orb))

    onsite_energies = [

        delta + disorder_values[i] if i % 2 == 0 else -delta + disorder_values[i]

        for i in range(len(orb))

    ]

    model.set_onsite(onsite_energies)


    # Hopping terms

    for lvec in ([0, 0, 0], [-1, 0, 0], [0, -1, 0]):

        model.set_hop(t, 0, 1, lvec)


    for lvec in ([1, 0, 0], [-1, 1, 0], [0, -1, 0]):

        model.set_hop(soc * 1.j * sigma_z, 0, 0, lvec)

    for lvec in ([-1, 0, 0], [1, -1, 0], [0, 1, 0]):

        model.set_hop(soc * 1.j * sigma_z, 1, 1, lvec)


    model.set_hop(0.1 * soc * 1j * sigma_z, 1, 1, [0, 0, 1])

    model.set_hop(-0.1 * soc * 1j * sigma_z, 0, 0, [0, 0, 1])


    model.set_hop(1.j * rashba * sigma_a, 0, 1, [0, 0, 0], mode="add")

    model.set_hop(1.j * rashba * sigma_b, 0, 1, [-1, 0, 0], mode="add")

    model.set_hop(1.j * rashba * sigma_c, 0, 1, [0, -1, 0], mode="add")


    return model

model = set_model(1.0, 0.3, 0.1, 1.0, 0.0)
model.info()
