import sys
sys.path.append("../lib")
import spoc1 as sp1
import matplotlib.pyplot as plt
import numpy as np

mu = sp1.MU_TRAPPIST

vinf_max = np.array([3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2]) * 1e4
vinf_min = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) * 1e4

amin = sp1.planets[0].orbital_elements[0]

# a,e,i,W,w,M
vinf_n = 10
alpha_n = 50
alpha_vec = np.linspace(0, np.pi, 50)

# plot energy of all planets
plt.figure()

for i, planet in enumerate(sp1.planets):
    oe_kep = planet.orbital_elements
    r = oe_kep[0]  # assume zero eccentricity
    energy = -mu/2/r
    period = 2 * np.pi * np.sqrt(r**3/mu) / 24/ 3600
    plt.xlim(0,1)
    plt.ylabel('Periods [days]')
    plt.hlines(period, 0, 1, color=sp1.pl2c[planet.name], label=planet.name)
    plt.legend()

#plt.show()

plt.figure(figsize=(10,6))
for i, planet in enumerate(sp1.planets):
    oe_kep = planet.orbital_elements
    r = oe_kep[0]  # assume zero eccentricity
    v2 = np.sqrt(mu/r)
    vinf_vec = np.linspace(vinf_min[i], vinf_max[i], vinf_n)
    [Alpha, Vinf] = np.meshgrid(alpha_vec, vinf_vec)
    v_sc = np.sqrt(v2**2 + Vinf**2 + 2 * Vinf * v2 * np.cos(Alpha))
    a = 1/(2/r - v_sc**2/mu)
    period = 2 * np.pi * np.sqrt(a**3/mu) / 24/ 3600
    energy = - mu/2/a

    a_bar = a / r
    vinf_bar = Vinf / np.sqrt(mu/r)
    tmp1 = 3 - vinf_bar**2 - 1/a_bar
    tmp2 = tmp1**2/4/a_bar

    ecc = np.sqrt(1 - tmp2)
    peri_r = a * (1 - ecc)

    print("planet:", i, "max_ecc:", np.max(ecc), " min_ecc:", np.min(ecc), "energy:", -mu/2/r)

    # ax.set_xlim(0, 5)
    plt.xlabel('Periapsis Radius (/x planet_b)')
    plt.ylabel('Orbital Period')
    for j in range(vinf_n):
        xplot = peri_r[j,:] / amin
        # yplot = energy[j,:]
        yplot = period[j,:]
        plt.plot(xplot, yplot, color=sp1.pl2c[planet.name])
        # ax.hlines(-mu/2/r, 0, 5, 'k', linestyles='dashed')

plt.savefig("Tisserand.png")
plt.show()
print("Finshed!")