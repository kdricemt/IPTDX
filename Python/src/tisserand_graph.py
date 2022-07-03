import matplotlib.pyplot as plt
import numpy as np
from pykep.core import AU

pl2c = ['coral', 'seagreen', 'purple', 'steelblue', 'olive', 'gray', 'salmon', 'cyan']


def tisserand_graph(planets, mu, epoch, vinf_min, vinf_max, safe_fb_rp=1.1, vinf_n=50, alpha_n=50,
                    xlabel="rp", ylabel="ra", xlim=None, ylim=None):

    alpha_vec = np.linspace(0, np.pi, alpha_n)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    plt.grid()

    for i, planet in enumerate(planets):
        oe_kep = planet.osculating_elements(epoch)
        r_p = planet.radius * safe_fb_rp
        mu_planet = planet.mu_self
        r = oe_kep[0]  # assume zero eccentricity
        v2 = np.sqrt(mu / r)
        vinf_vec = np.linspace(vinf_min[i], vinf_max[i], vinf_n)
        delta_max = 2 * np.arcsin(np.ones_like(vinf_vec) / (1 + r_p * vinf_vec ** 2 / mu_planet))
        [Alpha, Vinf] = np.meshgrid(alpha_vec, vinf_vec)
        v_sc = np.sqrt(v2 ** 2 + Vinf ** 2 + 2 * Vinf * v2 * np.cos(Alpha))
        a = 1 / (2 / r - v_sc ** 2 / mu)

        period = 2 * np.pi * np.sqrt(a ** 3 / mu) / 24 / 3600
        energy = - mu / 2 / a

        a_bar = a / r
        vinf_bar = Vinf / np.sqrt(mu / r)
        tmp1 = 3 - vinf_bar ** 2 - 1 / a_bar
        tmp2 = tmp1 ** 2 / 4 / a_bar

        ecc = np.sqrt(1 - tmp2)
        peri_r = a * (1 - ecc)
        apo_r = a * (1 + ecc)

        # ax.set_xlim(0, 5)
        if xlabel=="rp":
            ax.set_xlabel('Periapsis Radius (AU)')
            xplot =peri_r / AU
        elif xlabel=="ra":
            ax.set_xlabel('Apoapsis Radius (AU)')
            xplot = apo_r / AU
        elif xlabel=="energy":
            ax.set_xlabel('Energy')
            xplot = energy
        elif xlabel=="period":
            ax.set_xlabel('Period (days)')
            xplot = period
        else:
            print("Invalid plot label -  Choose from 'peri', 'apo', 'energy', 'period'")
            return fig, ax

        if ylabel=="rp":
            ax.set_ylabel('Periapsis Radius (AU)')
            yplot = peri_r / AU
        elif ylabel=="ra":
            ax.set_ylabel('Apoapsis Radius (AU)')
            yplot = apo_r / AU
        elif ylabel=="energy":
            ax.set_ylabel('Energy')
            yplot = energy
        elif ylabel=="period":
            ax.set_ylabel('Period (days)')
            yplot = period
        else:
            print("Invalid plot label -  Choose from 'peri', 'apo', 'energy', 'period'")
            return fig, ax

        min_energy = np.zeros(vinf_n)
        max_ecc = np.zeros(vinf_n)
        min_ecc = np.zeros(vinf_n)

        create_legend = True
        for j in range(vinf_n):
            ecc_bound = np.where(ecc[j, :] < 1.0)[0]

            if ecc_bound.size > 0:
                min_energy[j] = np.min(energy[j, ecc_bound])
                max_ecc[j] = np.max(ecc[j, ecc_bound])
                min_ecc[j] = np.min(ecc[j, ecc_bound])

                # yplot = period[j,ecc_bound]
                if create_legend:
                    ax.plot(xplot[j, ecc_bound], yplot[j, ecc_bound], color=pl2c[i], label=planet.name)
                    create_legend = False
                else:
                    ax.plot(xplot[j, ecc_bound], yplot[j, ecc_bound], color=pl2c[i], label='_nolegend_')
            else:
                min_energy[j] = np.inf
                max_ecc[j] = 0
                min_ecc[j] = np.inf

        min_energy = np.min(min_energy)
        max_ecc = np.max(max_ecc)
        min_ecc = np.min(min_ecc)
        print("planet:", planet.name, "max_ecc:", max_ecc, " min_ecc:", min_ecc, "Min Energy:", min_energy)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.legend()
    plt.show()

    return fig, ax

