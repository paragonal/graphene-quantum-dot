import matplotlib.pyplot as plt
import numpy as np
import kwant
import scipy.sparse.linalg as sla
import scipy.linalg as la
from tqdm import tqdm
from math import *
import scipy.integrate as integrate

k_b = 1.380649e-23  # m^2 kg/s^2/K
q_e = 1.6e-19
h = 6.626176e-34
hbar = 1.054571e-34


def make_system(r, t, tp, left_pot, right_pot, leads):
    side = -1

    a = 2.46  # angstroms
    # Define the graphene lattice
    sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
    lat = kwant.lattice.general([(a, 0), (a * sin_30, a * cos_30)],
                                [(0, 0), (0, a / sqrt(3))])
    electrode_bound = 1.5 * r
    electrode_slope = 1 / 2

    b, c = lat.sublattices

    def hopx(site1, site2, B):
        y = site1.pos[0]
        assert (site1.pos[0] == site2.pos[0])  # this must be true for our field
        return t * np.exp(-1j * B * y * q_e / hbar)  # need to multiply by some Amps / Newton constant here
        # try e / (hbar c)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    # side = -1 for left, 1 for right
    def electrode_shape(pos):
        x, y = pos
        if x ** 2 + y ** 2 < r ** 2:
            return False
        return abs(y) + r / 4 < side * x * electrode_slope and abs(x) < electrode_bound

    def make_lead(pot):
        # Start with an empty lead with a single square lattice
        lead_lattice = kwant.lattice.honeycomb(a)

        sym_lead = kwant.TranslationalSymmetry(lead_lattice.vec((-1, 0)))
        lead = kwant.Builder(sym_lead)

        # build up one unit cell of the lead, and add the hoppings
        # to the next unit cell
        def lead_shape(pos):
            x, y = pos
            return abs(y) + r / 4 < electrode_slope * electrode_bound - 2

        lead[lead_lattice.shape(lead_shape, (0, 0))] = pot

        lead[lead_lattice.neighbors()] = t
        return lead

    def potential(site):
        x, y = site.pos
        m = (left_pot - right_pot) / (2 * r)
        return -m * (x - r) + right_pot

    syst = kwant.Builder()
    syst[lat.shape(circle, (0, 0))] = potential

    syst[lat.shape(electrode_shape, (-r + side, 0))] = left_pot
    side *= -1
    syst[lat.shape(electrode_shape, (r + side, 0))] = right_pot

    hoppings = (((0, 1), b, c), ((-1, 1), b, c))
    syst[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = t
    ##
    hoppings_with_field = ((0, 0), b, c)
    syst[kwant.builder.HoppingKind(*hoppings_with_field)] = hopx

    syst.eradicate_dangling()

    if leads:
        syst.attach_lead(make_lead(pot=left_pot))
        syst.attach_lead(make_lead(pot=right_pot).reversed())

    if tp:
        syst[lat.neighbors(2)] = tp

    return syst


def hopping_lw(site1, site2):
    return 0.04 if site1.family == site2.family else 0.1


def sort_evals(vals, vecs):
    vecs = [vecs[:, n] for n in range(len(vals))]
    print(np.shape(vecs), np.shape(vals))
    temp = sorted(zip(vals, vecs))
    vecs = list(np.array(temp)[:, 1])
    return np.array(temp)[:, 0], vecs


def plot_data(syst, ns):
    syst = syst.finalized()
    ham = syst.hamiltonian_submatrix(sparse=True)
    print("Solving N={} matrix".format(ham.shape[0]))
    evals, evecs = sla.eigsh(ham, max(ns) + 1)
    evals, evecs = sort_evals(evals, evecs)
    for n in ns:
        wf = abs(evecs[n]) ** 2

        def site_size(i):
            return 3 * wf[i] / wf.max()

        kwant.plot(syst, site_size=site_size, site_color=(0, 1, 1, 0.3),
                   hop_lw=0.1, fig_size=(6, 6))
    plt.plot(evals)
    plt.show()


default_B = 0  # tesla meter
t = -2.7
doping_level = -0.5  # eV
v_sd = 0  # V
pot = doping_level / abs(t)

params = dict(B=default_B)


def fermi(eps, mu, T):
    eps = eps * abs(t) * q_e
    mu = mu * abs(t) * q_e
    return 1 / (1 + np.exp((eps - mu) / (k_b * T)))


def plot_conductance(syst, energies, mu_left, mu_right, temp, silent=False):
    # Compute transmission as a function of energy
    data = []
    data2 = []
    modes = []
    current_sum = 0
    dE = energies[1] - energies[0]

    for energy in energies:
        smatrix = kwant.smatrix(syst, energy, params=params)

        # print(smatrix.transmission(0,0) + smatrix.transmission(0,1))
        modes.append(smatrix.num_propagating(0))
        data.append(smatrix.transmission(1, 0))  # *q_e**2/h)
        current_sum += dE * data[-1] * (fermi(energy * abs(t),
                                              mu_left * abs(t), temp) - fermi(energy * abs(t), mu_right * (abs(t)),
                                                                              temp))
        data2.append(data[-1] * (fermi(energy, mu_left, temp) - fermi(energy, mu_right, temp)))
        # smatrix nmodes
    if not silent:
        fig, axs = plt.subplots(3, figsize=(10, 10))
        axs[0].plot(energies * abs(t), data)

        axs[0].set_xlabel("energy [eV]")
        axs[0].set_ylabel("conductance [S]")

        axs[1].set_ylabel("Number of available modes")
        axs[1].plot(energies * abs(t), modes)
        axs[1].grid()

        axs[2].plot(energies * abs(t), data2)
        axs[2].set_ylabel("Contribution to total current")

        plt.tight_layout()
        plt.savefig("graphs/conductance plot")
        plt.show()
    return current_sum, energies, data


def plot_current_density(fsyst, energy):
    # Compute the scattering wave function due to the modes in lead 0 at given energy.
    psi = kwant.wave_function(fsyst, energy, params=params)(0)
    # Create a current operator.
    J = kwant.operator.Current(fsyst)
    # Compute the current observable.
    current = sum(J(p) for p in psi)
    # Plot the current as a stream plot.
    kwant.plotter.current(fsyst, current)


def plot_bandstructure(flead, momenta):
    bands = kwant.physics.Bands(flead)
    energies = [bands(k) * abs(t) for k in momenta]

    plt.figure(figsize=(10, 6))
    plt.plot(momenta, energies, alpha=0.5)

    # plt.plot(momenta, [doping_level - 0.3/2 for _ in range(len(momenta))], label = "$v_sd$=0.3 V")
    # plt.plot(momenta, [doping_level - 1/2 for _ in range(len(momenta))], label = "$v_sd$=1 V")
    plt.legend()
    plt.xlabel("momentum [(lattice constant)^-1]")
    plt.ylabel("energy [ev]")
    plt.ylim(-2.5, 2.5)
    plt.grid()
    plt.show()


def make_IV_curve(voltages, temp, span_override= None):
    currents = []
    if -pot * 2 in voltages:
        print("Remove 0 term otherwise we get 0 momentum modes")
    for v_sd in voltages:
        left_pot = v_sd / 2 / abs(t)
        right_pot = -v_sd / 2 / abs(t)
        syst = make_system(r=150, t=-t, left_pot=left_pot, right_pot=right_pot, tp=None, leads=True)

        syst = syst.finalized()

        if span_override == None:
            span = left_pot - right_pot
        else:
            span = span_override / abs(t)

        energies = np.linspace(pot - span, pot + span, 70)
        current, _, _ = \
            plot_conductance(syst, energies, pot + left_pot, pot + right_pot, temp, silent=True)
        print("Current for {}V = {}".format(v_sd, current))
        currents.append(current)
    return currents


voltages = np.linspace(0, 2.3, 2)
frames = []
for v_sd in voltages:
    temp = 50

    left_pot = v_sd / 2 / abs(t)
    right_pot = -v_sd / 2 / abs(t)
    syst = make_system(r=150, t=-t, left_pot=left_pot, right_pot=right_pot, tp=None, leads=True)

    syst = syst.finalized()

    span = 6 / abs(t)

    energies = list(np.linspace(pot - span, pot + span, 51))

    current, energies, conductances = \
        plot_conductance(syst, energies, pot + left_pot, pot + right_pot, temp, silent=True)
    frames.append((current, v_sd, energies, conductances))

    np.save("currents_energies_conductances_test", np.array(frames))
