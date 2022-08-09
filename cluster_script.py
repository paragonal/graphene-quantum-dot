#### See graphene quantum dot.ipynb for details about functions ###

import matplotlib.pyplot as plt
import numpy as np
import kwant
import scipy.sparse.linalg as sla
import scipy.linalg as la
from tqdm import tqdm
from math import *
import scipy.integrate as integrate
import tinyarray
import warnings

# suppress numpy overflow warnings
warnings.filterwarnings('ignore')

k_b = 1.380649e-23  # m^2 kg/s^2/K
q_e = 1.6e-19
h = 6.626176e-34  # in meter
hbar = 1.054571e-34  # in meters

sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


def make_system(r, t, tp, left_pot, right_pot, leads, parallel=True, lead_zeeman=True):
    norbs = 2
    side = -1
    # if r > 1:
    a = 2.46  # angstroms
    # Define the graphene lattice
    sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
    lat = kwant.lattice.general([(a, 0), (a * sin_30, a * cos_30)],
                                [(0, 0), (0, a / sqrt(3))], norbs=norbs)
    electrode_bound = 1.5 * r
    electrode_slope = 1 / 2

    b, c = lat.sublattices

    # hopping along the A field direction (perpendicular to B field)
    def hopy(site1, site2, B):
        x = site1.pos[0]
        return sigma_0 * t * np.exp(1j * B * x * q_e / hbar * a / sqrt(3))

    def hop_right(site1, site2, B):
        x_0 = site1.pos[0]
        const = -B * x_0 * a / sqrt(12) - B * a ** 2 / (4 * sqrt(12))  # computed with pierels integral
        return sigma_0 * t * np.exp(1j * const * q_e / hbar)

    def hop_left(site1, site2, B):
        x_0 = site1.pos[0]
        const = -B * x_0 * a / sqrt(12) + B * a ** 2 / (4 * sqrt(12))  # computed with pierels integral
        return sigma_0 * t * np.exp(1j * const * q_e / hbar)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 <= r ** 2

    # side = -1 for left, 1 for right
    def electrode_shape(pos):
        x, y = pos
        if x ** 2 + y ** 2 < r ** 2:
            return False
        return abs(y) + r / 4 < side / abs(side) * x * electrode_slope and abs(x) < electrode_bound

    def make_lead(pot):
        # Start with an empty lead with a single square lattice
        lead_lattice = kwant.lattice.honeycomb(a, norbs=norbs)

        sym_lead = kwant.TranslationalSymmetry(lead_lattice.vec((-1, 0)))
        lead = kwant.Builder(sym_lead, conservation_law=-sigma_z)

        def lead_shape(pos):
            x, y = pos
            return abs(y) + r / 4 < electrode_slope * electrode_bound - 0.5 * a

        def lead_potential(site, E_z):
            x, y = site.pos
            if not parallel and x < 0:
                E_z *= -1
            if not lead_zeeman:
                E_z = 0
            return pot * sigma_0 + E_z * sigma_z

        lead[lead_lattice.shape(lead_shape, (0, 0))] = lead_potential

        lead[lead_lattice.neighbors()] = t * sigma_0
        return lead

    def potential(site, E_z):
        x, y = site.pos
        m = (left_pot - right_pot) / (2 * r)
        if not parallel and x < 0:
            E_z *= -1
        return (-m * (x - r) + right_pot) * sigma_0 + E_z * sigma_z

    def electrode_potential(site, E_z):
        x, y = site.pos
        if x < 0:
            pot = left_pot
        else:
            pot = right_pot
        if not parallel and x < 0:
            E_z *= -1
        if not lead_zeeman:
            E_z = 0

        return pot * sigma_0 + E_z * sigma_z

    syst = kwant.Builder()
    syst[lat.shape(circle, (0, 0))] = potential

    if leads:
        syst[lat.shape(electrode_shape, (-r, 0))] = electrode_potential
        side *= -1
        syst[lat.shape(electrode_shape, (r, 0))] = electrode_potential

    ## hoppings
    hoppings_up = ((0, 0), b, c)
    syst[kwant.builder.HoppingKind(*hoppings_up)] = hopy

    hoppings_left = ((0, 1), b, c)
    syst[kwant.builder.HoppingKind(*hoppings_left)] = hop_left

    hoppings_right = ((-1, 1), b, c)
    syst[kwant.builder.HoppingKind(*hoppings_right)] = hop_right
    ## hop_right and left might be swapped, pretty sure they are correct though

    syst.eradicate_dangling()

    if leads:
        syst.attach_lead(make_lead(pot=left_pot))
        syst.attach_lead(make_lead(pot=right_pot).reversed())

    if tp:
        syst[lat.neighbors(2)] = tp

    return syst



def fermi(eps, mu, T):
    eps *= ev_to_j
    mu *= ev_to_j
    return 1/(1+np.exp((eps-mu)/(k_b*T)))

def plot_conductance(syst, energies, mu_left, mu_right, temp, params, silent=False):
    # Compute transmission as a function of energy
    spin_ups = []
    spin_downs = []
    down_up = []
    up_down = []
    total = []
    conductance_quantum = q_e**2/h
    data2 = []
    current_sum = 0
    dE = energies[1] - energies[0]

    iter_energies = energies
    if not silent:
        iter_energies = tqdm(iter_energies)

    for energy in iter_energies:
        smatrix = kwant.smatrix(syst, energy, params = params)

        # total conductance
        spin_ups.append(smatrix.transmission((1,0), (0,0)))
        spin_downs.append(smatrix.transmission((1,1), (0,1)))

        down_up.append(smatrix.transmission((1,0), (0,1)))
        up_down.append(smatrix.transmission((1,1), (0,0)))
        total.append(smatrix.transmission(1,0))

        if down_up[-1] != 0 or up_down[-1] != 0:
            print("something odd is afoot")


        occupation_difference = (fermi(energy,mu_left, temp) - fermi(energy, mu_right, temp))
        current_sum += dE * total[-1] * occupation_difference / q_e * conductance_quantum
        # data2.append(dE * total[-1] * conductance_quantum * occupation_difference / q_e)
        data2.append(occupation_difference)

    if not silent:
        fig, axs = plt.subplots(2,figsize=(10,10))
        axs[0].plot(energies, np.array(spin_ups) , label="Up")
        axs[0].plot(energies, np.array(spin_downs) , label = "Down")
        axs[0].plot(energies, np.array(down_up), label="du")
        axs[0].plot(energies, np.array(up_down), label = "ud")
        axs[0].plot(energies, np.array(total) , label = "total")
        axs[0].set_xlabel("energy [eV]")
        axs[0].set_ylabel("conductance [$q^2$/$\hbar$]")
        axs[0].legend()

        axs[1].plot(energies, data2)
        axs[1].set_ylabel("Contribution to total current")

        plt.tight_layout()
        plt.savefig("graphs/conductance plot")
        plt.show()
    return current_sum, energies, total, spin_downs, spin_ups


t = -2.7
temp = 50
doping_level = -.5
voltages = np.linspace(0, 0.7, 150)

### Blocks to create IV Curves ###
### Create parallel IV curve, saving as we go ###
frames = []
for v_sd in voltages:
    left_pot = v_sd / 2
    right_pot = -v_sd / 2
    span = max(1, left_pot - right_pot)

    syst = make_system(r=50, t=-t, left_pot=left_pot, right_pot=right_pot, tp=None, leads=True,
                       parallel=True, lead_zeeman=True)
    syst = syst.finalized()

    energies = np.linspace(-span + doping_level, span + doping_level, 200)
    params = dict(B=0, E_z=0.01)

    out = \
        plot_conductance(syst, energies, doping_level + left_pot, doping_level + right_pot, temp,
                         params=params, silent=True)
    frames.append(out)
    np.save("IV_Curve_Parallel_Z10meV", np.array(frames))

### Create antiparallel IV curve ###
frames = []
for v_sd in voltages:
    left_pot = v_sd / 2
    right_pot = -v_sd / 2
    span = max(1, left_pot - right_pot)

    syst = make_system(r=50, t=-t, left_pot=left_pot, right_pot=right_pot, tp=None, leads=True,
                       parallel=False, lead_zeeman=True)
    syst = syst.finalized()

    energies = np.linspace(-span + doping_level, span + doping_level, 200)
    params = dict(B=0, E_z=0.01)

    out = \
        plot_conductance(syst, energies, doping_level + left_pot, doping_level + right_pot, temp,
                         params=params, silent=True)
    frames.append(out)
    np.save("IV_Curve_Antiparallel_Z10meV", np.array(frames))