"""
This module contains classes and functions helping to simulate charge breeding in an EBIS
including effects of dielectronic recombination
"""
import os
# import multiprocessing as mp

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import scipy.integrate
import scipy.constants
import scipy.stats
import scipy.interpolate

##### Physical constants
Q_E = scipy.constants.elementary_charge
M_E = scipy.constants.electron_mass
PI = scipy.constants.pi
EPS_0 = scipy.constants.epsilon_0
K_B = scipy.constants.Boltzmann
C_L = scipy.constants.speed_of_light

M_E_EV = M_E * C_L**2 / Q_E # Electron mass in eV

##### Logic for robust file imports
_MODULEDIR = os.path.dirname(os.path.abspath(__file__))
_RESOURCEDIR = os.path.join(_MODULEDIR, "resources/")
_GETRESDIR = lambda f: os.path.join(_RESOURCEDIR, f)

##### Helper functions for translating chemical symbols
def _load_chemical_elements():
    """
    Reads atomic number Z, symbol, and name information from external file
    """
    _z = [] # Atomic Number
    _es = [] # Element Symbol
    _name = [] # Element Name
    with open(_GETRESDIR("./ChemicalElements.csv")) as f:
        for line in f:
            data = line.split(",")
            _z.append(int(data[0]))
            _es.append(data[1])
            _name.append(data[2].strip())
    return (_z, _es, _name)

(_ELEM_Z, _ELEM_ES, _ELEM_NAME) = _load_chemical_elements()

def element_z(element):
    """
    Returns the Atomic Number of the Element, can give name or Element Symbol
    """
    if len(element) < 3:
        idx = _ELEM_ES.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_Z[idx]

def element_symbol(element):
    """
    Returns the Symbol of the Element, can give name or Atomic Number
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_ES[idx]

def element_name(element):
    """
    Returns the Name of the Element, can give Symbol or Atomic Number
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_ES.index(element)
    return _ELEM_NAME[idx]

##### Plotting functions
def plot_charge_state_evolution(ode_solution, xlim=(1e-4, 1e3), ylim=(1e-4, 1),
                                title="Charge State Evolution"):
    """
    Method that plots the solution of an EBIS charge breeding simulation
    """
    fig = plt.figure(figsize=(8, 6), dpi=150)
    for cs in range(ode_solution.y.shape[0]):
        plt.semilogx(ode_solution.t, ode_solution.y[cs, :], label=str(cs) + "+")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(which="both")
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Abundance")
    return fig

# ##### Electron Beam Physics
# def electron_velocity(e_kin):
#     """
#     Returns the electron velocity corresponding to a kin. energy in eV
#     """
#     return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)

# def herrmann_radius(cur, e_kin, b, r_c, t_c, b_c):
#     v = electron_velocity(e_kin)
#     s1 = M_E * cur / (PI * EPS_0 * Q_E * v * b**2)
#     s2 = 8 * K_B * t_c * M_E * r_c**2 / (Q_E**2 * b**2)
#     s3 = b_c**2 * r_c**4 / (b**2)
#     return np.sqrt(s1 + np.sqrt(s1**2 + s2 + s3))

# def rex_herrmann_radius(cur, e_kin):
#     b = 2
#     r_c = .8/1000
#     t_c = 1600
#     b_c = .2
#     return herrmann_radius(cur, e_kin, b, r_c, t_c, b_c)


##### Class Definitions
class ElectronBeam:
    """
    This class contains logic that allows estimating the space charge corrected energy
    of an electron beam and the resulting energy spread
    """
    def __init__(self, cur, b_d, r_d, b_c, r_c, t_c):
        """
        Set the constant machine parameters

        Input Parameters
        cur - The electron current in A
        b_d - magnetic flux density in the trap centre in Tesla
        r_d - drift tube radius in m
        r_c - cathode radius in m
        t_c - cathode temperatur in K
        b_c - magnetic flux density on the cathode surface
        """
        self._cur = cur
        self._b_d = b_d
        self._r_d = r_d
        self._r_c = r_c
        self._t_c = t_c
        self._b_c = b_c

    @property
    def current(self):
        """
        Can read and write a new current to the electron beam object, which will then be used
        for future computations
        """
        return self._cur

    @current.setter
    def current(self, val):
        self._cur = val

    @staticmethod
    def electron_velocity(e_kin):
        """
        Returns the electron velocity corresponding to a kin. energy in m/s

        Input Parameters
        e_kin - electron energy in eV
        """
        return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)

    def characteristic_potential(self, e_kin):
        """
        Returns the characteristic potential due to spacecharge in V

        Input Parameters
        e_kin - electron energy in eV
        """
        v = ElectronBeam.electron_velocity(e_kin)
        return self._cur / (4 * PI * EPS_0 * v)

    def herrmann_radius(self, e_kin):
        """
        Returns the Hermann radius of an electron beam with the given machine parameters in m

        Input Parameters
        e_kin - electron energy in eV
        """
        v = ElectronBeam.electron_velocity(e_kin)
        s1 = M_E * self._cur / (PI * EPS_0 * Q_E * v * self._b_d**2)
        s2 = 8 * K_B * self._t_c * M_E * self._r_c**2 / (Q_E**2 * self._b_d**2)
        s3 = self._b_c**2 * self._r_c**4 / (self._b_d**2)
        return np.sqrt(s1 + np.sqrt(s1**2 + s2 + s3))

    def space_charge_correction(self, e_kin, r=0):
        """
        Returns the space charge correction at a given radius r, current and electron beam energy
        This is done by iteratively computing the spacecharge correction and the hermann radius
        until the realtive change in space charge correction is < 1e-6

        Input Parameters
        e_kin - electron energy in eV
        r - distance from axis in m, defaults to 0
        """
        if r > self._r_d or r < 0:
            raise ValueError("r cannot be bigger than the drift tube radius or smaller than 0")

        # Init iterative solution
        # This is mainly required to adjust phi_0 as a function of e_kin
        # the hermann radius is essentially constant over large dynamic ranges
        sc_on_ax_new = 1
        sc_on_ax_old = 0
        while (sc_on_ax_new - sc_on_ax_old)/sc_on_ax_new > 1e-6: # Check relative difference
            # Compute kinetic energy correction
            corr_e_kin = e_kin + sc_on_ax_new
            # Compute Herrmann radius and characteristic potential
            r_e = self.herrmann_radius(corr_e_kin)
            phi_0 = self.characteristic_potential(corr_e_kin)
            # Compute space charge correction
            sc_on_ax_old = sc_on_ax_new
            sc_on_ax_new = phi_0 * (2 * np.log(r_e / self._r_d) - 1)

        # When loop is exited r_e and phi_0 should have the right values
        if r < r_e:
            multip = 2 * np.log(r_e / self._r_d) + (r / r_e)**2 - 1
        else:
            multip = 2 * np.log(r / self._r_d)
        return phi_0 * multip

class RexElectronBeam(ElectronBeam):
    """
    ElectronBeam Class with REXEBIS Parameters set by default
    """
    def __init__(self, cur):
        """
        Set the constant machine parameters

        Input Parameters
        cur - The electron current in A
        """
        b_d = 2
        r_d = 5/1000
        r_c = .8/1000
        t_c = 1600
        b_c = .2
        super().__init__(cur, b_d, r_d, b_c, r_c, t_c)

class LotzCrossSections:
    """
    An object that provides convenient handling of the Lotz cross sections
    UNIT: cm^2
    """
    def __init__(self, element):
        """
        Initiates the object by importing the relevant binding energies
        from FAC simulation results (performed by Robert Mertzig)

        Input Parameters
        element - Atomic Number or Element Symbol or Element object
        """
        if not isinstance(element, ElementProperties):
            element = ElementProperties(element)
        self._z = element.atomic_number
        self._es = element.symbol
        self._name = element.name

        self._cache = {} # Cache for Cross Section Matrices

        self._load_binding_energies()
        self._load_electron_configuration()

    def _load_binding_energies(self):
        """
        Private Helper Method for loading the binding energies
        """
        # Import binding energies for each electron in all charge states
        # list of lists where each sublist hold the energies for one charge state
        # self._e_bind[n] describes charge state n+
        self._e_bind = []
        with open(_GETRESDIR("./BindingEnergies/%d.txt" % self._z)) as fobj:
            for line in fobj:
                line = line.split()
                line = [float(elem.strip()) for elem in line]
                self._e_bind.append(line)

    def _load_electron_configuration(self):
        """
        Private Helper Method for loading the electron configuration
        """
        # Import Electron Configurations for each charge state
        # list of lists where each sublist hold the configuration for on charge state
        # self._cfg[n] describes charge state n+
        self._cfg = []
        with open(_GETRESDIR("./BindingEnergies/%dconf.txt" % self._z)) as fobj:
            for line in fobj:
                line = line.split()
                line = [int(elem.strip()) for elem in line]
                self._cfg.append(line)

    def cross_section(self, cs, e_kin):
        """
        Computes the Lotz cross section of a given charge state at a given electron energy
        UNIT: cm^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        cross_sect = 0
        for e_bind, num_el in zip(self._e_bind[cs], self._cfg[cs]):
            if e_kin > e_bind and num_el > 0:
                cross_sect += num_el * np.log(e_kin / e_bind) / (e_kin * e_bind)
        cross_sect *= 4.5e-14
        return cross_sect

    def cross_section_matrix(self, e_kin):
        """
        Computes and returns a matrix with the lotz cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are cached for better performance, be careful as this can occupy memory when a lot
        of energies are polled over time

        UNIT: cm^2

        Input Parameters
        e_kin - Electron impact energy
        """
        # Check cache
        if e_kin in self._cache.keys():
            return self._cache[e_kin]

        # If nonexistent, compute:
        cross_mat = np.zeros((self._z+1, self._z+1))
        cross_sec = np.zeros(self._z)

        for cs in range(self._z):
            cross_sec[cs] = self.cross_section(cs, e_kin)

        for cs in range(self._z+1):
            if cs > 0:
                cross_mat[cs, cs-1] = cross_sec[cs-1]
            if cs < self._z:
                cross_mat[cs, cs] = -cross_sec[cs]

        self._cache[e_kin] = cross_mat
        return cross_mat

    def create_plot(self, xlim=None, ylim=None):
        """
        Creates a figure showing the Lotz Cross section and returns the figure handle

        Input Parameters
        xlim, ylim - plot limits (optional)
        """
        e_min = self._e_bind[0][-1]/10
        e_max = 10*self._e_bind[-1][0]
        e_max = 10**np.ceil(np.log10(e_max))
        if xlim:
            e_min = np.min(e_min, xlim[0])
            e_max = np.max(e_max, xlim[1])
        else:
            xlim = (1, e_max)
        ene = np.logspace(np.log10(e_min), np.log10(e_max), 5000)

        fig = plt.figure(figsize=(8, 6), dpi=150)
        for cs in range(self._z):
            res = []
            for e in ene:
                res.append(self.cross_section(cs, e))
            plt.loglog(ene, np.array(res), label=str(cs)+"+")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(self._name + " (Z = " + str(self._z) + ") Lotz ionisation cross sections")
        plt.xlabel("Electron impact energy (eV)")
        plt.ylabel("Cross section (cm$^2$)")
        plt.grid(which="both")
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        return fig

    def show_plot(self, xlim=None, ylim=None):
        """
        Creates a figure showing the Lotz Cross section and calls the show method

        Input Parameters
        xlim, ylim - plot limits (optional)
        """
        self.create_plot(xlim, ylim)
        plt.show()

class KLLCrossSections:
    """
    An object that provides convenient handling of the KLL DR cross sections
    UNIT: cm^2
    """
    def __init__(self, element):
        """
        Initiates the object by importing the relevant KLL DR information
        extracted from FAC simulation results (performed by Steffen Kuehn)

        Input Parameters
        element - Atomic Number or Element Symbol or Element object
        """
        if not isinstance(element, ElementProperties):
            element = ElementProperties(element)
        self._z = element.atomic_number
        self._es = element.symbol
        self._name = element.name

        self._cache = {} # Cache for Cross Section Matrices

        self._load_kll_data()

    def _import_kll_block(self, fname):
        """
        Private Helper Method for importing the prepared DR data (for one charge state)

        Input parameters
        fname - filename
        """
        frame = pd.read_csv(fname, sep=' ', index_col=False)
        return frame

    def _load_kll_data(self):
        """
        Private Helper Method for loading the DR Data
        """
        # Convenient expression that generates the full path to the DR files in a robust manner
        kll_path_gen = lambda x: _GETRESDIR("./DRDataKLLGroundState/"
                                            + self._es + "." + x + ".dr.txt")
        # Load files
        he_like = self._import_kll_block(kll_path_gen("He"))
        li_like = self._import_kll_block(kll_path_gen("Li"))
        be_like = self._import_kll_block(kll_path_gen("Be"))
        b_like = self._import_kll_block(kll_path_gen("B"))
        c_like = self._import_kll_block(kll_path_gen("C"))
        n_like = self._import_kll_block(kll_path_gen("N"))
        o_like = self._import_kll_block(kll_path_gen("O"))

        self._kll_frames = {2:he_like, 3:li_like, 4:be_like,
                            5:b_like, 6:c_like, 7:n_like, 8:o_like}
        self._kll_names = {2:"He-like", 3:"Li-like", 4:"Be-like",
                           5:"B-like", 6:"C-like", 7:"N-like", 8:"O-like"}

    def _gaussian_line(self, mean, fwhm, strength):
        """
        Private Helper function
        Comfortable handle to create a gaussian profile function

        Input Parameters
        mean - mean of the gaussian
        fwhm - Full width half max
        strength - Strength of the line (area under curve)
        """
        sig = fwhm/(2*np.sqrt(2*np.log(2)))
        def g(x):
            """a gaussian line function with a given width and strength"""
            return strength * scipy.stats.norm.pdf(x, loc=mean, scale=sig)
        return g

    def _spectrum(self, line_list):
        """
        Private Helper function
        Generates the sum of a list of gaussian lines

        Input parameters
        line_list - list of line profiles to add to a single function
        """
        def spec(x):
            """function holding the sum of gaussians for each transition"""
            return sum(l(x) for l in line_list)
        return spec

    def _cross_section_func(self, rem_el, fwhm):
        """
        Helper function returning a function handle to a function computing the cross section
        for a given number of remaining electrons and energy spread
        """
        kll_fr = self._kll_frames[rem_el]
        lines = []
        for (_, row) in kll_fr.iterrows():
            # Rescale FAC cross sections (10^-20cm^2) to cm^2
            lines.append(self._gaussian_line(row.DELTA_E, fwhm, row.DR_RECOMB_STRENGTH * 1e-20))
        return self._spectrum(lines) # Handle to the function that can compute cross_sec

    def cross_section(self, cs, e_kin, fwhm):
        """
        Computes the Lotz cross section of a given charge state at a given electron energy
        UNIT: cm^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        fwhm - width of the spectral lines (gaussian electron energy spread)
        """
        rem_el = self._z - cs
        if rem_el not in range(2, 8+1):
            return 0
        cs_spectrum = self._cross_section_func(rem_el, fwhm)
        return cs_spectrum(e_kin)

    def cross_section_matrix(self, e_kin, fwhm):
        """
        Computes and returns a matrix with the KLL cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are cached for better performance, be careful as this can occupy memory when a lot
        of energies / fwhms are polled over time

        UNIT: cm^2

        Input Parameters
        e_kin - Electron impact energy
        fwhm - width of the spectral lines (gaussian electron energy spread)
        """
        # Check cache
        if (e_kin, fwhm) in self._cache:
            return self._cache[(e_kin, fwhm)]

        # If nonexistent, compute:
        A = np.zeros((self._z+1, self._z+1))
        for rem_el in range(2, 8+1):
            cs = self._z - rem_el
            cross_sec = self.cross_section(cs, e_kin, fwhm)
            A[cs, cs] = -cross_sec
            A[cs-1, cs] = cross_sec

        self._cache[(e_kin, fwhm)] = A
        return A

    def create_plot(self, fwhm, xlim=None, ylim=None):
        """
        Creates a figure showing the KLL Cross section and returns the figure handle

        Input Parameters
        xlim, ylim - plot limits (optional)
        fwhm - width of the spectral lines (gaussian electron energy spread)
        """
        e_min = self._kll_frames[2].DELTA_E.min() - 3 * fwhm
        e_max = self._kll_frames[8].DELTA_E.min() + 3 * fwhm
        if xlim:
            e_min = np.min(e_min, xlim[0])
            e_max = np.max(e_max, xlim[1])
        else:
            xlim = (e_min, e_max)
        ene = np.arange(e_min, e_max)

        fig = plt.figure(figsize=(12, 4), dpi=150)
        # x = np.arange(2300, 2800)
        for rem_el in range(2, 8+1):
            func = self._cross_section_func(rem_el, fwhm)
            plt.plot(ene, 1e20 * func(ene), label=self._kll_names[rem_el])
        # for sp, na in zip(dr_spectra(fwhm), dr_names):
        #     plt.plot(x, sp(x), label=na)
        plt.legend()
        plt.xlabel('Electron beam energy (eV)')
        plt.ylabel('Cross Section ($10^{-20}$ cm$^2$)')
        plt.title(self._name + " (Z = "
                  + str(self._z) + ") KLL-Recombination Cross Sections / Electron Energy FWHM = "
                  + str(fwhm) + " eV")
        plt.tight_layout()
        plt.grid(which="both")
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        return fig

    def show_plot(self, fwhm, xlim=None, ylim=None):
        """
        Creates a figure showing the KLL Cross section and calls the show method

        Input Parameters
        xlim, ylim - plot limits (optional)
        fwhm - width of the spectral lines (gaussian electron energy spread)
        """
        self.create_plot(fwhm, xlim, ylim)
        plt.show()

class ElementProperties:
    """
    Simple class holding some essential information about a chemical element
    """
    def __init__(self, element_id):
        """
        Initiates the object translating the input parameter and finding the other quantities

        Input Parameters
        element_id - Atomic Number or Element Symbol
        """
        if isinstance(element_id, int):
            self._z = element_id
            self._es = element_symbol(element_id)
        else:
            self._z = element_z(element_id)
            self._es = element_id
        self._name = element_name(self._z)

    @property
    def atomic_number(self):
        """Returns the atomic number"""
        return self._z

    @property
    def name(self):
        """Returns the name"""
        return self._name

    @property
    def symbol(self):
        """Returns the chemical symbol"""
        return(self)._es

class EBISSpecies:
    """
    collection of properties relevant to an atomic species in an EBIS for solving rate equations
    """
    def __init__(self, element_id):
        """
        Creates the species by defining the element and automatically creating objects for the
        Lotz and KLL cross section

        Input Parameters
        element_id - Atomic Number or Element Symbol
        """
        self._element_properties = ElementProperties(element_id)
        self._lotz = LotzCrossSections(self._element_properties)
        self._kll = KLLCrossSections(self._element_properties)

    @property
    def ElementProperties(self):
        """Returns the ElementProperties Object of the species"""
        return self._element_properties

    @property
    def KLLCrossSections(self):
        """Returns the KLLCrossSections Object of the species"""
        return self._kll

    @property
    def LotzCrossSections(self):
        """Returns the LotzCrossSections Object of the species"""
        return self._lotz

class SimpleEBISProblem:
    """
    class defining an EBIS charge breeding simulation and providing the interface to solve it
    """

    def __init__(self, species, j, e_kin, fwhm):
        """
        Defines the general problem constants (current density, electron energy and spread)

        Input parameters
        species - the Species object containing the physical information about cross section etc
        j - current density
        e_kin - electron energy
        fwhm - electron energy spread (for DR width)
        """
        self._species = species
        self._j = j
        self._e_kin = e_kin
        self._fwhm = fwhm
        self._solution = None
        #Default initial condition for solving the EBIS ODE System (all atoms in 1+ charge state)
        self._default_initial = np.zeros(self._species.ElementProperties.atomic_number + 1)
        self._default_initial[1] = 1

    @property
    def solution(self):
        """Returns the solution of the lates solve of this problem"""
        return self._solution

    def _ode_system(self, _, y): # No time dependence in this problem
        """
        The ode system describing the charge breeding
        """
        j = self._j / Q_E
        dydt = self._species.LotzCrossSections.cross_section_matrix(self._e_kin).dot(y)
        dydt += self._species.KLLCrossSections.cross_section_matrix(self._e_kin, self._fwhm).dot(y)
        dydt *= j
        return dydt

    def return_ode_system(self):
        """
        Releases a handle to the internal ODE system method
        This can be used for use in an external solver
        """
        return self._ode_system

    def solve(self, max_time, y0=None, **kwargs):
        """
        solves the charge state evolution up to the given time
        An intital distribution can be provided, if None is given, all ions will start as 1+

        Returns the ODE solution object and saves it in solution property for later access

        Input Parameters
        max_time - Integration maximum time
        y0 - initial distribution (numpy vector where the python index = charge state)
        **kwargs - are forwarded to the solver (scipy.integrate.solve_ivp)
        """
        if y0 is None:
            y0 = self._default_initial
        solution = scipy.integrate.solve_ivp(self._ode_system, [0, max_time], y0, **kwargs)
        solution.y = solution.y / np.sum(solution.y, axis=0)
        self._solution = solution
        return solution

    def plot_charge_state_evolution(self):
        """
        After the problem has been solved, the charge state evolution can be plotted by calling
        this function

        Returns figure handle and does not call show()
        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        tmax = self.solution.t.max()
        xlim = (1e-4, tmax)
        title = ("Charge State Evolution of " + self._species.ElementProperties.name
                 + " (Z = " + str(self._species.ElementProperties.atomic_number) + "), $E_{kin} = "
                 + str(self._e_kin) + "$ eV, FWHM = " + str(self._fwhm) + " eV")
        return plot_charge_state_evolution(self.solution, xlim=xlim, title=title)

class ContinuousNeutralInjectionEBISProblem(SimpleEBISProblem):
    """
    The class has been modified to increase the abundance in the neutral charge state
    at a constant rate to mimick continous neutral injection
    """

    def __init__(self, species, j, e_kin, fwhm):
        """
        Initialiser for a continuous injection problem, calls init of SimpleEBISProblem and adjusts
        the default initial condition, i.e. emtpy trap.
        """
        super().__init__(species, j, e_kin, fwhm)
        #Default initial condition for solving the EBIS ODE System (all atoms in neutral state)
        self._default_initial = np.zeros(self._species.ElementProperties.atomic_number + 1)
        self._default_initial[0] = 1e-9

    def _ode_system(self, _, y): # No time dependence in this problem
        """
        The ode system describing the charge breeding
        """
        j = self._j / Q_E
        dydt = self._species.LotzCrossSections.cross_section_matrix(self._e_kin).dot(y)
        dydt += self._species.KLLCrossSections.cross_section_matrix(self._e_kin, self._fwhm).dot(y)
        dydt *= j
        dydt[0] += 1
        return dydt

class EnergyScan:
    """
    Class that provides a simple interface to perform a scan of the electron beam energy for
    a SimpleEBISProblem (and inherited)
    """
    def __init__(self, problemtype, species, j, fwhm, energies, eval_times):
        """
        Initialises the energy scan settings

        Input Parameters
        problemtype - class handle specifiying the problem
        species - the Species object containing the physical information about cross section etc
        j - current density
        fwhm - electron energy spread (for DR width)
        energies - list of the energies to scan
        eval_times - list of the times to evaluate
        """
        self._problemtype = problemtype
        self._species = species
        self._j = j
        self._fwhm = fwhm
        self._energies = np.array(energies)
        self._eval_times = np.array(eval_times)
        self._solution = None

    @property
    def solution(self):
        """
        Returns the results of the energy scan as a single pandas frame
        """
        return self._solution

    def solve(self):
        """
        Trigger the computation of the energy scan that has previously been set up
        This can be fairly time consuming!
        """
        # Integration time
        max_time = np.max(self._eval_times)
        # Solve ODES
        scan_solutions = pd.DataFrame()
        for e_kin in self._energies:
            problem = self._problemtype(self._species, self._j, e_kin, self._fwhm)
            solution = problem.solve(max_time, t_eval=self._eval_times)
            sol_df = pd.DataFrame(solution.y.T)
            sol_df["t"] = solution.t
            sol_df["e_kin"] = e_kin
            scan_solutions = scan_solutions.append(sol_df, ignore_index=True)
        self._solution = scan_solutions
        return scan_solutions

    ###### The parallel solving is not working yet
    # def solve_parallel(self, processes=2):
    #     """
    #     Trigger the computation of the energy scan that has previously been set up
    #     This uses parallel computing
    #     This can be fairly time consuming!

    #     Input Parameters
    #     workers - number of processes to launch, should be about the number of cores of the PC
    #     """
    #     # Integration time
    #     max_time = np.max(self._eval_times)

    #     def workertask(e_kin):
    #         problem = self._problemtype(self._species, self._j, e_kin, self._fwhm)
    #         solution = problem.solve(max_time, t_eval=self._eval_times)
    #         sol_df = pd.DataFrame(solution.y.T)
    #         sol_df["t"] = solution.t
    #         sol_df["e_kin"] = e_kin
    #         return sol_df

    #     # Solve ODES in parallel
    #     pool = mp.Pool(processes=processes)
    #     sol_dfs = pool.map(workertask, self._energies)

    #     scan_solutions = pd.DataFrame()
    #     for sol_df in sol_dfs:
    #         scan_solutions = scan_solutions.append(sol_df, ignore_index=True)
    #     self._solution = scan_solutions

    def plot_abundance_at_time(self, t, cs, normalise=False, invert_hor=False,
                               x2fun=None, x2label=""):
        """
        Plots the charge state abundance at a time close to t (depends on eval_times)
        Returns the figure handle

        Input Parameters
        t - time for which to plot
        cs - list of charge states (integers) to plot
        normalise - normalise each curve with its own mean
        invert_hor - should the horizontal axis be inverted
        x2fun - (optional) function to compute values on 2nd x axis
        x2label - (optional) label for second x axis
        """
        t = self._eval_times[np.argmin(np.abs(self._eval_times - t))] # find closest t
        data = self._solution.loc[self._solution["t"] == t]

        fig = plt.figure(figsize=(6, 3), dpi=150)
        ax1 = fig.add_subplot(111)

        for c in cs:
            if normalise:
                plt.plot(data["e_kin"], data[c]/data[c].mean(), label=str(c) + "+")
            else:
                plt.plot(data["e_kin"], data[c], label=str(c) + "+")

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(which="both")
        plt.xlim(np.min(self._energies), np.max(self._energies))
        plt.xlabel("Electron Energy (eV)")
        if normalise:
            plt.ylabel("Normalised Abundance")
            title = ("Normalised Abundance of %s at $T=%.3G$ s"
                     %(self._species.ElementProperties.name, t))
        else:
            plt.ylabel("Relative Abundance")
            title = ("Relative Abundance of %s at $T=%.3G$ s"
                     %(self._species.ElementProperties.name, t))
            plt.ylim(0.01, 1)

        if invert_hor:
            plt.gca().invert_xaxis()

        if x2fun:
            ax2 = ax1.twiny()
            def tick_function(x):
                """tick function for the second axis"""
                V = x2fun(x)
                return ["%.0f" % z for z in V]
            new_tick_locations = ax1.get_xticks()
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_xticks(new_tick_locations)
            ax2.set_xticklabels(tick_function(new_tick_locations))
            ax2.set_xlabel(x2label)
            title += "\n\n"
        plt.title(title)
        plt.tight_layout()

        return fig

# print(element_name(1))
# print(element_name(8))
# print(element_name("He"))
# print(element_z("He"))
# print(element_z("Helium"))
# print(element_symbol("Helium"))
# print(element_symbol(2))
# PotassiumLCS = LotzCrossSections(19)
# PotassiumLCS.show_plot()
# PotassiumKLLCS = KLLCrossSections("K")
# PotassiumKLLCS.show_plot(20)
# plt.figure()
# plt.matshow(1e20*(PotassiumKLLCS.cross_section_matrix(2500, 10)))
# plt.grid()
# plt.figure()
# plt.matshow(1e20*(PotassiumKLLCS.cross_section_matrix(2550, 10)))
# plt.grid()
# plt.show()
# plt.close()
# input()
# IronLCS = LotzCrossSections(26)
# IronLCS.show_plot()
# print(PotassiumLCS.cross_section_matrix(2500))
# Potassium = EBISSpecies(19)
# prb = SimpleEBISProblem(Potassium, 1, 2500, 10)
# prb.solve_problem(100)
# prb.plot_charge_state_evolution()
# plt.show()
