"""
Module containing the classes providing access to the relevant cross section for ionisation and
recombination
"""

from functools import lru_cache
import math
import numpy as np
import pandas as pd
import numba

from . import utils
from . import elements
from .physconst import RY_EV, ALPHA, PI, COMPT_E_RED

XS_CACHE_MAXSIZE = 10000 # The maxsize for the caching of xs matrices

### The following are low level functions going into the cross section computation
### They are jit compiled by numba to increase their performance
### Unfortunately the readability of the code suffers a bit this way, but ... ce la vie
@numba.jit
def _normpdf(x, mu, sigma):
    """
    The pdf of the normal distribution
    """
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (2 * PI * sigma**2)**0.5

@numba.jit
def _eixs_xs_vector(e_kin, e_bind_mat, cfg_mat):
    n = e_bind_mat.shape[1] #=Z+1
    m = e_bind_mat.shape[0] # number of rows
    xs_vec = np.zeros(n)
    for cs in range(n-1):
        xs = 0
        for row in range(m):
            e_bind = e_bind_mat[row, cs]
            num_el = cfg_mat[row, cs]
            if e_kin > e_bind and num_el > 0:
                xs += num_el * math.log(e_kin / e_bind) / (e_kin * e_bind)
        xs_vec[cs] = xs
    xs_vec *= 4.5e-18
    return xs_vec

@numba.njit
def _eixs_vector_2(element, e_kin):
    css = element.ebind.shape[0]
    shells = element.ebind.shape[1]
    xs_vec = np.zeros(css + 1)
    for cs in range(css):
        xs = 0
        for shell in range(shells):
            e = element.ebind[cs, shell]
            n = element.cfg[cs, shell]
            if e_kin > e and n > 0:
                xs += n * math.log(e_kin / e) / (e_kin * e)
        xs_vec[cs] = xs
    xs_vec *= 4.5e-18
    return xs_vec

@numba.njit
def _rrxs_vector_2(element, e_kin):
    chi = 2 * element.z_eff**2 * RY_EV / e_kin
    xs = 8 * PI * ALPHA / (3 * np.sqrt(3)) * COMPT_E_RED**2 * \
            chi * np.log(1 + chi/(2 * element.n_0_eff**2))
    xs[0] = 0
    return xs

@numba.njit
def _drxs_vector_2(element, e_kin, fwhm):
    xs_vec = np.zeros(element.z + 1)
    if element.dr_cs.size > 0:
        sig = fwhm/2.35482 # 2.35482approx.(2*np.sqrt(2*np.log(2)))
        tmp = element.dr_strength * _normpdf(e_kin, element.dr_e_res, sig)*1e-24
        for k in range(element.dr_cs.size):
            cs = int(element.dr_cs[k])
            xs_vec[cs] = xs_vec[cs] + tmp[k]
    return xs_vec

@numba.jit
def _drxs_xs(e_kin, fwhm, recomb_strengths, resonance_energies):
    """
    This functions computes dr cross sections as a weighted sum auf normal pdfs
    """
    sig = fwhm/2.35482 # 2.35482approx.(2*np.sqrt(2*np.log(2)))
    return np.sum(recomb_strengths * _normpdf(e_kin, resonance_energies, sig))*1e-24

@numba.njit
def precompute_rr_quantities(cfg, shell_n):
    """
    Precomputes the effective valence shell and nuclear charge for all charge states,
    needed for radiative recombinations cross sections
    """
    z = cfg.shape[0]
    shell_n = shell_n[:cfg.shape[1]] # Crop shell_n to the shells described in cfg

    n_0 = np.zeros(z + 1)
    occup = np.zeros(z + 1)

    # Determine, for each charge state, the valence shell (n_0),
    # and the number of electrons in it (occup)
    # Fully ionised
    n_0[z] = 1
    occup[z] = 0
    # All other charge states
    for cs in range(z):
        conf = cfg[cs, :]
        n_0[cs] = np.max(shell_n[np.nonzero(conf)])
        occup[cs] = np.sum(np.extract((shell_n == n_0[cs]), conf))

    w_n0 = (2 * n_0**2 - occup) / (2 * n_0**2)
    n_0_eff = n_0 + (1 - w_n0) - 0.3
    z_eff = (z + np.arange(z + 1)) / 2

    return z_eff, n_0_eff

### Here start the class definitions for the different cross section classes
class EIXS:
    """
    A class that deals with Electron Ionisation Cross Sections computed from the Lotz formula

    UNIT: m^2
    """
    def __init__(self, element):
        """
        Initialise cross section object

        Input Parameters
        element - Atomic Number, Name, or Symbol, or ChemicalElement object
        """
        # Get basic properties of the element in question
        self._element = elements.cast_to_ChemicalElement(element)
        # Activate caching for Cross Section Vectors
        self.xs_vector = lru_cache(maxsize=XS_CACHE_MAXSIZE)(self.xs_vector)

        # Load required data from resource files, can set further fields
        # Import binding energies for each electron in all charge states
        # list of lists where each sublist hold the energies for one charge state
        # self._e_bind[n] describes charge state n+
        self._e_bind = []
        with utils.open_resource("BindingEnergies/%d.txt" % self._element.z) as fobj:
            for line in fobj:
                line = line.split()
                line = np.array([float(elem.strip()) for elem in line])
                self._e_bind.append(line)

        # Import Electron Configurations for each charge state
        # list of lists where each sublist hold the configuration for on charge state
        # self._cfg[n] describes charge state n+
        self._cfg = []
        n_cfg_max = 0
        with utils.open_resource("BindingEnergies/%dconf.txt" % self._element.z) as fobj:
            for line in fobj:
                line = line.split()
                line = np.array([int(elem.strip()) for elem in line])
                n_cfg_max = max(n_cfg_max, len(line))
                self._cfg.append(line)

        # This block could be useful in the future for parallelising cross section computations
        self._e_bind_mat = np.zeros((n_cfg_max, self.element.z+1))
        self._cfg_mat = np.zeros((n_cfg_max, self.element.z+1))
        for cs, data in enumerate(self._e_bind):
            self._e_bind_mat[:len(data), cs] = data
        for cs, data in enumerate(self._cfg):
            self._cfg_mat[:len(data), cs] = data

        self._e_bind_min = self._e_bind[0][-1]
        self._e_bind_max = self._e_bind[-1][0]

    @property
    def element(self):
        """Returns the ChemicalElement Object of the xs object"""
        return self._element

    @property
    def e_bind_min(self):
        """
        Property returning the smallest binding energy within all charge states
        """
        return self._e_bind_min

    @property
    def e_bind_max(self):
        """
        Property returning the highest binding energy within all charge states
        """
        return self._e_bind_max

    def xs(self, cs, e_kin):
        """
        Computes the Lotz cross section of a given charge state at a given electron energy
        UNIT: m^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        return self.xs_vector(e_kin)[cs] #fast enough to do it this way

    def xs_vector(self, e_kin):
        # pylint: disable=E0202
        """
        Returns a vector with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        The vector index of each entry corresponds to the charge state

        Vectors are cached for better performance
        Be careful as this can occupy memory when a lot of energies are polled over time
        Cachesize is adjustable via XS_CACHE_MAXSIZE variable

        UNIT: m^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        return _eixs_xs_vector(e_kin, self._e_bind_mat, self._cfg_mat)

    def xs_matrix(self, e_kin):
        """
        Returns a matrix with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are assembled from (cached) vectors for better performance

        UNIT: m^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        xs = self.xs_vector(e_kin)
        return np.diag(xs[:-1], -1) - np.diag(xs)


class RRXS(EIXS):
    """
    A class derived of IIXS that provides convenient handling of the Radiative recombination
    cross sections

    UNIT: m^2
    """
    def __init__(self, element):
        """
        Initialise cross section object

        Input Parameters
        element - Atomic Number, Name, or Symbol, or ChemicalElement object
        """
        super().__init__(element)
        #Precompute some quantities going into the xs that only depend on cs for efficiency
        n_0 = []
        occup = []
        for cs in range(self._element.z + 1):
            if cs < self._element.z:
                cfg = self._cfg[cs]
                ### Determine number of electrons in ion valence shell (highest occupied)
                # The sorting of orbitals in Roberts files is a bit obscure but seems consistent
                # and correct (checked a few configurations to verify that)
                # According to the readme files the columns are:
                # 1s 2s 2p- 2p+ 3s 3p- 3p+ 3d- 3d+ 4s 4p- 4p+ 4d- 4d+ ...
                # 5s 5p- 5p+ 4f- 4f+ 5d- 5d+ 6s 6p- 6p+ 5f- 5f+ 6d- 6d+ 7s
                SHELL_KEY = [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                             5, 5, 5, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6, 7, 7] #n of each orbit
                temp_shell = SHELL_KEY[:len(cfg)]
                n_0_temp = max(temp_shell[k] if cfg[k] != 0 else 0 for k in range(len(cfg)))
                n_0.append(n_0_temp)
                occup.append(sum(cfg[k] for k in range(len(cfg)) if SHELL_KEY[k] == n_0_temp))
                # print("n_0:", n_0, "occup", occup)
            elif cs == self._element.z:
                n_0.append(1)
                occup.append(0)
        self._n_0 = np.array(n_0)
        self._occup = np.array(occup)
        self._w_n0 = (2*self._n_0**2 - self._occup)/(2*self._n_0**2)
        self._n_0eff = self._n_0 + (1 - self._w_n0) - 0.3
        self._z_eff = (self._element.z + np.arange(self._element.z+1)) / 2
        self._chi_prep = 2 * self._z_eff**2 * RY_EV
        del(self._n_0, self._occup, self._w_n0, self._z_eff, self._e_bind) # Not needed atm

    def xs(self, cs, e_kin):
        """
        Computes the RR cross section of a given charge state at a given electron energy
        UNIT: m^2
        According to Kim and Pratt Phys Rev A (27,6) p.27/2913 (1983)

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        return self.xs_vector(e_kin)[cs] #Vectorized solution is fast enough to do it this way

    def xs_vector(self, e_kin):
        # pylint: disable=E0202
        """
        Returns a vector with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        The vector index of each entry corresponds to the charge state

        Vectors are cached for better performance
        Be careful as this can occupy memory when a lot of energies are polled over time
        Cachesize is adjustable via XS_CACHE_MAXSIZE variable

        UNIT: m^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        chi = self._chi_prep / e_kin
        xs = 8 * PI * ALPHA / (3 * np.sqrt(3)) * COMPT_E_RED**2 * \
                chi * np.log(1 + chi/(2 * self._n_0eff**2))
        xs[0] = 0
        return xs

    def xs_matrix(self, e_kin):
        """
        Returns a matrix with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are assembled from (cached) vectors for better performance

        UNIT: m^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        xs = self.xs_vector(e_kin)
        return np.diag(xs[1:], 1) - np.diag(xs)


class DRXS:
    """
    A class derived of XSBase that deals with Di(Multi)electronic Recombination Cross Sections
    Assuming a Gaussian profile widening due to the electron beam enegry spread

    UNIT: m^2
    """
    def __init__(self, element):
        """
        Initialise cross section object

        Input Parameters
        element - Atomic Number, Name, or Symbol, or ChemicalElement object
        fwhm - Width of the Gaussian energy spread profile used to compute the spectrum
        """
        # Get basic properties of the element in question
        self._element = elements.cast_to_ChemicalElement(element)
        # Activate caching for Cross Section Vectors
        self.xs_vector = lru_cache(maxsize=XS_CACHE_MAXSIZE)(self.xs_vector)

        # Import DR Transitions for this Element
        # Columns: DELTA_E_AI (eV), RECOMB_STRENGTH (1e-20 cm**2 eV),
        # RECOMB_TYPE(DR, TR, QR,...), RECOMB_NAME(KLL, KLM, ...), CHARGE_STATE
        try:
            with utils.open_resource("DR/%s_KLL.csv" % self._element.symbol) as fobj:
                dr_by_cs = pd.read_csv(fobj, index_col=None)
        except FileNotFoundError as e:
            print("NO DR FILE FOUND")
            print(e)
            print("--> FALLING BACK TO DUMMY")
            with utils.open_resource("DR/_FALLBACKDUMMY.csv") as fobj:
                dr_by_cs = pd.read_csv(fobj, index_col=None)
        dr_by_cs = dr_by_cs.groupby("CHARGE_STATE")

        self._resonance_energies = {}
        self._recomb_strengths = {}
        n_trans_max = 0
        for cs in dr_by_cs.groups:
            drdf = dr_by_cs.get_group(cs)
            n_trans_max = max(len(drdf.index), n_trans_max)
            self._resonance_energies[cs] = drdf.DELTA_E_AI.values.copy()
            self._recomb_strengths[cs] = drdf.RECOMB_STRENGTH.values.copy()

        # This block could be useful in the future for parallelising cross section computations
        # self._resonance_energies_mat = np.zeros((n_trans_max, self.element.z+1))
        # self._recomb_strengths_mat = np.zeros((n_trans_max, self.element.z+1))
        # for cs, data in self._resonance_energies.items():
        #     self._resonance_energies_mat[:len(data), cs] = data
        # for cs, data in self._recomb_strengths.items():
        #     self._recomb_strengths_mat[:len(data), cs] = data

        self._avail_cs = sorted(self._resonance_energies.keys())
        self._e_res_min = dr_by_cs.min().DELTA_E_AI.min()
        self._e_res_max = dr_by_cs.max().DELTA_E_AI.max()

    @property
    def element(self):
        """Returns the ChemicalElement Object of the xs object"""
        return self._element

    @property
    def e_res_min(self):
        """
        Property returning the smallest DRR energy within all charge states
        """
        return self._e_res_min

    @property
    def e_res_max(self):
        """
        Property returning the highest DRR energy within all charge states
        """
        return self._e_res_max

    def xs(self, cs, e_kin, fwhm):
        """
        Computes the DR (MR) cross section of a given charge state at a given electron energy
        assuming that the resonances are delta peaks that are smeared out due to the
        energy spread of the electron beam of the EBIS
        UNIT: m^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        return self.xs_vector(e_kin, fwhm)[cs] #fast enough to do it this way

    def xs_vector(self, e_kin, fwhm):
        # pylint: disable=E0202
        """
        Returns a vector with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        The vector index of each entry corresponds to the charge state

        Vectors are cached for better performance
        Be careful as this can occupy memory when a lot of energies are polled over time
        Cachesize is adjustable via XS_CACHE_MAXSIZE variable

        UNIT: m^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        xs = np.zeros(self._element.z + 1)
        for cs in self._avail_cs:
            xs[cs] = _drxs_xs(e_kin, fwhm, self._recomb_strengths[cs], self._resonance_energies[cs])
        return xs

    def xs_matrix(self, e_kin, fwhm):
        """
        Returns a matrix with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are assembled from (cached) vectors for better performance

        UNIT: m^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        xs = self.xs_vector(e_kin, fwhm)
        return np.diag(xs[1:], 1) - np.diag(xs)


class EBISSpecies:
    """
    collection of properties relevant to an atomic species in an EBIS for solving rate equations
    """
    def __init__(self, element):
        """
        Creates the species by defining the element and automatically creating objects for the
        Lotz and KLL cross section

        Input Parameters
        element - Atomic Number, Symbol or Name
        fwhm - fwhm of the Gaussian used for spreading the DR cross sections
        """
        # Get basic properties of the element in question
        self._element = elements.cast_to_ChemicalElement(element)
        self._eixs = EIXS(self._element)
        self._rrxs = RRXS(self._element)
        self._drxs = DRXS(self._element)

    def __repr__(self):
        return "EBISSpecies('%s')"%(self.element.symbol)

    def __str__(self):
        return "EBISSpecies - Element: %s (%s, Z = %d)"%(
            self.element.name, self.element.symbol, self.element.z)

    @property
    def element(self):
        """Returns the ChemicalElement Object of the species"""
        return self._element

    @property
    def eixs(self):
        """Returns the IIXS Object of the species"""
        return self._eixs

    @property
    def rrxs(self):
        """Returns the RRXS Object of the species"""
        return self._rrxs

    @property
    def drxs(self):
        """Returns the DRXS Object of the species"""
        return self._drxs
