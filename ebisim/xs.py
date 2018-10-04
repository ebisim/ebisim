"""
Module containing the classes providing access to the relevant cross section for ionisation and
recombination
"""

from functools import lru_cache
import numpy as np
import pandas as pd

from . import utils
from . import elements
from .physconst import RY_EV, ALPHA, PI, COMPT_E_RED

XS_CACHE_MAXSIZE = 10000 # The maxsize for the caching of xs matrices

# The normpdf function is used by the DR cross section method
def normpdf(x, mu, sigma):
    """
    The pdf of the normal distribution
    """
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / np.sqrt(2 * PI * sigma**2)

class XSBase:
    """
    Base class for cross section classes
    """

    XSTYPE = None
    # Needs to be set to "RECOMB" or "IONISE" by derived class to assemble xs_matrices correctly

    def __init__(self, element):
        """
        Initialise cross section object

        Input Parameters
        element - Atomic Number, Name, or Symbol, or ChemicalElement object
        """
        # Get basic properties of the element in question
        self._element = elements.cast_to_ChemicalElement(element)

        # Activate caching for Cross Section Matrices
        self.xs_matrix = lru_cache(maxsize=XS_CACHE_MAXSIZE)(self.xs_matrix)

        # Load required data from resource files, can set further fields
        self._load_data()

    @property
    def element(self):
        """Returns the ChemicalElement Object of the xs object"""
        return self._element

    def _load_data(self):
        """
        Load the required data from the resource directory
        --> Needs to be implemented by derived class
        """
        pass

    def xs(self, cs, e_kin):
        """
        Computes the cross section of a given charge state at a given electron energy
        UNIT: cm^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron

        --> Needs to be implemented by derived class
        --> Needs to accept any kinetic energy and 0 <= cs <= Z
        """
        return 1000*e_kin+cs # dummy return for easy debugging and testing

    def xs_matrix(self, e_kin):
        # pylint: disable=E0202
        """
        Returns a matrix with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are cached for better performance
        Be careful as this can occupy memory when a lot of energies are polled over time
        Cachesize is adjustable via XS_CACHE_MAXSIZE variable

        UNIT: cm^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        n = self._element.z + 1

        # Compute all cross sections
        xs = np.array([self.xs(cs, e_kin) for cs in range(n)])

        # Assemble matrix
        xs_mat = np.diag(-1 * xs) # negative cross section on diagonal (losses)
        # gains depend on whether looking at ionisation or recombination
        if self.__class__.XSTYPE == "IONISE":
            xs_mat += np.diag(xs[:-1], -1)
        elif self.__class__.XSTYPE == "RECOMB":
            xs_mat += np.diag(xs[1:], 1)

        return xs_mat


class EIXS(XSBase):
    """
    A class derived of XSBase that deals with Impact Ionisation Cross Sections computed from the
    Lotz formula

    UNIT: cm^2
    """
    XSTYPE = "IONISE"
    def _load_data(self):
        """
        Private Helper Method for loading the binding energies and electron configuration
        """
        # Import binding energies for each electron in all charge states
        # list of lists where each sublist hold the energies for one charge state
        # self._e_bind[n] describes charge state n+
        self._e_bind = []
        with utils.open_resource("BindingEnergies/%d.txt" % self._element.z) as fobj:
            for line in fobj:
                line = line.split()
                line = [float(elem.strip()) for elem in line]
                self._e_bind.append(line)

        # Import Electron Configurations for each charge state
        # list of lists where each sublist hold the configuration for on charge state
        # self._cfg[n] describes charge state n+
        self._cfg = []
        with utils.open_resource("BindingEnergies/%dconf.txt" % self._element.z) as fobj:
            for line in fobj:
                line = line.split()
                line = [int(elem.strip()) for elem in line]
                self._cfg.append(line)

        self._e_bind_min = self._e_bind[0][-1]
        self._e_bind_max = self._e_bind[-1][0]

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
        UNIT: cm^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        if cs == self._element.z:
            return 0
        xs = 0
        for e_bind, num_el in zip(self._e_bind[cs], self._cfg[cs]):
            if e_kin > e_bind and num_el > 0:
                xs += num_el * np.log(e_kin / e_bind) / (e_kin * e_bind)
        xs *= 4.5e-14
        return xs


class RRXS(EIXS):
    """
    A class derived of IIXS that provides convenient handling of the Radiative recombination
    cross sections

    UNIT: cm^2
    """
    XSTYPE = "RECOMB"
    def xs(self, cs, e_kin):
        """
        Computes the RR cross section of a given charge state at a given electron energy
        UNIT: cm^2
        According to Kim and Pratt Phys Rev A (27,6) p.27/2913 (1983)

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        if cs == 0:
            return 0 # Atom cannot recombine

        if cs < self._element.z:
            cfg = self._cfg[cs]
            ### Determine number of electrons in ion valence shell (highest occupied)
            # The sorting of orbitals in Roberts files is a bit obscure but seems to be consistent
            # and correct (checked a few configurations to verify that)
            # According to the readme files the columns are:
            # 1s 2s 2p- 2p+ 3s 3p- 3p+ 3d- 3d+ 4s 4p- 4p+ 4d- 4d+ ...
            # 5s 5p- 5p+ 4f- 4f+ 5d- 5d+ 6s 6p- 6p+ 5f- 5f+ 6d- 6d+ 7s
            SHELL_KEY = [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
                         5, 5, 5, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6, 7] # n of each orbital in order
            n_0 = max(SHELL_KEY[:len(cfg)])
            occup = sum(cfg[k] for k in range(len(cfg)) if SHELL_KEY[k] == n_0)
            # print("n_0:", n_0, "occup", occup)
        elif cs == self._element.z:
            n_0 = 1
            occup = 0

        w_n0 = (2*n_0**2 - occup)/(2*n_0**2)
        n_0eff = n_0 + (1 - w_n0) - 0.3
        # print("cs:", cs, "w_n0:", w_n0, "n_0eff", n_0eff)

        ### the rest
        z_eff = (self._element.z + cs) / 2
        chi = 2 * z_eff**2 * RY_EV / e_kin

        xs = 8 * PI * ALPHA / (3 * np.sqrt(3)) * COMPT_E_RED**2 * \
                     chi * np.log(1 + chi/(2 * n_0eff**2))

        return xs*1e4 #convert to cm^2


class DRXS(XSBase):
    """
    A class derived of XSBase that deals with Di(Multi)electronic Recombination Cross Sections
    Assuming a Gaussian profile widening due to the electron beam enegry spread

    UNIT: cm^2
    """
    XSTYPE = "RECOMB"
    def __init__(self, element, fwhm):
        """
        Initialise cross section object

        Input Parameters
        element - Atomic Number, Name, or Symbol, or ChemicalElement object
        fwhm - Width of the Gaussian energy spread profile used to compute the spectrum
        """
        self._fwhm = fwhm
        super().__init__(element)

    @property
    def fwhm(self):
        """"
        Returns the current value of fwhm set for this cross section object
        Setting a new fwhm clears the xs_matrix cache
        """
        return self._fwhm

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
    ##### The fwhm is currently immutable because I need to find a way to deal with the case
    ##### that the object has several users of which one may change the fwhm, which would
    ##### break the validity for all other users
    # @fwhm.setter
    # def fwhm(self, val):
    #     """fwhm setter (clears cache on set)"""
    #     if self._fwhm != val:
    #         self._xsmat_cache = {}
    #         self._fwhm = val

    def _load_data(self):
        """
        Private Helper Method for loading the binding energies and electron configuration
        """
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
        for cs in dr_by_cs.groups:
            drdf = dr_by_cs.get_group(cs)
            self._resonance_energies[cs] = drdf.DELTA_E_AI.values.copy()
            self._recomb_strengths[cs] = drdf.RECOMB_STRENGTH.values.copy()

        self._e_res_min = dr_by_cs.min().DELTA_E_AI.min()
        self._e_res_max = dr_by_cs.max().DELTA_E_AI.max()


    def xs(self, cs, e_kin):
        """
        Computes the DR (MR) cross section of a given charge state at a given electron energy
        assuming that the resonances are delta peaks that are smeared out due to the
        energy spread of the electron beam of the EBIS
        UNIT: cm^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        if cs not in self._resonance_energies: #Check if key cs exists
            return 0 # If no DR Data available for this CS return 0

        sig = self._fwhm/2.35482 # 2.35482approx.(2*np.sqrt(2*np.log(2)))
        xs = np.sum(self._recomb_strengths[cs] * normpdf(e_kin, self._resonance_energies[cs], sig))

        return xs*1e-20 # normalise to cm**2


class EBISSpecies:
    """
    collection of properties relevant to an atomic species in an EBIS for solving rate equations
    """
    def __init__(self, element, fwhm):
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
        self._drxs = DRXS(self._element, fwhm)

    def __repr__(self):
        return "EBISSpecies('%s', %s)"%(self.element.symbol, str(self.fwhm))

    def __str__(self):
        return "EBISSpecies - Element: %s (%s, Z = %d), FWHM = %.2f eV"%(
            self.element.name, self.element.symbol, self.element.z, self.fwhm)

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

    @property
    def fwhm(self):
        """"
        Returns the current value of fwhm set for this cross section object
        Setting a new fwhm clears the xs_matrix cache
        """
        return self._drxs.fwhm

    ##### The fwhm is currently immutable because I need to find a way to deal with the case
    ##### that the object has several users of which one may change the fwhm, which would
    ##### break the validity for all other users
    # @fwhm.setter
    # def fwhm(self, val):
    #     """fwhm setter (clears cache on set)"""
    #     self._drxs.fwhm = val
