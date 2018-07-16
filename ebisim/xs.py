"""
Module containing the classes providing access to the relevant cross section for ionisation and
recombination
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from . import utils
from . import elements
from . import plotting
from .physconst import RY_EV, ALPHA, PI, COMPT_E_RED

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
        if not isinstance(element, elements.ChemicalElement):
            element = elements.ChemicalElement(element)
        self._z = element.z
        self._es = element.symbol
        self._name = element.name

        # Instantiate cache for Cross Section Matrices
        self._xsmat_cache = {}

        # Load reuired data from resource files
        self._load_data()

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

    def _compute_xs_mat(self, e_kin):
        """
        computes the cross section matrix for a given kinetic energy if not found in cache
        """
        n = self._z + 1

        # Compute all cross sections
        xs = np.zeros(n)
        for cs in range(n):
            xs[cs] = self.xs(cs, e_kin)

        # Assemble matrix
        xs_mat = np.diag(-1 * xs) # negative cross section on diagonal (losses)
        # gains depend on whether looking at ionisation or recombination
        if self.__class__.XSTYPE == "IONISE":
            for cs in range(0, n-1):
                xs_mat[cs+1, cs] = xs[cs]
        elif self.__class__.XSTYPE == "RECOMB":
            for cs in range(1, n):
                xs_mat[cs-1, cs] = xs[cs]

        return xs_mat

    def xs_matrix(self, e_kin):
        """
        Returns a matrix with the cross sections for a given electron energy
        that can be used to solve the rate equations in a convenient manner.

        Matrices are cached for better performance
        Be careful as this can occupy memory when a lot of energies are polled over time

        UNIT: cm^2

        Input Parameters
        e_kin - Electron kinetic energy
        """
        if e_kin not in self._xsmat_cache.keys():
            self._xsmat_cache[e_kin] = self._compute_xs_mat(e_kin)
        return self._xsmat_cache[e_kin]

    def _compute_xs_df_for_plot(self, energies):
        """
        Private helper function that generates the dataframe for plotting
        """
        rows = []
        for ek in energies:
            xs = -1 * np.diag(self.xs_matrix(ek))
            rows.append(np.hstack([ek, xs]))
        colnames = ["ekin"] + [str(cs) for cs in range(self._z+1)]
        xs_df = pd.DataFrame(rows, columns=colnames)
        return xs_df

    def create_plot(self, xscale="log", yscale="log", xlim=None, ylim=None,
                    legend=False, label_lines=True):
        """
        Creates a figure showing the cross sections and returns the figure handle
        # Needs to be implemented by inheriting class

        Input Parameters
        xscale, yscale - (optional) Scaling of x and y axis (log or linear)
        xlim, ylim - (optional) plot limits
        legend - (optional) show legend?
        line_labels - annotate lines?
        """
        # Generate Data with _compute_xs_df_for_plot
        # call plotting.plot_xs()
        # Return figure handle
        return None

    def show_plot(self, xscale="log", yscale="log", xlim=None, ylim=None,
                  legend=False, label_lines=True):
        """
        Creates a figure showing the Cross section and calls the show method

        Input Parameters
        xscale, yscale - (optional) Scaling of x and y axis (log or linear)
        xlim, ylim - (optional) plot limits
        legend - (optional) show legend?
        line_labels - annotate lines?
        """
        self.create_plot(xscale=xscale, yscale=yscale, xlim=xlim, ylim=ylim,
                         legend=legend, label_lines=label_lines)
        plt.show()



class IIXS(XSBase):
    """
    A class derived of XSBase that deals with Impact Ionisation Cross Sections computed from the
    Lotz formula
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
        with utils.open_resource("BindingEnergies/%d.txt" % self._z) as fobj:
            for line in fobj:
                line = line.split()
                line = [float(elem.strip()) for elem in line]
                self._e_bind.append(line)

        # Import Electron Configurations for each charge state
        # list of lists where each sublist hold the configuration for on charge state
        # self._cfg[n] describes charge state n+
        self._cfg = []
        with utils.open_resource("BindingEnergies/%dconf.txt" % self._z) as fobj:
            for line in fobj:
                line = line.split()
                line = [int(elem.strip()) for elem in line]
                self._cfg.append(line)

    def xs(self, cs, e_kin):
        """
        Computes the Lotz cross section of a given charge state at a given electron energy
        UNIT: cm^2

        Input Parameters
        cs - Charge State (0 for neutral atom)
        e_kin - kinetic energy of projectile Electron
        """
        if cs == self._z:
            return 0
        xs = 0
        for e_bind, num_el in zip(self._e_bind[cs], self._cfg[cs]):
            if e_kin > e_bind and num_el > 0:
                xs += num_el * np.log(e_kin / e_bind) / (e_kin * e_bind)
        xs *= 4.5e-14
        return xs

    def create_plot(self, xscale="log", yscale="log", xlim=None, ylim=None,
                    legend=False, label_lines=True):
        """
        Creates a figure showing the cross sections and returns the figure handle

        Input Parameters
        xscale, yscale - (optional) Scaling of x and y axis (log or linear)
        xlim, ylim - (optional) plot limits
        legend - (optional) show legend?
        line_labels - annotate lines?
        """
        # Generate Data with _compute_xs_df_for_plot
        e_min = self._e_bind[0][-1]/10
        e_max = 10*self._e_bind[-1][0]
        e_max = 10**np.ceil(np.log10(e_max))
        if xlim:
            e_min = np.min([e_min, xlim[0]])
            e_max = np.max([e_max, xlim[1]])
        else:
            xlim = (1, e_max)
        energies = np.logspace(np.log10(e_min), np.log10(e_max), 5000)
        xs_df = self._compute_xs_df_for_plot(energies)
        # call plotting.plot_xs()
        title = self._name + " (Z = " + str(self._z) + ") Lotz ionisation cross sections"
        fig = plotting.plot_xs(xs_df, xscale=xscale, yscale=yscale,
            title=title, xlim=xlim, ylim=ylim, legend=legend, label_lines=label_lines)
        # Return figure handle
        return fig



# class LotzCrossSections:
#     """
#     An object that provides convenient handling of the Lotz cross sections
#     UNIT: cm^2
#     """
#     def __init__(self, element):
#         """
#         Initiates the object by importing the relevant binding energies
#         from FAC simulation results (performed by Robert Mertzig)

#         Input Parameters
#         element - Atomic Number or Element Symbol or Element object
#         """
#         if not isinstance(element, ChemicalElement):
#             element = ChemicalElement(element)
#         self._z = element.atomic_number
#         self._es = element.symbol
#         self._name = element.name

#         self._cache = {} # Cache for Cross Section Matrices

#         self._load_binding_energies()
#         self._load_electron_configuration()

#     def _load_binding_energies(self):
#         """
#         Private Helper Method for loading the binding energies
#         """
#         # Import binding energies for each electron in all charge states
#         # list of lists where each sublist hold the energies for one charge state
#         # self._e_bind[n] describes charge state n+
#         self._e_bind = []
#         with utils.open_resource("BindingEnergies/%d.txt" % self._z) as fobj:
#             for line in fobj:
#                 line = line.split()
#                 line = [float(elem.strip()) for elem in line]
#                 self._e_bind.append(line)

#     def _load_electron_configuration(self):
#         """
#         Private Helper Method for loading the electron configuration
#         """
#         # Import Electron Configurations for each charge state
#         # list of lists where each sublist hold the configuration for on charge state
#         # self._cfg[n] describes charge state n+
#         self._cfg = []
#         with utils.open_resource("BindingEnergies/%dconf.txt" % self._z) as fobj:
#             for line in fobj:
#                 line = line.split()
#                 line = [int(elem.strip()) for elem in line]
#                 self._cfg.append(line)

#     def cross_section(self, cs, e_kin):
#         """
#         Computes the Lotz cross section of a given charge state at a given electron energy
#         UNIT: cm^2

#         Input Parameters
#         cs - Charge State (0 for neutral atom)
#         e_kin - kinetic energy of projectile Electron
#         """
#         cross_sect = 0
#         for e_bind, num_el in zip(self._e_bind[cs], self._cfg[cs]):
#             if e_kin > e_bind and num_el > 0:
#                 cross_sect += num_el * np.log(e_kin / e_bind) / (e_kin * e_bind)
#         cross_sect *= 4.5e-14
#         return cross_sect

#     def cross_section_matrix(self, e_kin):
#         """
#         Computes and returns a matrix with the lotz cross sections for a given electron energy
#         that can be used to solve the rate equations in a convenient manner.

#         Matrices are cached for better performance, be careful as this can occupy memory when a lot
#         of energies are polled over time

#         UNIT: cm^2

#         Input Parameters
#         e_kin - Electron kinetic energy
#         """
#         # Check cache
#         if e_kin in self._cache.keys():
#             return self._cache[e_kin]

#         # If nonexistent, compute:
#         cross_mat = np.zeros((self._z+1, self._z+1))
#         cross_sec = np.zeros(self._z)

#         for cs in range(self._z):
#             cross_sec[cs] = self.cross_section(cs, e_kin)

#         for cs in range(self._z+1):
#             if cs > 0:
#                 cross_mat[cs, cs-1] = cross_sec[cs-1]
#             if cs < self._z:
#                 cross_mat[cs, cs] = -cross_sec[cs]

#         self._cache[e_kin] = cross_mat
#         return cross_mat

#     def create_plot(self, xlim=None, ylim=None):
#         """
#         Creates a figure showing the Lotz Cross section and returns the figure handle

#         Input Parameters
#         xlim, ylim - plot limits (optional)
#         """
#         e_min = self._e_bind[0][-1]/10
#         e_max = 10*self._e_bind[-1][0]
#         e_max = 10**np.ceil(np.log10(e_max))
#         if xlim:
#             e_min = np.min([e_min, xlim[0]])
#             e_max = np.max([e_max, xlim[1]])
#         else:
#             xlim = (1, e_max)
#         ene = np.logspace(np.log10(e_min), np.log10(e_max), 5000)

#         fig = plt.figure(figsize=(8, 6), dpi=150)
#         for cs in range(self._z):
#             res = []
#             for e in ene:
#                 res.append(self.cross_section(cs, e))
#             plt.loglog(ene, np.array(res), lw=1, label=str(cs)+"+")
#         # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         plt.title(self._name + " (Z = " + str(self._z) + ") Lotz ionisation cross sections")
#         plt.xlabel("Electron kinetic energy (eV)")
#         plt.ylabel("Cross section (cm$^2$)")
#         plt.grid(which="both", alpha=0.5)
#         plt.xlim(xlim)
#         if ylim:
#             plt.ylim(ylim)
#         labelLines(plt.gca().get_lines(), size=7, bbox={"pad":0.1, "fc":"w", "ec":"none"})
#         return fig

#     def show_plot(self, xlim=None, ylim=None):
#         """
#         Creates a figure showing the Lotz Cross section and calls the show method

#         Input Parameters
#         xlim, ylim - plot limits (optional)
#         """
#         self.create_plot(xlim, ylim)
#         plt.show()

# class RRCrossSections(LotzCrossSections):
#     """
#     An object that provides convenient handling of the Radiative recombination cross sections
#     UNIT: cm^2
#     """
#     def cross_section(self, cs, e_kin):
#         """
#         Computes the RR cross section of a given charge state at a given electron energy
#         UNIT: cm^2
#         According to Kim and Pratt Phys Rev A (27,6) p.27/2913 (1983)

#         Input Parameters
#         cs - Charge State (0 for neutral atom)
#         e_kin - kinetic energy of projectile Electron
#         """
#         if cs == 0:
#             return 0 # Atom cannot recombine

#         if cs < self._z:
#             cfg = self._cfg[cs]
#             ### Determine number of electrons in ion valence shell (highest occupied)
#             # The sorting of orbitals in Roberts files is a bit obscure but seems to be consistent
#             # and correct (checked a few configurations to verify that)
#             # According to the readme files the columns are:
#             # 1s 2s 2p- 2p+ 3s 3p- 3p+ 3d- 3d+ 4s 4p- 4p+ 4d- 4d+ ...
#             # 5s 5p- 5p+ 4f- 4f+ 5d- 5d+ 6s 6p- 6p+ 5f- 5f+ 6d- 6d+ 7s
#             SHELL_KEY = [1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4,
#                          5, 5, 5, 4, 4, 5, 5, 6, 6, 6, 5, 5, 6, 6, 7] # n of each orbital in order
#             n_0 = max(SHELL_KEY[:len(cfg)])
#             occup = sum(cfg[k] for k in range(len(cfg)) if SHELL_KEY[k] == n_0)
#             # print("n_0:", n_0, "occup", occup)
#         elif cs == self._z:
#             n_0 = 1
#             occup = 0

#         w_n0 = (2*n_0**2 - occup)/(2*n_0**2)
#         n_0eff = n_0 + (1 - w_n0) - 0.3
#         # print("cs:", cs, "w_n0:", w_n0, "n_0eff", n_0eff)

#         ### the rest
#         z_eff = (self._z + cs) / 2
#         chi = 2 * z_eff**2 * RY_EV / e_kin

#         cross_sect = 8 * PI * ALPHA / (3 * np.sqrt(3)) * COMPT_E_RED**2 * \
#                      chi * np.log(1 + chi/(2 * n_0eff**2))

#         return cross_sect*1e4 #convert to cm^2

#     def cross_section_matrix(self, e_kin):
#         """
#         Computes and returns a matrix with the RR cross sections for a given electron energy
#         that can be used to solve the rate equations in a convenient manner.

#         Matrices are cached for better performance, be careful as this can occupy memory when a lot
#         of energies are polled over time

#         UNIT: cm^2

#         Input Parameters
#         e_kin - Electron kinetic energy
#         """
#         # Check cache
#         if e_kin in self._cache.keys():
#             return self._cache[e_kin]

#         # If nonexistent, compute:
#         cross_mat = np.zeros((self._z+1, self._z+1))
#         cross_sec = np.zeros(self._z+1)

#         for cs in range(self._z+1):
#             cross_sec[cs] = self.cross_section(cs, e_kin)

#         for cs in range(self._z+1):
#             cross_mat[cs, cs] = -cross_sec[cs]
#             if cs < self._z:
#                 cross_mat[cs, cs+1] = cross_sec[cs+1]

#         self._cache[e_kin] = cross_mat
#         return cross_mat

#     def create_plot(self, xlim=None, ylim=None):
#         """
#         Creates a figure showing the RR Cross section and returns the figure handle

#         Input Parameters
#         xlim, ylim - plot limits (optional)
#         """
#         e_min = self._e_bind[0][-1]/10
#         e_max = 10*self._e_bind[-1][0]
#         e_max = 10**np.ceil(np.log10(e_max))
#         if xlim:
#             e_min = np.min([e_min, xlim[0]])
#             e_max = np.max([e_max, xlim[1]])
#         else:
#             xlim = (1, e_max)
#         ene = np.logspace(np.log10(e_min), np.log10(e_max), 5000)

#         fig = plt.figure(figsize=(8, 6), dpi=150)
#         for cs in range(1, self._z+1):
#             res = []
#             for e in ene:
#                 res.append(self.cross_section(cs, e))
#             plt.loglog(ene, np.array(res), lw=1, label=str(cs)+"+")
#         # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#         plt.title(self._name + " (Z = " + str(self._z) +
#                   ") Radiative recombination ionisation cross sections")
#         plt.xlabel("Electron kinetic energy (eV)")
#         plt.ylabel("Cross section (cm$^2$)")
#         plt.grid(which="both", alpha=0.5)
#         plt.xlim(xlim)
#         if ylim:
#             plt.ylim(ylim)
#         labelLines(plt.gca().get_lines(), size=7, bbox={"pad":0.1, "fc":"w", "ec":"none"})
#         return fig

#     def show_plot(self, xlim=None, ylim=None):
#         """
#         Creates a figure showing the Lotz Cross section and calls the show method

#         Input Parameters
#         xlim, ylim - plot limits (optional)
#         """
#         self.create_plot(xlim, ylim)
#         plt.show()


# class KLLCrossSections:
#     """
#     An object that provides convenient handling of the KLL DR cross sections
#     UNIT: cm^2
#     """
#     def __init__(self, element):
#         """
#         Initiates the object by importing the relevant KLL DR information
#         extracted from FAC simulation results (performed by Steffen Kuehn)

#         Input Parameters
#         element - Atomic Number or Element Symbol or Element object
#         """
#         if not isinstance(element, ChemicalElement):
#             element = ChemicalElement(element)
#         self._z = element.atomic_number
#         self._es = element.symbol
#         self._name = element.name

#         self._cache = {} # Cache for Cross Section Matrices

#         self._load_kll_data()

#     def _import_kll_block(self, fname):
#         """
#         Private Helper Method for importing the prepared DR data (for one charge state)

#         Input parameters
#         fname - filename
#         """
#         frame = pd.read_csv(fname, sep=' ', index_col=False)
#         return frame

#     def _load_kll_data(self):
#         """
#         Private Helper Method for loading the DR Data
#         """
#         # Convenient expression that generates the full path to the DR files in a robust manner
#         kll_path_gen = lambda x: _GETRESDIR("./DRDataKLLGroundState/"
#                                             + self._es + "." + x + ".dr.txt")
#         # Load files
#         he_like = self._import_kll_block(kll_path_gen("He"))
#         li_like = self._import_kll_block(kll_path_gen("Li"))
#         be_like = self._import_kll_block(kll_path_gen("Be"))
#         b_like = self._import_kll_block(kll_path_gen("B"))
#         c_like = self._import_kll_block(kll_path_gen("C"))
#         n_like = self._import_kll_block(kll_path_gen("N"))
#         o_like = self._import_kll_block(kll_path_gen("O"))

#         self._kll_frames = {2:he_like, 3:li_like, 4:be_like,
#                             5:b_like, 6:c_like, 7:n_like, 8:o_like}
#         self._kll_names = {2:"He-like", 3:"Li-like", 4:"Be-like",
#                            5:"B-like", 6:"C-like", 7:"N-like", 8:"O-like"}

#     def _gaussian_line(self, mean, fwhm, strength):
#         """
#         Private Helper function
#         Comfortable handle to create a gaussian profile function

#         Input Parameters
#         mean - mean of the gaussian
#         fwhm - Full width half max
#         strength - Strength of the line (area under curve)
#         """
#         sig = fwhm/(2*np.sqrt(2*np.log(2)))
#         def g(x):
#             """a gaussian line function with a given width and strength"""
#             return strength * scipy.stats.norm.pdf(x, loc=mean, scale=sig)
#         return g

#     def _spectrum(self, line_list):
#         """
#         Private Helper function
#         Generates the sum of a list of gaussian lines

#         Input parameters
#         line_list - list of line profiles to add to a single function
#         """
#         def spec(x):
#             """function holding the sum of gaussians for each transition"""
#             return sum(l(x) for l in line_list)
#         return spec

#     def _cross_section_func(self, rem_el, fwhm):
#         """
#         Helper function returning a function handle to a function computing the cross section
#         for a given number of remaining electrons and energy spread
#         """
#         kll_fr = self._kll_frames[rem_el]
#         lines = []
#         for (_, row) in kll_fr.iterrows():
#             # Rescale FAC cross sections (10^-20cm^2) to cm^2
#             lines.append(self._gaussian_line(row.DELTA_E, fwhm, row.DR_RECOMB_STRENGTH * 1e-20))
#         return self._spectrum(lines) # Handle to the function that can compute cross_sec

#     def cross_section(self, cs, e_kin, fwhm):
#         """
#         Computes the Lotz cross section of a given charge state at a given electron energy
#         UNIT: cm^2

#         Input Parameters
#         cs - Charge State (0 for neutral atom)
#         e_kin - kinetic energy of projectile Electron
#         fwhm - width of the spectral lines (gaussian electron energy spread)
#         """
#         rem_el = self._z - cs
#         if rem_el not in range(2, 8+1):
#             return 0
#         cs_spectrum = self._cross_section_func(rem_el, fwhm)
#         return cs_spectrum(e_kin)

#     def cross_section_matrix(self, e_kin, fwhm):
#         """
#         Computes and returns a matrix with the KLL cross sections for a given electron energy
#         that can be used to solve the rate equations in a convenient manner.

#         Matrices are cached for better performance, be careful as this can occupy memory when a lot
#         of energies / fwhms are polled over time

#         UNIT: cm^2

#         Input Parameters
#         e_kin - Electron kinetic energy
#         fwhm - width of the spectral lines (gaussian electron energy spread)
#         """
#         # Check cache
#         if (e_kin, fwhm) in self._cache:
#             return self._cache[(e_kin, fwhm)]

#         # If nonexistent, compute:
#         A = np.zeros((self._z+1, self._z+1))
#         for rem_el in range(2, 8+1):
#             cs = self._z - rem_el
#             cross_sec = self.cross_section(cs, e_kin, fwhm)
#             A[cs, cs] = -cross_sec
#             A[cs-1, cs] = cross_sec

#         self._cache[(e_kin, fwhm)] = A
#         return A

#     def create_plot(self, fwhm, xlim=None, ylim=None):
#         """
#         Creates a figure showing the KLL Cross section and returns the figure handle

#         Input Parameters
#         xlim, ylim - plot limits (optional)
#         fwhm - width of the spectral lines (gaussian electron energy spread)
#         """
#         e_min = self._kll_frames[2].DELTA_E.min() - 3 * fwhm
#         e_max = self._kll_frames[8].DELTA_E.min() + 3 * fwhm
#         if xlim:
#             e_min = np.min(e_min, xlim[0])
#             e_max = np.max(e_max, xlim[1])
#         else:
#             xlim = (e_min, e_max)
#         ene = np.arange(e_min, e_max)

#         fig = plt.figure(figsize=(12, 4), dpi=150)
#         # x = np.arange(2300, 2800)
#         for rem_el in range(2, 8+1):
#             func = self._cross_section_func(rem_el, fwhm)
#             plt.plot(ene, 1e20 * func(ene), label=self._kll_names[rem_el])
#         # for sp, na in zip(dr_spectra(fwhm), dr_names):
#         #     plt.plot(x, sp(x), label=na)
#         plt.legend()
#         plt.xlabel('Electron beam energy (eV)')
#         plt.ylabel('Cross Section ($10^{-20}$ cm$^2$)')
#         plt.title(self._name + " (Z = "
#                   + str(self._z) + ") KLL-Recombination Cross Sections / Electron Energy FWHM = "
#                   + str(fwhm) + " eV")
#         plt.tight_layout()
#         plt.grid(which="both")
#         plt.xlim(xlim)
#         if ylim:
#             plt.ylim(ylim)
#         return fig

#     def show_plot(self, fwhm, xlim=None, ylim=None):
#         """
#         Creates a figure showing the KLL Cross section and calls the show method

#         Input Parameters
#         xlim, ylim - plot limits (optional)
#         fwhm - width of the spectral lines (gaussian electron energy spread)
#         """
#         self.create_plot(fwhm, xlim, ylim)
#         plt.show()
