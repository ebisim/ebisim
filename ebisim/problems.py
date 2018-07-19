"""
Module containing classes that provide interface for problem setup and solution
"""
import numpy as np
import pandas as pd
import scipy.integrate

from . import plotting
from . import xs
from . import elements
from .physconst import Q_E

class SimpleEBISProblem:
    """
    class defining an EBIS charge breeding simulation and providing the interface to solve it
    """

    def __init__(self, element, j, e_kin, fwhm):
        """
        Defines the general problem constants (current density, electron energy and spread)

        Input parameters
        element - Identifier of the element under investigation
        j - current density
        e_kin - electron energy
        fwhm - electron energy spread (for DR width)
        """
        #species - the Species object containing the physical information about cross section etc
        self._species = xs.EBISSpecies(element, fwhm)
        self._j = j
        self._e_kin = e_kin
        self._solution = None
        #Default initial condition for solving the EBIS ODE System (all atoms in 1+ charge state)
        self._default_initial = np.zeros(self._species.element.z + 1)
        self._default_initial[1] = 1

    @property
    def solution(self):
        """Returns the solution of the lates solve of this problem"""
        return self._solution

    @property
    def e_kin(self):
        """Returns the kinetic energy"""
        return self._e_kin

    @e_kin.setter
    def e_kin(self, val):
        """Set e_kin to new value and delete existing solution"""
        if val != self._e_kin:
            self._solution = None
            self._e_kin = val

    def _ode_system(self, _, y): # No time dependence in this problem
        """
        The ode system describing the charge breeding
        """
        j = self._j / Q_E
        xs_mat = self._species.iixs.xs_matrix(self._e_kin) \
                 + self._species.rrxs.xs_matrix(self._e_kin) \
                 + self._species.drxs.xs_matrix(self._e_kin) \
        # dydt = self._species.LotzCrossSections.cross_section_matrix(self._e_kin).dot(y)
        #dydt += self._species.KLLCrossSections.cross_section_matrix(self._e_kin, self._fwhm).dot(y)
        dydt = j * xs_mat.dot(y)
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

    def plot_solution(self):
        """
        After the problem has been solved, the charge state evolution can be plotted by calling
        this function

        Returns figure handle and does not call show()
        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        tmax = self.solution.t.max()
        xlim = (1e-4, tmax)
        title = "$_{%d}$%s charge state evolution, $E_{kin} = %0.1f$ eV, FWHM = %0.1f eV)"\
                %(self._species.element.z, self._species.element.symbol,
                  self._e_kin, self._species.fwhm)
        return plotting.plot_cs_evolution(self.solution, xlim=xlim, title=title)

class ContinuousNeutralInjectionEBISProblem(SimpleEBISProblem):
    """
    The class has been modified to increase the abundance in the neutral charge state
    at a constant rate to mimick continous neutral injection
    """

    def __init__(self, element, j, e_kin, fwhm):
        """
        Initialiser for a continuous injection problem, calls init of SimpleEBISProblem and adjusts
        the default initial condition, i.e. emtpy trap.
        """
        super().__init__(element, j, e_kin, fwhm)
        #Default initial condition for solving the EBIS ODE System (all atoms in neutral state)
        self._default_initial = np.zeros(self._species.element.z + 1)
        self._default_initial[0] = 1e-9

    def _ode_system(self, _, y): # No time dependence in this problem
        """
        The ode system describing the charge breeding
        """
        dydt = super()._ode_system(None, y)
        dydt[0] += 1
        return dydt

class EnergyScan:
    """
    Class that provides a simple interface to perform a scan of the electron beam energy for
    a SimpleEBISProblem (and inherited)
    """
    def __init__(self, problemtype, element, j, fwhm, energies, eval_times):
        """
        Initialises the energy scan settings

        Input Parameters
        problemtype - class handle specifiying the problem
        element - Identifier of the element under investigation
        j - current density
        fwhm - electron energy spread (for DR width)
        energies - list of the energies to scan
        eval_times - list of the times to evaluate
        """
        # species - the Species object containing the physical information about cross section etc
        self._problem = problemtype(element, j, 0, fwhm)
        self._element = elements.ChemicalElement(element)
        self._j = j
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
            self._problem.e_kin = e_kin
            solution = self._problem.solve(max_time, t_eval=self._eval_times)
            sol_df = pd.DataFrame(solution.y.T)
            sol_df["t"] = solution.t
            sol_df["e_kin"] = e_kin
            scan_solutions = scan_solutions.append(sol_df, ignore_index=True)
        self._solution = scan_solutions
        return scan_solutions

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
        # data = self._solution.loc[self._solution["t"] == t]
        data = self._solution.loc.groupyby("t").get_group(t)
        if normalise:
            for c in range(self._element.z+1):
                data[c] = data[c]/data[c].mean()
            title = ("Normalised abundance of $_{%d}$%s at $T=%.1G$ ms"
                     %(self._element.z, self._element.symbol, 1000*t))
            ylim = None
        else:
            title = ("Relative abundance of $_{%d}$%s at $T=%.1G$ ms"
                     %(self._element.z, self._element.symbol, 1000*t))
            ylim = (0.01, 1)

        fig = plotting.plot_energy_scan(data, cs, ylim=ylim, title=title, invert_hor=invert_hor,
                                        x2fun=x2fun, x2label=x2label)
        return fig

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
