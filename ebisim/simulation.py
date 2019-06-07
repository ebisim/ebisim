"""
Module containing classes that provide interface for problem setup and solution
"""
import sys
import numpy as np
import pandas as pd
import scipy.integrate

from . import plotting
from . import xs
from . import elements
from . import plasma
from .physconst import Q_E, M_P, PI
from .physconst import MINIMAL_DENSITY, MINIMAL_KBT

class Result:
    """
    Instances of this class are containers for the results of ebisim simulations and contain a
    variety of convenience methods for simple plot generation etc.

    The required attributes can either be set during instantiation or manually added afterwards.

    Parameters
    ----------
    param : dict, optional
        A dictionary containing general simulation parameters, by default None.
    t : numpy.array, optional
        An array holding the time step coordinates, by default None.
    N : numpy.array, optional
        An array holding the occupancy of each charge state at a given time, by default None.
    kbT : numpy.array, optional
        An array holding the temperature of each charge state at a given time, by default None.
    N_is_density : bool, optional
        Indicates whether N describes the occupany in abstract terms or as an actual density.
        This has an influence on some default plot labels, by default False.

    """

    def __init__(self, param=None, t=None, N=None, kbT=None, N_is_density=False):
        self.param = param
        self.t = t
        self.N = N
        self.kbT = kbT
        self.N_is_density = N_is_density


    def _param_title(self, stub):
        """
        Generates a plot title by merging the stub with some general simulation parameters.
        Defaults to the stub if no parameters are available

        Parameters
        ----------
        stub : str
            Title stub for the plot

        Returns
        -------
        str
            A LaTeX formatted title string compiled from the stub, current density and fwhm.

        """
        if self.param:
            title = f"{self.param['element'].latex_isotope()} {stub.lower()} " \
                    f"($j = {self.param['j']:0.1f}$ A/cm$^2$, " \
                    f"$E_{{e}} = {self.param['e_kin']:0.1f}$ eV)"
            if "fwhm" in self.param:
                title = title[:-1] + f", FWHM = {self.param['fwhm']:0.1f} eV)"
        else:
            title = stub
        return title


    def plot_charge_states(self, relative=False, **kwargs):
        """
        Plot the charge state evolution of this result object.

        Parameters
        ----------
        relative : bool, optional
            Flags whether the absolute numbers relative fraction should be plotted at each
            timestep, by default False.
        **kwargs
            Keyword arguments are handed down to plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.figure.Figure
            Figure handle of the generated plot

        Raises
        ------
        ValueError
            If the required data (self.t, self.N) is not available, i.e.
            corresponding attributes of this Result instance have not been set correctly.

        """
        if self.t is None or self.N is None:
            raise ValueError("The t or N field does not contain any plottable data.")

        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault("ylim", (0, self.N.sum(axis=0).max()*1.05))
        kwargs.setdefault("title", self._param_title("Charge states"))
        kwargs.setdefault("yscale", "linear")
        kwargs.setdefault("plot_total", True)

        if self.N_is_density and not relative:
            kwargs.setdefault("ylabel", "Density (m$^{-3}$)")
        else:
            kwargs.setdefault("ylabel", "Relative Abundance")

        if relative:
            fig = plotting.plot_generic_evolution(self.t, self.N/np.sum(self.N, axis=0), **kwargs)
        else:
            fig = plotting.plot_generic_evolution(self.t, self.N, **kwargs)
        return fig


    plot = plot_charge_states #: Alias for plot_charge_states


    def plot_energy_density(self, **kwargs):
        """
        Plot the energy density evolution of this result object.

        Parameters
        ----------
        **kwargs
            Keyword arguments are handed down to plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.figure.Figure
            Figure handle of the generated plot

        Raises
        ------
        ValueError
            If the required data (self.t, self.kbT) is not available, i.e.
            corresponding attributes of this Result instance have not been set correctly.

        """
        if self.t is None or self.kbT is None:
            raise ValueError("The t or kbT field does not contain any plottable data.")

        e_den = self.N * self.kbT
        ymin = 10**(np.floor(np.log10(e_den[:, 0].sum(axis=0)) - 1))
        ymax = 10**(np.ceil(np.log10(e_den.sum(axis=0).max()) + 1))

        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault("ylim", (ymin, ymax))
        kwargs.setdefault("title", self._param_title("Energy density"))
        kwargs.setdefault("ylabel", "Energy density (eV / m$^{-3}$)")
        kwargs.setdefault("plot_total", True)

        fig = plotting.plot_generic_evolution(self.t, e_den, **kwargs)
        return fig


    def plot_temperature(self, **kwargs):
        """
        Plot the temperature evolution of this result object.

        Parameters
        ----------
        **kwargs
            Keyword arguments are handed down to plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.figure.Figure
            Figure handle of the generated plot

        Raises
        ------
        ValueError
            If the required data (self.t, self.kbT) is not available, i.e.
            corresponding attributes of this Result instance have not been set correctly.

        """
        if self.t is None or self.kbT is None:
            raise ValueError("The t or kbT field does not contain any plottable data.")

        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault("ylim", (0.01, 10**(np.ceil(np.log10(self.kbT.max()) + 1))))
        kwargs.setdefault("title", self._param_title("Temperature"))
        kwargs.setdefault("ylabel", "Temperature (eV)")

        fig = plotting.plot_generic_evolution(self.t, self.kbT, **kwargs)
        return fig


def basic_simulation(element, j, e_kin, t_max,
                     dr_fwhm=None, N_initial=None, CNI=False,
                     solver_kwargs=None):
    """
    Interface for performing basic charge breeding simulations.

    These simulations only include the most important effects, i.e. electron ionisation,
    radiative recombination and optionally dielectronic recombination (for those transitions whose
    data is available in the resource directory). All other effects are not ignored.

    The results only represent the proportions of different charge states, not actual densities.

    Parameters
    ----------
    element : ebisim.elements.Element or str or int
        An instance of the Element class, or an identifier for the element, i.e. either its
        name, symbol or proton number.
    j : float
        <A/cm^2>
        Current density
    e_kin : float
        <eV>
        Electron beam energy
    t_max : float
        <s>
        Simulated breeding time
    dr_fwhm : None or float, optional
        <eV>
        If a value is given, determines the energy spread of the electron beam
        (in terms of Full Width Half Max) and hence the effective width of DR resonances.
        Otherwise DR is excluded from the simulation. By default None.
    N_initial : None or numpy.array, optional
        Determines the initial charge state distribution if given, must have Z + 1 entries, where
        the array index corresponds to the charge state.
        If no value is given the distribution defaults to 100 % 1+ ions at t = 0 s, or a small
        amount of neutral atoms in the case of continuous neutral injection.
        By default None.
    CNI : bool, optional
        Determines whether there is a continuous injection of neutral atoms to the distribution.
        If True, the feed rate is 1/s.
        By default False.
    solver_kwargs : dict, optional
        If supplied these keyword arguments are unpacked in the solver call, refer to
        the documentation of scipy.integrate.solve_ivp for more information.
        By default None.

    Returns
    -------
    ebisim.simulation.Result
        An instance of the Result class, holding the simulation parameters, timesteps and
        charge state distribution.
    """

    # cast element to Element if necessary
    if not isinstance(element, elements.Element):
        element = elements.Element(element)

    # set initial conditions if not supplied by user
    if not N_initial:
        N_initial = np.zeros(element.z + 1)
        if CNI:
            N_initial[0] = 1e-12
        else:
            N_initial[1] = 1

    # prepare solver options
    if not solver_kwargs:
        solver_kwargs = {}
    solver_kwargs.setdefault("method", "LSODA")

    # save adjusted call parameters for passing on to Result
    param = locals().copy()

    # convert current density A/cm**2 to particle flux density electrons/s/m**2
    j = j * 1.e4 / Q_E

    # compute cross section
    xs_mat = xs.eixs_mat(element, e_kin) + xs.rrxs_mat(element, e_kin)
    if dr_fwhm:
        xs_mat += xs.drxs_mat(element, e_kin, dr_fwhm)

    # the jacobian of a basic simulation
    _jac = j * xs_mat
    if solver_kwargs["method"] == "LSODA":
        jac = lambda _, N: _jac # LSODA solver requires "callable" jacobian
    else:
        jac = _jac

    # the rate equation
    if CNI:
        feed = np.zeros(element.z + 1)
        feed[0] = 1
        dNdt = lambda _, N: _jac.dot(N) + feed
    else:
        dNdt = lambda _, N: _jac.dot(N)

    sol = scipy.integrate.solve_ivp(dNdt, (0, t_max), N_initial, jac=jac, **solver_kwargs)
    return Result(param=param, t=sol.t, N=sol.y)


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
        j - current density in A/cm**2
        fwhm - electron energy spread (for DR width)
        energies - list of the energies to scan
        eval_times - list of the times to evaluate
        """
        self._problem = problemtype(element, j, 0, fwhm)
        if not isinstance(element, elements.Element):
            element = elements.Element(element)
        self._element = element
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

    def solve(self, y0=None, show_progress=True, **kwargs):
        """
        Trigger the computation of the energy scan that has previously been set up
        This can be fairly time consuming!
        """
        # Integration time
        max_time = np.max(self._eval_times)
        # Solve ODES
        scan_solutions = pd.DataFrame()
        prog = 0
        for e_kin in self._energies:
            self._problem.e_kin = e_kin
            solution = self._problem.solve(max_time, y0=y0, t_eval=self._eval_times, **kwargs)
            y = solution.y.T
            if isinstance(self._problem, ComplexEBISProblem):
                y = y[:, :int(y.shape[1]/2)] # Cut off temperatures
                y = y / np.sum(y, axis=1)[:, np.newaxis] # Normalise
            sol_df = pd.DataFrame(y)
            sol_df["t"] = solution.t
            sol_df["e_kin"] = e_kin
            scan_solutions = scan_solutions.append(sol_df, ignore_index=True)
            prog += 1
            if show_progress:
                sys.stdout.write(f"\rProgress:  {100 * prog / len(self._energies):>4.1f}%")
        self._solution = scan_solutions
        if show_progress:
            sys.stdout.writelines([])
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
        data = self._solution.groupby("t").get_group(t)
        if normalise:
            for c in range(self._element.z+1):
                data[c] = data[c]/data[c].mean()
            title = f"Normalised abundance of {self._element.latex_isotope()} " \
                    f"at $T={1000*t:.1f}$ ms"
            ylim = None
        else:
            title = f"Relative abundance of {self._element.latex_isotope()} at $T={1000*t:.1f}$ ms"
            ylim = (0.01, 1)

        fig = plotting.plot_energy_scan(data, cs, ylim=ylim, title=title, invert_hor=invert_hor,
                                        x2fun=x2fun, x2label=x2label)
        return fig

    def plot_abundance_of_cs(self, cs, xlim=None, ylim=None):
        """
        Creates a 2D plot of the abundance of a charge state for a given range of energies and
        breeding times (as available in the solution object)

        Input Parameters
        cs - charge state to plot
        xlim/ylim - axes limits
        """
        title = f"Abundance of {self._element.latex_isotope()}$^{{{cs}+}}$"
        fig = plotting.plot_energy_time_scan(self.solution, cs, xlim=xlim, ylim=ylim, title=title)
        return fig


class ComplexEBISProblem:
    """
    class defining an EBIS charge breeding simulation and providing the interface to solve it
    Time independent problems only

    Solving is done depending on the latest setting for the kinetic energy so this could create
    race conditions. Be a bit careful with that, when doing batch work.
    """

    def __init__(self, element, j, e_kin, fwhm):
        """
        Defines the general problem constants (current density, electron energy and spread)

        Input parameters
        element - Identifier of the element under investigation
        j - current density in A / cm^2
        e_kin - electron energy
        fwhm - electron energy spread (for DR width)
        """
        if not isinstance(element, elements.Element):
            element = elements.Element(element)
        self._element = element
        self._fwhm = fwhm
        self._j = j * 1e4 # convert to A/m**2
        self._e_kin = e_kin
        self._solution = None
        self._Vtrap_ax = 300
        self._Vtrap_ra = 50
        self._B_ax = 2
        self._r_dt = 5e-3
        self._bg_N0 = 1e-7 / (0.025 * Q_E) # 10e-10mbar at 300K
        self._bg_IP = 21.56 # eV
        #Default initial condition for solving the EBIS ODE System (all atoms in 1+ charge state)
        self._default_initial = np.ones(2*(self._element.z + 1))
        self._default_initial[:self._element.z + 1] *= MINIMAL_DENSITY
        self._default_initial[self._element.z + 1:] *= MINIMAL_KBT
        self._default_initial[1] = 1e16
        self._default_initial[self._element.z + 2] = 0.5

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

    def _rhs(self, t, y):
        """
        The right hand side of the ODE system for the complex EBIS problem
        """
        del t # not currently needed
        ### Electron beam stuff
        e_kin = self.e_kin
        ve = plasma.electron_velocity(e_kin)
        je = self._j / Q_E
        Ne = je / ve
        q = np.arange(self._element.z + 1)
        A = self._element.a

        ### Split y vector into density and temperature
        N = y[:self._element.z + 1]
        kbT = y[self._element.z + 1:]
        # E = N * kbT # Not currently needed

        # precompute collision rates
        rij = plasma.ion_coll_rate_mat(N, N, kbT, kbT, A, A)
        ri = np.sum(rij, axis=1)

        ### Particle density rates
        R_ei = je * N * xs.eixs_mat(self._element, e_kin)
        R_rr = je * N * xs.rrxs_mat(self._element, e_kin)
        R_dr = je * N * xs.drxs_mat(self._element, e_kin, self._fwhm)

        sigcx = 1.43e-16 * q**1.17 * self._bg_IP**-2.76
        R_cx = N * self._bg_N0 * np.sqrt(8 * Q_E * np.clip(kbT, 0, None)/(PI * A * M_P)) * sigcx
        R_ax = plasma.escape_rate_axial(N, kbT, ri, self._Vtrap_ax)
        R_ra = plasma.escape_rate_radial(N, kbT, ri, A, self._Vtrap_ra, self._B_ax, self._r_dt)

        ### Energy density rates
        S_ei = R_ei * kbT
        S_rr = R_rr * kbT
        S_dr = R_dr * kbT
        S_cx = R_cx * kbT
        S_ax = R_ax * (kbT + q * self._Vtrap_ax)
        S_ra = R_ra * (kbT + q * (self._Vtrap_ra + self._r_dt * self._B_ax * \
                                  np.sqrt(2 * Q_E * np.clip(kbT, 0, None) / (3 * A *M_P))))
        # Electron heating
        S_eh = plasma.electron_heating_vec(N, Ne, kbT, e_kin, A)
        # Energy transfer between charge states within same "species"
        S_tr = plasma.energy_transfer_vec(N, N, kbT, kbT, A, A, rij)

        ### Construct rhs for N (density)
        R_tot = -(R_ei + R_rr + R_dr + R_cx) - (R_ax + R_ra)
        R_tot[1:] += R_ei[:-1]
        R_tot[:-1] += R_rr[1:] + R_dr[1:] + R_cx[1:]

        ### Construct rates for energy density flow
        S_tot = -(S_ei + S_rr + S_dr + S_cx) + S_eh + S_tr - (S_ax + S_ra)
        S_tot[1:] += S_ei[:-1]
        S_tot[:-1] += S_rr[1:] + S_dr[1:] + S_cx[1:]

        ### Deduce temperature flow -> Integrating temperature instead of energy has proven more
        ### numerically stable
        Q_tot = (S_tot - kbT * R_tot) / N
        Q_tot[N <= MINIMAL_DENSITY] = 0 ### Freeze temperature if density is low (-> not meaningful)

        return np.concatenate((R_tot, Q_tot))

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
        lb = np.ones(y0.size) * MINIMAL_DENSITY
        lb[self._element.z + 1:] *= MINIMAL_KBT
        method = kwargs.pop("method", "Radau")
        solution = scipy.integrate.solve_ivp(self._rhs, [0, max_time], y0, method=method, **kwargs)
        self._solution = solution
        return solution

    def plot_cs_evo(self):
        """
        After the problem has been solved, the charge state evolution can be plotted by calling
        this function

        Returns figure handle and does not call show()
        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        tmax = self.solution.t.max()
        xlim = (1e-5, tmax)
        title = f"{self._element.latex_isotope()} charge state evolution " \
                f"($E_{{e}} = {self._e_kin:0.1f}$ eV, FWHM = {self._fwhm:0.1f} eV)"
        t = self.solution.t
        N = self.solution.y[:self._element.z + 1, :]
        ylim = (0, N.sum(axis=0).max()*1.05)
        ylabel = ("Density (m$^{-3}$)")
        fig = plotting.plot_generic_evolution(t, N, xlim=xlim, ylim=ylim, ylabel=ylabel,
                                              title=title, yscale="linear", plot_total=True)
        return fig

    def plot_energy_evo(self):
        """
        After the problem has been solved, the energy denisty evolution can be plotted by calling
        this function

        Returns figure handle and does not call show()
        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        tmax = self.solution.t.max()
        xlim = (1e-5, tmax)
        title = f"{self._element.latex_isotope()} energy density evolution " \
                f"($E_{{e}} = {self._e_kin:0.1f}$ eV, FWHM = {self._fwhm:0.1f} eV)"
        t = self.solution.t
        E = self.solution.y[self._element.z + 1:, :] * self.solution.y[:self._element.z + 1, :]
        ymin = 10**(np.floor(np.log10(E[:, 0].sum(axis=0)) - 1))
        ymax = 10**(np.ceil(np.log10(E.sum(axis=0).max()) + 1))
        ylim = (ymin, ymax)
        ylabel = ("Energy density (eV / m$^{-3}$)")
        fig = plotting.plot_generic_evolution(t, E, xlim=xlim, ylim=ylim, ylabel=ylabel,
                                              title=title, plot_total=True)
        return fig

    def plot_temperature_evo(self):
        """
        After the problem has been solved, the temperature evolution can be plotted by calling
        this function

        Returns figure handle and does not call show()
        """
        if self.solution is None:
            print("Error! Need to solve problem before plotting")
        tmax = self.solution.t.max()
        xlim = (1e-5, tmax)
        title = f"{self._element.latex_isotope()} temperature evolution " \
                f"($E_{{e}} = {self._e_kin:0.1f}$ eV, FWHM = {self._fwhm:0.1f} eV)"
        t = self.solution.t
        T = self.solution.y[self._element.z + 1:, :]
        ymin = 0.01
        ymax = 10**(np.ceil(np.log10(T.max()) + 1))
        ylim = (ymin, ymax)
        ylabel = ("Temperature (eV)")
        fig = plotting.plot_generic_evolution(t, T, xlim=xlim, ylim=ylim, ylabel=ylabel,
                                              title=title)
        return fig
