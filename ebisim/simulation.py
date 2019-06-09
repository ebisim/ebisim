"""
Module containing classes that provide interface for problem setup and solution
"""

import sys
from warnings import warn
import numpy as np
import pandas as pd
import scipy.integrate
# import numba

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
    ode_sol : optional
        The solution object returned by scipy.integrate.solve_ivp. This can contain useful
        information about the solver performance etc. Refer to scipy documentation for details.

    """

    def __init__(self, param=None, t=None, N=None, kbT=None, N_is_density=False, ode_sol=None):
        self.param = param
        self.t = t
        self.N = N
        self.kbT = kbT
        self.N_is_density = N_is_density
        self.ode_sol = ode_sol

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

    Continuous Neutral Injection (CNI) can be activated on demand.

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
        If no value is given the distribution defaults to 100% of 1+ ions at t = 0 s, or a small
        amount of neutral atoms in the case of CNI.
        By default None.
    CNI : bool, optional
        Determines whether there is a continuous addition of neutral atoms to the distribution.
        If True, the feed rate is 1/s.
        By default False.
    solver_kwargs : None or dict, optional
        If supplied these keyword arguments are unpacked in the solver call.
        Refer to the documentation of scipy.integrate.solve_ivp for more information.
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
    return Result(param=param, t=sol.t, N=sol.y, ode_sol=sol)


def advanced_simulation(element, j, e_kin, t_max,
                        dr_fwhm=None, N_kbT_initial=None, adv_param=None,
                        solver_kwargs=None):
    """
    !!!UNDER ACTIVE DEVELOPMENT!!! - API not stable!

    Interface for performing advanced charge breeding simulations.

    These simulations only include the most important effects, i.e. electron ionisation,
    radiative recombination and optionally dielectronic recombination (for those transitions whose
    data is available in the resource directory). All other effects are not ignored.

    Continuous Neutral Injection (CNI) can be activated on demand.

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
    N_kbT_initial : None or numpy.array, optional
        <1/m^3 and eV>
        Determines the initial charge state distribution and temperatures if given.
        Must have 2(Z + 1) entries, where the first Z + 1 entries give the ion densities in
        1/m^3 with the array index corresponding to the charge state and
        the last Z + 1 entries describe the temperatures in eV.
        If no value is given the distribution defaults to approx 100% of 1+ ions at t = 0 s with a
        temperature of 0.5 eV.
        By default None.
    adv_param : None or dict, optional
        This dict can be used to set advanced parameters of the simulation.
    solver_kwargs : None or dict, optional
        If supplied these keyword arguments are unpacked in the solver call.
        Refer to the documentation of scipy.integrate.solve_ivp for more information.
        By default None.

    Returns
    -------
    ebisim.simulation.Result
        An instance of the Result class, holding the simulation parameters, timesteps and
        charge state distribution including the species temperature.
    """
    # TODO: verify docstring

    # cast element to Element if necessary
    if not isinstance(element, elements.Element):
        element = elements.Element(element)

    # set initial conditions if not supplied by user
    if not N_kbT_initial:
        N_kbT_initial = np.ones(2 * (element.z + 1))
        N_kbT_initial[:element.z + 1] *= MINIMAL_DENSITY
        N_kbT_initial[element.z + 1:] *= MINIMAL_KBT
        N_kbT_initial[1] = 1e16
        N_kbT_initial[element.z + 2] = 0.5

    if adv_param is None:
        adv_param = {}
    adv_param.setdefault("Vtrap_ax", 300)
    adv_param.setdefault("Vtrap_ra", 50)
    adv_param.setdefault("B_ax", 2)
    adv_param.setdefault("r_dt", 0.005)
    adv_param.setdefault("bg_N0", 1e-7 / (0.025 * Q_E)) # 10e-10mbar at 300K
    adv_param.setdefault("bg_IP", 21.56) # eV

    # prepare solver options
    if not solver_kwargs:
        solver_kwargs = {}
    solver_kwargs.setdefault("method", "Radau")

    # save adjusted call parameters for passing on to Result
    param = locals().copy()

    # @numba.jit
    def rhs(_, y):
        ve = plasma.electron_velocity(e_kin)
        je = j / Q_E * 1e4
        Ne = je / ve
        q = np.arange(element.z + 1)
        A = element.a

        ### Split y vector into density and temperature
        N = y[:element.z + 1]
        kbT = y[element.z + 1:]
        # E = N * kbT # Not currently needed

        # precompute collision rates
        rij = plasma.ion_coll_rate_mat(N, N, kbT, kbT, A, A)
        ri = np.sum(rij, axis=1)

        ### Particle density rates
        R_ei = je * N * xs.eixs_vec(element, e_kin)
        R_rr = je * N * xs.rrxs_vec(element, e_kin)
        if dr_fwhm is not None:
            R_dr = je * N * xs.drxs_vec(element, e_kin, dr_fwhm)
        else:
            R_dr = np.zeros(element.z + 1)

        sigcx = 1.43e-16 * q**1.17 * adv_param["bg_IP"]**-2.76
        R_cx = N * adv_param["bg_N0"] * \
            np.sqrt(8 * Q_E * np.clip(kbT, 0, None)/(PI * A * M_P)) * sigcx
        R_ax = plasma.escape_rate_axial(N, kbT, ri, adv_param["Vtrap_ax"])
        R_ra = plasma.escape_rate_radial(
            N, kbT, ri, A, adv_param["Vtrap_ra"], adv_param["B_ax"], adv_param["r_dt"]
            )

        ### Energy density rates
        S_ei = R_ei * kbT
        S_rr = R_rr * kbT
        S_dr = R_dr * kbT
        S_cx = R_cx * kbT
        S_ax = R_ax * (kbT + q * adv_param["Vtrap_ax"])
        S_ra = R_ra * (kbT + q * (adv_param["Vtrap_ra"] + adv_param["r_dt"] * adv_param["B_ax"] * \
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

    sol = scipy.integrate.solve_ivp(rhs, (0, t_max), N_kbT_initial, **solver_kwargs)
    return Result(
        param=param,
        t=sol.t,
        N=sol.y[:element.z + 1, :],
        kbT=sol.y[element.z + 1:, :],
        N_is_density=True,
        ode_sol=sol
        )


def energy_scan(sim_func, sim_kwargs, energies):
    if "e_kin" in sim_kwargs:
        sim_kwargs = sim_kwargs.copy()
        del sim_kwargs["e_kin"]
        warn(f"sim_kwargs contains a value for e_kin, this item will be ignored.")

    sim_kwargs.setdefault("solver_kwargs", {})
    sim_kwargs["solver_kwargs"]["dense_output"] = True # need dense output for interpolation

    energies = np.array(energies)
    energies.sort()

    def sim(e_kin):
        return sim_func(e_kin=e_kin, **sim_kwargs)

    results = [sim(e_kin) for e_kin in energies]
    return EnergyScanResult(energies, results)


class EnergyScanResult:
    def __init__(self, energies, results):
        self._energies = energies
        self._results = results
        self._t_max = min([res.param["t_max"] for res in results])
        self._element = self._results[0].param["element"]


    def abundance_at_time(self, t):
        if t < 0 or t > self._t_max:
            raise ValueError("This time has not been simulated during the energyscan.")
        per_energy = [res.ode_sol.sol(t) for res in self._results]
        return self._energies, np.column_stack(per_energy)


    def abundance_of_cs(self, cs):
        if cs > self._element.z:
            raise ValueError("This charge state is not available for the given element.")
        times = np.logspace(-4, np.log10(self._t_max), 500)
        print(times)
        times = np.clip(times, a_min=0, a_max=self._t_max)
        print(times)
        per_time = [self.abundance_at_time(t)[1][cs, :] for t in times]
        return self._energies, times, np.row_stack(per_time)


    def plot_abundance_at_time(self, t, cs=None, **kwargs):
        energies, abundance = self.abundance_at_time(t)
        kwargs.setdefault("title",
                          f"Abundance of {self._element.latex_isotope()} at $T={1000*t:.1f}$ ms")
        return plotting.plot_energy_scan(
            energies=energies,
            abundance=abundance,
            cs=cs,
            **kwargs
        )


    def plot_abundance_of_cs(self, cs, **kwargs):
        energies, times, abundance = self.abundance_of_cs(cs)
        kwargs.setdefault("title",
                          f"Abundance of {self._element.latex_isotope()}$^{{{cs}+}}$")
        return plotting.plot_energy_time_scan(
            energies=energies,
            times=times,
            abundance=abundance,
            **kwargs
        )
