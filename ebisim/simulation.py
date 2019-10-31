"""
This module containins functions and classes provide an interface to run simulations and inspect
their results.
"""

from warnings import warn
from multiprocessing.pool import Pool
from collections import namedtuple
import numpy as np
import scipy.integrate
import scipy.interpolate

from . import plotting
from . import xs
from . import elements
from . import plasma
from .physconst import Q_E, M_P, PI, EPS_0
from .physconst import MINIMAL_DENSITY, MINIMAL_KBT


_RATE_NAMES = dict(
    R_ei="Electron ionisation",
    R_rr="Radiative recombination",
    R_dr="Dielectronic recombination",
    R_cx="Charge exchange",
    R_ax="Axial losses",
    R_ra="Radial losses",
    S_ei="Electron ionisation",
    S_rr="Radiative recombination",
    S_dr="Dielectronic recombination",
    S_cx="Charge exchange",
    S_ax="Axial losses",
    S_ra="Radial losses",
    S_eh="Electron heating",
    S_tr="Heat transfer",
    S_ih="Ionisation heating",
    V_ii="Self collision rate",
    V_it="Total collision rate",
    Comp="Charge compensation"
) #: Rates names for plot annotation


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
    res : optional
        The result object returned by scipy.integrate.solve_ivp. This can contain useful
        information about the solver performance etc. Refer to scipy documentation for details,
        by default None.
    rates : dict, optional
        A dictionary containing the different breeding rates in arrays shaped like N,
        by default None.

    """

    def __init__(self, param=None, t=None, N=None, kbT=None, res=None, rates=None):
        self.param = param
        self.t = t
        self.N = N
        self.kbT = kbT
        self.res = res
        self.rates = rates


    def times_of_highest_abundance(self):
        """
        Yields the point of time with the highest abundance for each charge state

        Returns
        -------
        numpy.array
            <s>
            Array of times.

        """
        args = np.argmax(self.N, axis=1)
        return self.t[args]


    def abundance_at_time(self, t):
        """
        Yields the abundance of each charge state at a given time

        Parameters
        ----------
        t : float
            <s>
            Point of time to evaluate.

        Returns
        -------
        numpy.array
            Abundance of each charge state, array index corresponds to charge state.

        """
        if self.res and self.res.sol:
            return self.res.sol(t)
        interp = scipy.interpolate.interp1d(self.t, self.N)
        return interp(t)

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
            if "target" in self.param:
                target = self.param["target"]
                element = target.element
            else:
                element = self.param["element"]
            if "device" in self.param:
                device = self.param["device"]
                e_kin = device.e_kin
                j = device.j
            else:
                e_kin = self.param["e_kin"]
                j = self.param["j"]
            title = f"{element.latex_isotope()} {stub.lower()} " \
                    f"($j = {j:0.1f}$ A/cm$^2$, $E_{{e}} = {e_kin:0.1f}$ eV)"
            if self.param.get("dr_fwhm", None) is not None:
                title = title[:-1] + f", FWHM = {self.param['dr_fwhm']:0.1f} eV)"
        else:
            title = stub
        return title


    def plot_charge_states(self, relative=False, **kwargs):
        """
        Plot the charge state evolution of this result object.

        Parameters
        ----------
        relative : bool, optional
            Flags whether the absolute numbers or a relative fraction should be plotted at each
            timestep, by default False.
        **kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.Figure
            Figure handle of the generated plot.

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

        if relative or self.kbT is None: # Hack to determine whether basic or advanced sim
            kwargs.setdefault("ylabel", "Relative abundance")
        else:
            kwargs.setdefault("ylabel", "Density (m$^{-3}$)")

        if relative:
            kwargs["ylim"] = (0, 1.1)
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
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.Figure
            Figure handle of the generated plot.

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
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.Figure
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


    def plot_rate(self, rate_key, **kwargs):
        """
        Plots the requested ionisation- or energy flow rates.

        Parameters
        ----------
        rate_key : str
            The key identifying the rate to be plotted.
            Valid keys are:
            R_ei, R_rr, R_dr, R_cx, R_ax, R_ra, S_ei, S_rr, S_dr, S_cx, S_ax, S_ra, S_eh, S_tr.

        Returns
        -------
        matplotlib.Figure
            Figure handle of the generated plot

        Raises
        ------
        ValueError
            If the required data (self.rates) is not available, or an invalid key is requested.

        """
        if self.rates is None:
            raise ValueError("Rates are not available for this result.")
        if rate_key not in self.rates:
            raise ValueError(
                f"The requested rate_key does not exist. Available rates are {self.rates.keys()}."
            )

        rate = self.rates[rate_key]

        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault("title", self._param_title(_RATE_NAMES.get(rate_key, "unkown rate")))
        kwargs.setdefault("yscale", "linear")
        kwargs.setdefault("plot_total", True)

        if rate_key.startswith("R"):
            kwargs.setdefault("ylabel", "Number density flow (m$^{-3}$ s$^{-1}$)")
        if rate_key.startswith("S"):
            kwargs.setdefault("ylabel", "Energy density flow (eV m$^{-3}$ s$^{-1}$)")

        fig = plotting.plot_generic_evolution(self.t, rate, **kwargs)
        return fig


def basic_simulation(element, j, e_kin, t_max,
                     dr_fwhm=None, N_initial=None, CNI=False,
                     solver_kwargs=None):
    """
    Interface for performing basic charge breeding simulations.

    These simulations only include the most important effects, i.e. electron ionisation,
    radiative recombination and optionally dielectronic recombination (for those transitions whose
    data is available in the resource directory). All other effects are ignored.

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
        element = elements.get_element(element)

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

    res = scipy.integrate.solve_ivp(dNdt, (0, t_max), N_initial, jac=jac, **solver_kwargs)
    return Result(param=param, t=res.t, N=res.y, res=res)


Target = namedtuple("Target", ["element", "N", "kbT", "cni", "cx"])

def get_gas_target(element, N, kbT=0.025, cni=True, cx=True):
    # cast element to Element if necessary
    if not isinstance(element, elements.Element):
        element = elements.get_element(element)
    _N = np.ones(element.z + 1) * MINIMAL_DENSITY
    _kbT = np.ones(element.z + 1) * MINIMAL_KBT
    _N[0] = N
    _kbT[0] = kbT
    return Target(element, _N, _kbT, cni, cx)

def get_ion_target(element, N, kbT=0.025, q=1):
    # cast element to Element if necessary
    if not isinstance(element, elements.Element):
        element = elements.get_element(element)
    _N = np.ones(element.z + 1) * MINIMAL_DENSITY
    _kbT = np.ones(element.z + 1) * MINIMAL_KBT
    _N[q] = N
    _kbT[q] = kbT
    return Target(element, _N, _kbT, cni=False, cx=False)

Device = namedtuple("Device", ["j", "e_kin", "current", "r_e", "v_ax", "v_ra", "b_ax", "r_dt"])

def advanced_simulation(device, target, t_max, dr_fwhm=None, solver_kwargs=None):
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
    N_kbT_initial = np.concatenate((target.N, target.kbT))

    # prepare solver options
    if not solver_kwargs:
        solver_kwargs = {}
    solver_kwargs.setdefault("method", "Radau")

    # save adjusted call parameters for passing on to Result
    param = locals().copy()

    # prepare rhs
    ## electrons
    ve = plasma.electron_velocity(device.e_kin)
    je = device.j / Q_E * 1e4
    Ne = je / ve
    on_ax_sc = device.current/(4 * PI * EPS_0 * ve) * (2*np.log(device.r_e/device.r_dt)-1)
    ## ions
    element = target.element
    q = np.arange(element.z + 1)
    A = element.a
    ## cross sections
    eixs = xs.eixs_vec(element, device.e_kin)
    rrxs = xs.rrxs_vec(element, device.e_kin)
    cxxs = 1.43e-16 * q**1.17 * element.ip**-2.76
    if dr_fwhm is not None:
        drxs = xs.drxs_vec(element, device.e_kin, dr_fwhm)
    else:
        drxs = np.zeros(element.z + 1)

    def rhs(_, y, rates=None):

        ### Split y vector into density and temperature
        N = y[:element.z + 1]
        kbT = y[element.z + 1:]
        # E = N * kbT # Not currently needed

        # Compensation
        comp = np.sum(q*N)/Ne
        mean_ion_heat = 1/2 * device.current/(4 * PI * EPS_0 * ve) * (1-comp)
        v_trap_rad = -1 * on_ax_sc * (1-comp)
        v_trap_ax = device.v_ax + comp * on_ax_sc

        # precompute collision rates
        rij = plasma.ion_coll_rate_mat(N, N, kbT, kbT, A, A)
        ri = np.sum(rij, axis=1)

        ### Particle density rates
        R_ei = je * N * eixs
        R_rr = je * N * rrxs
        R_dr = je * N * drxs

        if target.cx:
            R_cx = N * N[0] * np.sqrt(8 * Q_E * np.clip(kbT, 0, None)/(PI * A * M_P)) * cxxs
            R_cx[1] = R_cx[1] - np.sum(R_cx[2:])
        else:
            R_cx = np.zeros_like(R_ei)
        R_ax = plasma.escape_rate_axial(N, kbT, ri, v_trap_ax)
        R_ra = plasma.escape_rate_radial(N, kbT, ri, A, v_trap_rad, device.b_ax, device.r_dt)

        ### Energy density rates
        S_ei = R_ei * kbT
        S_rr = R_rr * kbT
        S_dr = R_dr * kbT
        S_cx = R_cx * kbT
        S_ax = R_ax * (kbT + q * v_trap_ax)
        S_ra = R_ra * (kbT + q * (v_trap_rad + device.r_dt * device.b_ax * \
                                  np.sqrt(2 * Q_E * np.clip(kbT, 0, None) / (3 * A *M_P))))
        # Electron heating
        S_eh = plasma.electron_heating_vec(N, Ne, kbT, device.e_kin, A)
        # Energy transfer between charge states within same "species"
        S_tr = plasma.energy_transfer_vec(N, N, kbT, kbT, A, A, rij)
        # Ionisation heating
        S_ih = np.zeros_like(S_eh)
        S_ih[1:] = R_ei[:-1] * mean_ion_heat
        S_ih[:-1] -= (R_rr + R_dr + R_cx)[1:] * mean_ion_heat


        ### Construct rhs for N (density)
        R_tot = -(R_ei + R_rr + R_dr + R_cx) - (R_ax + R_ra)
        R_tot[1:] += R_ei[:-1]
        R_tot[:-1] += R_rr[1:] + R_dr[1:] + R_cx[1:]

        ### Construct rates for energy density flow
        S_tot = -(S_ei + S_rr + S_dr + S_cx) + S_eh + S_tr - (S_ax + S_ra)
        S_tot[1:] += S_ei[:-1]
        S_tot[:-1] += S_rr[1:] + S_dr[1:] + S_cx[1:]

        if target.cni:
            R_tot[0] = 0
            S_tot[0] = 0

        ### Deduce temperature flow -> Integrating temperature instead of energy has proven more
        ### numerically stable
        Q_tot = (S_tot - kbT * R_tot) / N
        Q_tot[N <= MINIMAL_DENSITY] = 0 ### Freeze temperature if density is low (-> not meaningful)

        if isinstance(rates, dict):
            rates["R_ei"] = R_ei
            rates["R_rr"] = R_rr
            rates["R_dr"] = R_dr
            rates["R_cx"] = R_cx
            rates["R_ax"] = R_ax
            rates["R_ra"] = R_ra
            rates["S_ei"] = S_ei
            rates["S_rr"] = S_rr
            rates["S_dr"] = S_dr
            rates["S_cx"] = S_cx
            rates["S_ax"] = S_ax
            rates["S_ra"] = S_ra
            rates["S_eh"] = S_eh
            rates["S_tr"] = S_tr
            rates["S_ih"] = S_ih
            rates["V_ii"] = np.diag(rij)
            rates["V_it"] = ri
            rates["Comp"] = comp

        return np.concatenate((R_tot, Q_tot))

    res = scipy.integrate.solve_ivp(rhs, (0, t_max), N_kbT_initial, **solver_kwargs)

    # Recompute rates for final solution (this cannot be done parasitically due to
    # the solver approximating the jacobian and calling rhs with bogus values).
    nt = res.t.size
    poll = dict()
    _ = rhs(res.t[0], res.y[:, 0], rates=poll)
    rates = {k:np.zeros((poll[k].size, nt)) for k in poll}
    for idx in range(nt):
        _ = rhs(res.t[idx], res.y[:, idx], rates=poll)
        for key, val in poll.items():
            rates[key][:, idx] = val

    return Result(
        param=param,
        t=res.t,
        N=res.y[:element.z + 1, :],
        kbT=res.y[element.z + 1:, :],
        res=res,
        rates=rates
        )


def energy_scan(sim_func, sim_kwargs, energies, parallel=False):
    """
    This function provides a convenient way to repeat the same simulation for a number of different
    electron beam energies. This can reveal variations in the charge state balance due to
    weakly energy dependent ionisation cross sections or
    even resonant phenomena like dielectronic recombination.

    Parameters
    ----------
    sim_func : callable
        The function handle for the simulation e.g. ebisim.simulation.basic_simulation.
    sim_kwargs : dict
        A dictionary containing all the required and optional parameters of the simulations
        (except for the kinetic electron energy) as key value pairs.
        This is unpacked in the function call.
    energies : list or numpy.array
        A list or array of the energies at which the simulation should be performed.
    parallel : bool, optional
        Determine whether multiple simulations should be run in parallel using pythons
        multiprocessing.pool. This may accelerate the scan when performing a large number of
        simulations.
        By default False.

    Returns
    -------
    ebisim.simulation.EnergyScanResult
        An object providing convenient access to the generated scan data.
    """
    sim_kwargs = sim_kwargs.copy()

    if "e_kin" in sim_kwargs:
        del sim_kwargs["e_kin"]
        warn(f"sim_kwargs contains a value for e_kin, this item will be ignored.")

    sim_kwargs.setdefault("solver_kwargs", {}) # cast element to Element if necessary
    sim_kwargs["solver_kwargs"]["dense_output"] = True # need dense output for interpolation

    # cast element to Element if necessary
    if not isinstance(sim_kwargs["element"], elements.Element):
        sim_kwargs["element"] = elements.get_element(sim_kwargs["element"])

    energies = np.array(energies)
    energies.sort()

    proc = _EScanProcessor(sim_func, sim_kwargs)

    if parallel:
        with Pool() as pool:
            results = pool.map(proc, energies)
    else:
        results = list(map(proc, energies))

    return EnergyScanResult(sim_kwargs, energies, results)


class _EScanProcessor:
    """A simple helper class to set default arguments when calling functions with pool.map"""

    def __init__(self, sim_func, sim_kwargs):
        """Create the container by supplying the function handle"""
        self.sim_func = sim_func
        self.sim_kwargs = sim_kwargs
        if "e_kin" in sim_kwargs:
            del sim_kwargs["e_kin"]

    def __call__(self, e_kin):
        """Call the undelying function injecting the correct energy and returning a
        dictionary containing the time, density and temperature fields"""
        return self.sim_func(e_kin=e_kin, **self.sim_kwargs)


class EnergyScanResult:
    """
    This class provides a convenient interface to access and evaluate the
    the results of the ebisim.simulation.energy_scan function.
    Abundances at arbitrary times are provided by performing linear interpolations of the solutions
    of the rate equation.

    Parameters
    ----------
    sim_kwargs : dict
        The sim_kwargs dictionary as provided during the call to ebisim.simulation.energy_scan.
    energies : numpy.array
        <eV>
        A sorted array containing the energies at which the energy scan has been evaluated.
    results : list of ebisim.simulation.Result
        A list of Result objects holding the results of each individual simulation
    """
    def __init__(self, sim_kwargs, energies, results):
        self._sim_kwargs = sim_kwargs
        self._t_max = self._sim_kwargs["t_max"]
        self._element = self._sim_kwargs["element"]
        self._energies = energies
        self._results = results


    def get_result(self, e_kin):
        """
        Returns the result object corresponding to the simulation at a given energy

        Parameters
        ----------
        e_kin : float
            <eV>
            The energy of the simulation one wishes to retrieve.

        Returns
        -------
        ebisim.simulation.Result
            The Result object for the polled scan step.

        Raises
        ------
        ValueError
            If the polled energy is not available.
        """
        if e_kin in self._energies:
            return self._results[self._energies.index(e_kin)]
        else:
            raise ValueError(f"e_kin = {e_kin} eV has not been simulated during this energy scan.")


    def abundance_at_time(self, t):
        """
        Provides information about the charge state distribution at a given time for all energies.

        Parameters
        ----------
        t : float
            <s>
            Point of time to evaluate.

        Returns
        -------
        energies : numpy.array
            The evaluated energies.
        abundance : numpy.array
            Contains the abundance of each charge state (rows) for each energy (columns).

        Raises
        ------
        ValueError
            If 't' is not part of the simulated time domain
        """
        if t < 0 or t > self._t_max:
            raise ValueError("This time has not been simulated during the energyscan.")
        per_energy = [res.abundance_at_time(t) for res in self._results]
        return self._energies.copy(), np.column_stack(per_energy)


    def abundance_of_cs(self, cs):
        """
        Provides information about the abundance of a single charge states at all simulated times
        and energies.

        Parameters
        ----------
        cs : int
            The charge state to evaluate.

        Returns
        -------
        energies : numpy.array
            The evaluated energies.
        times : numpy.array
            The evaluated timesteps.
        abundance : numpy.array
            Abundance of charge state 'cs' at given times (rows) and energies (columns).

        Raises
        ------
        ValueError
            If 'cs' is not a sensible charge state for the performed simulation.
        """
        if cs > self._element.z:
            raise ValueError("This charge state is not available for the given element.")
        times = np.logspace(-4, np.log10(self._t_max), 500)
        times = np.clip(times, a_min=0, a_max=self._t_max)
        per_time = [self.abundance_at_time(t)[1][cs, :] for t in times]
        return self._energies.copy(), times, np.row_stack(per_time)


    def plot_abundance_at_time(self, t, cs=None, **kwargs):
        """
        Produces a plot of the charge state abundance for different energies at a given time.

        Parameters
        ----------
        t : float
            <s>
            Point of time to evaluate.
        cs : list or None, optional
            If None, all charge states are plotted. By supplying a list of int it
            is possible to filter the charge states that should be plotted.
            By default None.
        **kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_energy_scan,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.Figure
            Figure handle of the generated plot.
        """
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
        """
        Produces a 2D contour plot of the charge state abundance for all simulated
        energies and times.

        Parameters
        ----------
        cs : int
            The charge state to plot.
        **kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_energy_scan,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.Figure
            Figure handle of the generated plot.
        """
        energies, times, abundance = self.abundance_of_cs(cs)
        kwargs.setdefault("title",
                          f"Abundance of {self._element.latex_isotope()}$^{{{cs}+}}$")
        return plotting.plot_energy_time_scan(
            energies=energies,
            times=times,
            abundance=abundance,
            **kwargs
        )
