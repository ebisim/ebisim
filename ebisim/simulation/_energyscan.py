"""
This module contains the energy scan method and related resources.
"""

from warnings import warn
from multiprocessing.pool import Pool
import numpy as np

from .. import plotting
from ..elements import Element

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
        warn("sim_kwargs contains a value for e_kin, this item will be ignored.")

    sim_kwargs.setdefault("solver_kwargs", {})  # cast element to Element if necessary
    sim_kwargs["solver_kwargs"]["dense_output"] = True  # need dense output for interpolation

    # cast element to Element if necessary
    sim_kwargs["element"] = Element.as_element(sim_kwargs["element"])

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
