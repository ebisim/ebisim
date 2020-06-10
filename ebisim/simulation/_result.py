"""
This module contains the simulation result class.
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

from .. import plotting
from ..elements import Element
from ._radial_dist import boltzmann_radial_potential_linear_density_ebeam
from ..physconst import MINIMAL_DENSITY, MINIMAL_KBT


_RATE_NAMES = dict(
    R_ei="Electron ionisation",
    R_rr="Radiative recombination",
    R_dr="Dielectronic recombination",
    R_cx="Charge exchange",
    R_ax="Axial losses",
    R_ra="Radial losses",
    Q_ei="Electron ionisation",
    Q_rr="Radiative recombination",
    Q_dr="Dielectronic recombination",
    Q_cx="Charge exchange",
    Q_ax="Axial losses",
    Q_ra="Radial losses",
    Q_eh="Electron heating",
    Q_tr="Heat transfer",
    Q_ih="Ionisation heating",
    V_ii="Self collision rate",
    V_it="Total collision rate",
    f_ei="Overlap factors",
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
    t : numpy.ndarray, optional
        An array holding the time step coordinates, by default None.
    N : numpy.ndarray, optional
        An array holding the occupancy of each charge state at a given time, by default None.
    kbT : numpy.ndarray, optional
        An array holding the temperature of each charge state at a given time, by default None.
    res : optional
        The result object returned by scipy.integrate.solve_ivp. This can contain useful
        information about the solver performance etc. Refer to scipy documentation for details,
        by default None.
    rates : dict, optional
        A dictionary containing the different breeding rates in arrays shaped like N,
        by default None.
    target : ebisim.simulation.Target, optional
        If coming from advanced_simulation, this is the target represented by this Result.
    device : ebisim.simulation.Device, optional
        If coming from advanced_simulation, this is the machine / electron beam description.

    """

    def __init__(self, param=None, t=None, N=None, kbT=None, res=None, rates=None,
                 target=None, device=None, model=None, id_=None,
            ):
        self.param = param if param is not None else {}
        self.t = t
        self.N = N
        self.kbT = kbT
        self.res = res
        self.rates = rates
        self.target = target
        self.device = device
        self.model = model
        self.id = id_


    def times_of_highest_abundance(self):
        """
        Yields the point of time with the highest abundance for each charge state

        Returns
        -------
        numpy.ndarray
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
        numpy.ndarray
            Abundance of each charge state, array index corresponds to charge state.

        """
        if self.res and self.res.sol:
            return self.res.sol(t)
        interp = scipy.interpolate.interp1d(self.t, self.N)
        return interp(t)

    def radial_distribution_at_time(self, t):
        # TODO: docstring
        if self.res and self.res.sol:
            y = self.res.sol(t)
        else:
            # interp = scipy.interpolate.interp1d(self.t, self.res.y)
            # y = interp(t)
            y = self.res.y[np.argmin((t-self.res.t)**2)]
        n = y[:self.model.ub[-1]]
        n = np.maximum(n, MINIMAL_DENSITY)
        kT = y[self.model.ub[-1]:]
        kT = np.maximum(kT, MINIMAL_KBT)
        phi, n3d, shapes = boltzmann_radial_potential_linear_density_ebeam(
                self.device.rad_grid, self.device.current, self.device.r_e, self.device.e_kin,
                np.atleast_2d(n).T, np.atleast_2d(kT).T, np.atleast_2d(self.model._q).T,
                first_guess=self.device.rad_phi_uncomp,
                ldu=(self.device.rad_fd_l, self.device.rad_fd_d, self.device.rad_fd_u)
            )
        return phi, n3d, shapes


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
        if self.target:
            element = Element.as_element(self.target)
        elif "element" in self.param:
            element = self.param["element"]
        if element is not None:
            title = f"{element.latex_isotope()} {stub.lower()}"
        else:
            title = stub

        if self.device:
            e_kin = self.device.e_kin
            j = self.device.j
        elif "e_kin" in self.param and "j" in self.param:
            e_kin = self.param["e_kin"]
            j = self.param["j"]
        if e_kin is not None and j is not None:
            title = title + f" ($j = {j:0.1f}$ A/cm$^2$, $E_{{e}} = {e_kin:0.1f}$ eV)"
        if self.param.get("dr_fwhm", None) is not None:
            title = title[:-1] + f", FWHM = {self.param['dr_fwhm']:0.1f} eV)"
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
            kwargs.setdefault("ylabel", "Linear density (m$^{-1}$)")

        if relative:
            kwargs["ylim"] = (0, 1.1)
            fig = plotting.plot_generic_evolution(self.t, self.N/np.sum(self.N, axis=0), **kwargs)
        else:
            fig = plotting.plot_generic_evolution(self.t, self.N, **kwargs)
        return fig


    plot = plot_charge_states #: Alias for plot_charge_states


    def plot_radial_distribution_at_time(self, t, **kwargs):
        # TODO: docstring
        phi, n3d, shapes = self.radial_distribution_at_time(t)
        dens = (n3d * shapes)[self.model.lb[self.id]:self.model.ub[self.id]]
        denslim = 10**np.ceil(np.log10(dens.max()))

        title = self._param_title(f"Radial distribution at t = {1000*t:.1f} ms")

        kwargs.setdefault("xscale", "log")
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("title", title)
        kwargs.setdefault("xlim", (self.device.r_e/100, self.device.r_dt))
        kwargs.setdefault("ylim", (denslim/10**10, denslim))

        fig = plotting.plot_radial_distribution(self.device.rad_grid, dens, phi, self.device.r_e, **kwargs)

        return fig


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
        kwargs.setdefault("ylabel", "Linear energy density (eV / m$^{-1}$)")
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
            R_ei, R_rr, R_dr, R_cx, R_ax, R_ra, Q_ei, Q_rr, Q_dr, Q_cx, Q_ax, Q_ra, Q_eh, Q_tr.

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
            kwargs.setdefault("ylabel", "Number density flow (m$^{-1}$ s$^{-1}$)")
        if rate_key.startswith("Q"):
            kwargs.setdefault("ylabel", "Energy density flow (eV m$^{-1}$ s$^{-1}$)")
        if rate_key == "f_ei":
            kwargs.setdefault("ylabel", "Ion electron overlap factor")

        fig = plotting.plot_generic_evolution(self.t, rate, **kwargs)
        return fig
