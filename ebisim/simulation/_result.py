"""
This module contains the simulation result class.
"""

from enum import IntEnum
import numpy as np
import scipy.integrate
import scipy.interpolate

from .. import plotting
from ..elements import Element
from ._radial_dist import boltzmann_radial_potential_linear_density_ebeam
from ..physconst import MINIMAL_DENSITY, MINIMAL_KBT

class Rate(IntEnum):
    """
    Enum for conveniently identifying rates produced in advanced simulations
    """
    ELECTRON_IONISATION = 101
    EI = 101

    RADIATIVE_RECOMBINATION = 102
    RR = 102

    DIELECTRONIC_RECOMBINATION = 103
    DR = 103

    CHARGE_EXCHANGE = 104
    CX = 104

    LOSSES_AXIAL_COLLISIONAL = 105
    AX_CO = 105

    LOSSES_AXIAL_ROUNDTRIP = 106
    AX_RT = 106

    LOSSES_RADIAL_COLLISIONAL = 107
    RA_CO = 107

    LOSSES_RADIAL_ROUNDTRIP = 108
    RA_RT = 108

    # T_ELECTRON_IONISATION = 201
    # T_EI = 201

    # T_RADIATIVE_RECOMBINATION = 202
    # T_RR = 202

    # T_DIELECTRONIC_RECOMBINATION = 203
    # T_DR = 203

    # T_CHARGE_EXCHANGE = 204
    # T_CX = 204

    T_LOSSES_AXIAL_COLLISIONAL = 205
    T_AX_CO = 205

    T_LOSSES_AXIAL_ROUNDTRIP = 206
    T_AX_RT = 206

    T_LOSSES_RADIAL_COLLISIONAL = 207
    T_RA_CO = 207

    T_LOSSES_RADIAL_ROUNDTRIP = 208
    T_RA_RT = 208

    T_COLLISIONAL_THERMALISATION = 301
    COLLISIONAL_THERMALISATION = 301
    T_CT = 301

    T_SPITZER_HEATING = 302
    SPITZER_HEATING = 302
    T_SH = 302

    COLLISION_RATE_TOTAL = 401

    COLLISION_RATE_SELF = 402

    OVERLAP_FACTORS_EBEAM = 501
    F_EI = 501

    # CHARGE_COMPENSATION = 511

    TRAPPING_PARAMETER_AXIAL = 521
    W_AX = 521

    TRAPPING_PARAMETER_RADIAL = 522
    W_RA = 522

    TRAP_DEPTH_AXIAL = 523
    V_AX = 523

    TRAP_DEPTH_RADIAL = 524
    V_RA = 524

    E_KIN_MEAN = 531

    E_KIN_FWHM = 532


_PER_M_PER_S = r"(m$^{-1}$ s$^{-1}$)"
_EV_PER_S = r"(eV s$^{-1}$)"
_PER_S = r"(s$^{-1}$)"
_RATE_LABELS = {
    Rate.EI:dict(
        title="EI",
        ylabel="Ionisation rate " + _PER_M_PER_S,
    ),
    Rate.RR:dict(
        title="RR",
        ylabel="Recombination rate " + _PER_M_PER_S,
    ),
    Rate.DR:dict(
        title="DR",
        ylabel="Recombination rate " + _PER_M_PER_S,
    ),
    Rate.CX:dict(
        title="CX",
        ylabel="Recombination rate " + _PER_M_PER_S,
    ),
    Rate.AX_CO:dict(
        title="Ax. coll. losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    Rate.AX_RT:dict(
        title="Ax. roundtrip losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    Rate.RA_CO:dict(
        title="Rad. coll. losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    Rate.RA_RT:dict(
        title="Rad. roundtrip losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    # Rate.T_EI:dict(
    #     title="Electron ionisation"
    #     ylabel="",
    # ),
    # Rate.T_RR:dict(
    #     title="Radiative recombination"
    #     ylabel="",
    # ),
    # Rate.T_DR:dict(
    #     title="Dielectronic recombination"
    #     ylabel="",
    # ),
    # Rate.T_CX:dict(
    #     title="Charge exchange"
    #     ylabel="",
    # ),
    Rate.T_AX_CO:dict(
        title="Ax. coll. losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.T_AX_RT:dict(
        title="Ax. roundtrip losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.T_RA_CO:dict(
        title="Rad. coll. losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.T_RA_RT:dict(
        title="Rad. roundtrip losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.COLLISIONAL_THERMALISATION:dict(
        title="Coll. thermalisation",
        ylabel="Thermalisation rate " + _EV_PER_S,
    ),
    Rate.SPITZER_HEATING:dict(
        title="Spitzer heating",
        ylabel="Heating rate " + _EV_PER_S,
    ),
    Rate.COLLISION_RATE_TOTAL:dict(
        title="Total collision rate",
        ylabel=r"$r_i$ " + _PER_S,
    ),
    Rate.COLLISION_RATE_SELF:dict(
        title="Self collision rate",
        ylabel=r"$r_{ii}$ " + _PER_S,
    ),
    Rate.F_EI:dict(
        title="Electron beam overlap",
        ylabel=r"$f_{ei}$",
        ylim=(0, 1),
    ),
    Rate.TRAPPING_PARAMETER_AXIAL:dict(
        title="Ax. trapping parameter",
        ylabel=r"$\omega_{ax}$",
    ),
    Rate.TRAPPING_PARAMETER_RADIAL:dict(
        title="Rad. trapping parameter",
        ylabel=r"$\omega_{rad}$",
    ),
    Rate.TRAP_DEPTH_AXIAL:dict(
        title="Ax. trapping potential",
        ylabel=r"$V_{ax} (V)$",
    ),
    Rate.TRAP_DEPTH_RADIAL:dict(
        title="Rad. trapping potential",
        ylabel=r"$V_{rad} (V)$",
    ),
    Rate.E_KIN_MEAN:dict(
        title="Beam energy mean",
        ylabel=r"$E_e$ (eV)",
    ),
    Rate.E_KIN_FWHM:dict(
        title="Beam energy FWHM",
        ylabel=r"FWHM($E_e$) (eV)",
    ),
    Rate.E_KIN_FWHM:dict(
        title="Beam energy FWHM",
        ylabel=r"FWHM($E_e$) (eV)",
    ),
}



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
            return self.res.sol(t)[self.model.lb[self.id]:self.model.ub[self.id]]
        interp = scipy.interpolate.interp1d(self.t, self.N)
        return interp(t)


    def temperature_at_time(self, t):
        """
        Yields the temperature of each charge state at a given time

        Parameters
        ----------
        t : float
            <s>
            Point of time to evaluate.

        Returns
        -------
        numpy.ndarray
            <eV>
            Temperature of each charge state, array index corresponds to charge state.

        """
        if self.res and self.res.sol:
            return self.res.sol(t)[
                self.model.lb[self.id]+self.res.y.shape[0]//2:
                self.model.ub[self.id]+self.res.y.shape[0]//2]
        interp = scipy.interpolate.interp1d(self.t, self.kbT)
        return interp(t)


    def radial_distribution_at_time(self, t):
        """
        Yields the radial distribution information at time

        Parameters
        ----------
        t : float
            <s>
            Point of time to evaluate.

        Returns
        -------
        phi : numpy.ndarray
            Radial potential
        n3d : numpy.ndarray
            On axis 3D density for each charge state.
        shapes : numpy.ndarray
            The Boltzmann shape factors for each charge state.
        """
        if self.res and self.res.sol:
            y = self.res.sol(t)
        else:
            # interp = scipy.interpolate.interp1d(self.t, self.res.y)
            # y = interp(t)
            y = self.res.y[:, np.argmin((t-self.res.t)**2)]
        n = y[:self.model.ub[-1]]
        n = np.maximum(n, MINIMAL_DENSITY)
        kT = y[self.model.ub[-1]:]
        kT = np.maximum(kT, MINIMAL_KBT)
        phi, n3d, shapes = boltzmann_radial_potential_linear_density_ebeam(
            self.device.rad_grid, self.device.current, self.device.r_e, self.device.e_kin,
            np.atleast_2d(n).T, np.atleast_2d(kT).T, np.atleast_2d(self.model.q).T,
            first_guess=self.device.rad_phi_uncomp,
            ldu=(self.device.rad_fd_l, self.device.rad_fd_d, self.device.rad_fd_u)
        )
        return phi, n3d, shapes

    def _density_filter(self, data, threshold):
        filtered = data.copy()
        filtered[self.N < threshold] = np.nan
        return filtered

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
        """
        Plot the radial ion distribution at time t.

        Parameters
        ----------
        t : float
            <s>
            Point of time to evaluate.
        **kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_radial_distribution and
            ebisim.plotting.plot_generic_evolution, cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
        matplotlib.Figure
            Figure handle of the generated plot.
        """
        phi, n3d, shapes = self.radial_distribution_at_time(t)
        dens = (n3d * shapes)[self.model.lb[self.id]:self.model.ub[self.id]]
        denslim = 10**np.ceil(np.log10(dens.max()))

        title = self._param_title(f"Radial distribution at t = {1000*t:.1f} ms")

        kwargs.setdefault("xscale", "log")
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("title", title)
        kwargs.setdefault("xlim", (self.device.r_e/100, self.device.r_dt))
        kwargs.setdefault("ylim", (denslim/10**10, denslim))

        fig = plotting.plot_radial_distribution(
            self.device.rad_grid, dens, phi, self.device.r_e, **kwargs
        )

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


    def plot_temperature(self, dens_threshold=1000*MINIMAL_DENSITY, **kwargs):
        """
        Plot the temperature evolution of this result object.

        Parameters
        ----------
        dens_threshold : float, optional
            If given temperatures are only plotted where the particle denisty is larger than
            the threshold value.
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

        filtered_kbT = self._density_filter(self.kbT, dens_threshold)

        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault(
            "ylim",
            (
                10**np.floor(np.nanmin(np.log10(filtered_kbT))),
                10**np.ceil(np.nanmax(np.log10(filtered_kbT))),
            )
        )
        kwargs.setdefault("title", self._param_title("Temperature"))
        kwargs.setdefault("ylabel", "Temperature (eV)")

        fig = plotting.plot_generic_evolution(self.t, filtered_kbT, **kwargs)
        return fig


    def plot_rate(self, rate_key, dens_threshold=1000*MINIMAL_DENSITY, **kwargs):
        """
        Plots the requested ionisation- or energy flow rates.

        Parameters
        ----------
        rate_key : ebisim.simulation.Rate
            The key identifying the rate to be plotted.
            See ebisim.simulation.Rate for valid values.
        dens_threshold : float, optional
            If given temperatures are only plotted where the particle denisty is larger than
            the threshold value.
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
            If the required data (self.rates) is not available, or an invalid key is requested.

        """
        if self.rates is None:
            raise ValueError("Rates are not available for this result.")
        if rate_key not in self.rates:
            raise ValueError(
                f"The requested rate_key does not exist. Available rates are {self.rates.keys()}."
            )

        rate = self.rates[rate_key]
        if rate.shape[0] != 1: #Don't filter if scalar (i.e. not for indiv charge states)
            rate = self._density_filter(rate, dens_threshold)
        labels = _RATE_LABELS.get(rate_key, {})

        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault("ylim", labels.get("ylim", None))
        kwargs.setdefault("yscale", labels.get("yscale", "linear"))
        kwargs.setdefault("title", self._param_title(labels.get("title", "unkown rate")))
        kwargs.setdefault("ylabel", labels.get("ylabel", "Unknown"))
        kwargs.setdefault("plot_total", False)

        if len(rate.shape) == 1:
            rate = rate[np.newaxis, :]

        fig = plotting.plot_generic_evolution(self.t, rate, **kwargs)
        return fig
