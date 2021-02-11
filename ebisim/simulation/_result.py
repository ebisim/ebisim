"""
This module contains the simulation result class.
"""
from __future__ import annotations
from enum import IntEnum
from typing import Generic, Optional, Dict, Any, Tuple, TypeVar, Callable
import numpy as np
import scipy.integrate
import scipy.interpolate
from matplotlib.pyplot import Figure

from .. import plotting
from .. elements import Element
from ._advanced_helpers import Device, AdvancedModel
from ._basic_helpers import BasicDevice
from ._radial_dist import boltzmann_radial_potential_linear_density_ebeam
from ..physconst import MINIMAL_N_1D, MINIMAL_KBT


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

    IONISATION_HEAT = 303

    # CV = 304

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
_RATE_LABELS: Dict[Rate, Dict[str, Any]] = {
    Rate.EI: dict(
        title="EI",
        ylabel="Ionisation rate " + _PER_M_PER_S,
    ),
    Rate.RR: dict(
        title="RR",
        ylabel="Recombination rate " + _PER_M_PER_S,
    ),
    Rate.DR: dict(
        title="DR",
        ylabel="Recombination rate " + _PER_M_PER_S,
    ),
    Rate.CX: dict(
        title="CX",
        ylabel="Recombination rate " + _PER_M_PER_S,
    ),
    Rate.AX_CO: dict(
        title="Ax. coll. losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    Rate.AX_RT: dict(
        title="Ax. roundtrip losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    Rate.RA_CO: dict(
        title="Rad. coll. losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    Rate.RA_RT: dict(
        title="Rad. roundtrip losses",
        ylabel="Loss rate " + _PER_M_PER_S,
    ),
    # Rate.T_EI: dict(
    #     title="Electron ionisation"
    #     ylabel="",
    # ),
    # Rate.T_RR: dict(
    #     title="Radiative recombination"
    #     ylabel="",
    # ),
    # Rate.T_DR: dict(
    #     title="Dielectronic recombination"
    #     ylabel="",
    # ),
    # Rate.T_CX: dict(
    #     title="Charge exchange"
    #     ylabel="",
    # ),
    Rate.T_AX_CO: dict(
        title="Ax. coll. losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.T_AX_RT: dict(
        title="Ax. roundtrip losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.T_RA_CO: dict(
        title="Rad. coll. losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.T_RA_RT: dict(
        title="Rad. roundtrip losses",
        ylabel="Cooling rate " + _EV_PER_S,
    ),
    Rate.COLLISIONAL_THERMALISATION: dict(
        title="Coll. thermalisation",
        ylabel="Thermalisation rate " + _EV_PER_S,
    ),
    Rate.SPITZER_HEATING: dict(
        title="Spitzer heating",
        ylabel="Heating rate " + _EV_PER_S,
    ),
    Rate.IONISATION_HEAT: dict(
        title="Ionisation heat",
        ylabel="Ionisation heat (eV)",
    ),
    # Rate.CV: dict(
    #     title="Heat capacity",
    #     ylabel="$c_V$ (eV/eV)",
    # ),
    Rate.COLLISION_RATE_TOTAL: dict(
        title="Total collision rate",
        ylabel=r"$r_i$ " + _PER_S,
    ),
    Rate.COLLISION_RATE_SELF: dict(
        title="Self collision rate",
        ylabel=r"$r_{ii}$ " + _PER_S,
    ),
    Rate.F_EI: dict(
        title="Electron beam overlap",
        ylabel=r"$f_{ei}$",
        ylim=(0, 1.1),
    ),
    Rate.TRAPPING_PARAMETER_AXIAL: dict(
        title="Ax. trapping parameter",
        ylabel=r"$\omega_{ax}$",
    ),
    Rate.TRAPPING_PARAMETER_RADIAL: dict(
        title="Rad. trapping parameter",
        ylabel=r"$\omega_{rad}$",
    ),
    Rate.TRAP_DEPTH_AXIAL: dict(
        title="Ax. trapping potential",
        ylabel=r"$V_{ax} (V)$",
    ),
    Rate.TRAP_DEPTH_RADIAL: dict(
        title="Rad. trapping potential",
        ylabel=r"$V_{rad} (V)$",
    ),
    Rate.E_KIN_MEAN: dict(
        title="Beam energy mean",
        ylabel=r"$E_e$ (eV)",
    ),
    Rate.E_KIN_FWHM: dict(
        title="Beam energy FWHM",
        ylabel=r"FWHM($E_e$) (eV)",
    ),
    Rate.E_KIN_FWHM: dict(
        title="Beam energy FWHM",
        ylabel=r"FWHM($E_e$) (eV)",
    ),
}


GenericDevice = TypeVar("GenericDevice", Device, BasicDevice)  # Needed for parametric polymorphism of Result classes


class BasicResult(Generic[GenericDevice]):
    """
    Instances of this class are containers for the results of ebisim basic_simulations and contain a
    variety of convenience methods for simple plot generation etc.
    """
    _abundance_label = "Abundance"

    def __init__(self, *, t: np.ndarray, N: np.ndarray, device: GenericDevice,
                 target: Element, res: Optional[Any] = None):
        """
        Parameters
        ----------
        t :
            An array holding the time step coordinates.
        N :
            An array holding the occupancy of each charge state at a given time.
        device :
            If coming from advanced_simulation, this is the machine / electron beam description.
        target :
            If coming from advanced_simulation, this is the target represented by this Result.
        res :
            The result object returned by scipy.integrate.solve_ivp. This can contain useful
            information about the solver performance etc. Refer to scipy documentation for details.
        """
        self.t = t
        self.N = N
        self.device: GenericDevice = device
        self.target = target
        self.res = res

    @property
    def _dense_interpolator(self) -> Optional[Callable[[float], np.ndarray]]:
        if self.res and self.res.sol:
            return self.res.sol
        else:
            return None

    @property
    def _abundance_interpolator(self) -> Callable[[float], np.ndarray]:
        return self._dense_interpolator or scipy.interpolate.interp1d(self.t, self.N)

    def _check_time_in_domain(self, t: float) -> None:
        if t < self.t.min() or t > self.t.max():
            raise ValueError("Value for t lies outside the simulated domain.")

    def _param_title(self, stub):
        """
        Generates a plot title by merging the stub with some general simulation parameters.
        Defaults to the stub if no parameters are available

        Parameters
        ----------
        stub :
            Title stub for the plot

        Returns
        -------
            A LaTeX formatted title string compiled from the stub, current density and fwhm.

        """
        title = f"{self.target.latex_isotope()} {stub.lower()}"  #: TODO: Fix case

        e_kin = self.device.e_kin
        j = self.device.j

        title = title + f" ($j = {j:0.1f}$ A/cm$^2$, $E_{{e}} = {e_kin:0.1f}$ eV)"

        fwhm = self.device.fwhm
        if fwhm is not None:
            title = title[:-1] + f", FWHM = {fwhm:0.1f} eV)"
        return title

    def times_of_highest_abundance(self) -> np.ndarray:
        """
        Yields the point of time with the highest abundance for each charge state

        Returns
        -------
            <s>
            Array of times.

        """
        args = np.argmax(self.N, axis=1)
        return self.t[args]

    def abundance_at_time(self, t: float) -> np.ndarray:
        """
        Yields the abundance of each charge state at a given time

        Parameters
        ----------
        t :
            <s>
            Point of time to evaluate.

        Returns
        -------
            Abundance of each charge state, array index corresponds to charge state.

        """
        self._check_time_in_domain(t)
        return self._abundance_interpolator(t)

    def plot_charge_states(self, relative: bool = False, **kwargs: Any) -> Figure:
        """
        Plot the charge state evolution of this result object.

        Parameters
        ----------
        relative :
            Flags whether the absolute numbers or a relative fraction should be plotted at each
            timestep, by default False.
        kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
            Figure handle of the generated plot.

        Raises
        ------
        ValueError
            If the required data (self.t, self.N) is not available, i.e.
            corresponding attributes of this Result instance have not been set correctly.

        """
        kwargs.setdefault("xlim", (1e-4, self.t.max()))
        kwargs.setdefault("ylim", (0, self.N.sum(axis=0).max()*1.05))
        kwargs.setdefault("title", self._param_title("Charge states"))
        kwargs.setdefault("yscale", "linear")
        kwargs.setdefault("plot_total", True)

        if relative:
            kwargs["ylim"] = (0, 1.1)
            kwargs.setdefault("ylabel", "Relative abundance")
            fig = plotting.plot_generic_evolution(self.t, self.N/np.sum(self.N, axis=0), **kwargs)
        else:
            kwargs.setdefault("ylabel", self._abundance_label)
            fig = plotting.plot_generic_evolution(self.t, self.N, **kwargs)
        return fig

    plot = plot_charge_states  #: Alias for plot_charge_states


class AdvancedResult(BasicResult[Device]):
    """
    Instances of this class are containers for the results of ebisim advanced_simulations and contain a
    variety of convenience methods for simple plot generation etc.
    """
    _abundance_label = "Linear density (m$^{-1}$)"

    def __init__(self, *, t: np.ndarray, N: np.ndarray, device: Device, target: Element,
                 kbT: np.ndarray, model: AdvancedModel, id_: int,
                 res: Optional[Any] = None, rates: Optional[Dict[Rate, np.ndarray]] = None):
        """
        Parameters
        ----------
        t :
            An array holding the time step coordinates.
        N :
            An array holding the occupancy of each charge state at a given time.
        device :
            If coming from advanced_simulation, this is the machine / electron beam description.
        target :
            If coming from advanced_simulation, this is the target represented by this Result.
        kbT :
            An array holding the temperature of each charge state at a given time.
        model :
            The AdvancedModel instance that was underlying the advanced_simulation
        id_ :
            The position of this target in the list of all simulated targets.
        res :
            The result object returned by scipy.integrate.solve_ivp. This can contain useful
            information about the solver performance etc. Refer to scipy documentation for details.
        rates :
            A dictionary containing the different breeding rates in arrays shaped like N.
        """
        super().__init__(t=t, N=N, device=device, target=target, res=res)
        self.kbT = kbT
        self.target = target
        self.device = device
        self.rates = rates
        model = model._replace(
            targets=list(model.targets),
            bg_gases=list(model.bg_gases),
            cxxs_bggas=list(model.cxxs_bggas),
            cxxs_trgts=list(model.cxxs_trgts)
        )
        self.model = model
        self.id = id_

    @property
    def _abundance_interpolator(self) -> Callable[[float], np.ndarray]:
        dinterp = self._dense_interpolator
        if dinterp is not None:
            lower = self.model.lb[self.id]
            upper = self.model.ub[self.id]
            interp = lambda t: dinterp(t)[lower:upper]  # noqa:E731
        else:
            interp = scipy.interpolate.interp1d(self.t, self.N)
        return interp

    @property
    def _temperature_interpolator(self) -> Callable[[float], np.ndarray]:
        dinterp = self._dense_interpolator
        if dinterp is not None and self.res is not None:
            lower = self.res.y.shape[0]//2 + self.model.lb[self.id]
            upper = self.res.y.shape[0]//2 + self.model.ub[self.id]
            interp = lambda t: dinterp(t)[lower:upper]  # noqa:E731
        else:
            interp = scipy.interpolate.interp1d(self.t, self.kbT)
        return interp

    def _density_filter(self, data, threshold):
        filtered = data.copy()
        filtered[self.N < threshold] = np.nan
        return filtered

    def temperature_at_time(self, t: float) -> np.ndarray:
        """
        Yields the temperature of each charge state at a given time

        Parameters
        ----------
        t :
            <s>
            Point of time to evaluate.

        Returns
        -------
            <eV>
            Temperature of each charge state, array index corresponds to charge state.

        """
        self._check_time_in_domain(t)
        return self._temperature_interpolator(t)

    def radial_distribution_at_time(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Yields the radial distribution information at time

        Parameters
        ----------
        t :
            <s>
            Point of time to evaluate.

        Returns
        -------
        phi :
            Radial potential
        n3d :
            On axis 3D density for each charge state.
        shapes :
            The Boltzmann shape factors for each charge state.
        """
        self._check_time_in_domain(t)

        interp = self._dense_interpolator
        if interp is None:
            if self.res is None:
                raise ValueError("Required Data not available, "
                                 + "something went wrong during the creation of this result.")
            interp = scipy.interpolate.interp1d(self.t, self.res.y)
        y = interp(t)

        n = y[:self.model.ub[-1]]
        n = np.maximum(n, MINIMAL_N_1D)
        kT = y[self.model.ub[-1]:]
        kT = np.maximum(kT, MINIMAL_KBT)
        phi, n3d, shapes = boltzmann_radial_potential_linear_density_ebeam(
            self.device.rad_grid, self.device.current, self.device.r_e, self.device.e_kin,
            np.atleast_2d(n).T, np.atleast_2d(kT).T, np.atleast_2d(self.model.q).T,
            first_guess=self.device.rad_phi_uncomp,
            ldu=(self.device.rad_fd_l, self.device.rad_fd_d, self.device.rad_fd_u)
        )
        return phi, n3d, shapes

    def plot_radial_distribution_at_time(self, t: float, **kwargs: Any) -> Figure:
        """
        Plot the radial ion distribution at time t.

        Parameters
        ----------
        t :
            <s>
            Point of time to evaluate.
        kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_radial_distribution and
            ebisim.plotting.plot_generic_evolution, cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
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

    def plot_energy_density(self, **kwargs: Any) -> Figure:
        """
        Plot the energy density evolution of this result object.

        Parameters
        ----------
        kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
            Figure handle of the generated plot.

        Raises
        ------
        ValueError
            If the required data (self.t, self.kbT) is not available, i.e.
            corresponding attributes of this Result instance have not been set correctly.

        """
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

    def plot_temperature(self, dens_threshold: float = 1000*MINIMAL_N_1D, **kwargs: Any) -> Figure:
        """
        Plot the temperature evolution of this result object.

        Parameters
        ----------
        dens_threshold :
            If given temperatures are only plotted where the particle denisty is larger than
            the threshold value.
        kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
            Figure handle of the generated plot

        Raises
        ------
        ValueError
            If the required data (self.t, self.kbT) is not available, i.e.
            corresponding attributes of this Result instance have not been set correctly.

        """
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

    def plot_rate(self, rate_key: Rate,
                  dens_threshold: float = 1000*MINIMAL_N_1D, **kwargs) -> Figure:
        """
        Plots the requested ionisation- or energy flow rates.

        Parameters
        ----------
        rate_key :
            The key identifying the rate to be plotted.
            See ebisim.simulation.Rate for valid values.
        dens_threshold :
            If given temperatures are only plotted where the particle denisty is larger than
            the threshold value.
        kwargs
            Keyword arguments are handed down to ebisim.plotting.plot_generic_evolution,
            cf. documentation thereof.
            If no arguments are provided, reasonable default values are injected.

        Returns
        -------
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
        if rate.shape[0] != 1:  # Don't filter if scalar (i.e. not for indiv charge states)
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
        if rate.shape[0] == 1:
            kwargs.setdefault("label_lines", False)
            kwargs.setdefault("ls", "-")

        fig = plotting.plot_generic_evolution(self.t, rate, **kwargs)
        return fig
