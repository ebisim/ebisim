"""
This module contains the advanced simulation method and related resources.
"""
from __future__ import annotations
import logging
from typing import NamedTuple, Any, Union, Optional, List
import numpy as np
import numba

from .. import xs
from .. import plasma
from ..utils import patch_namedtuple_docstrings
from ..elements import Element
from ..physconst import PI, EPS_0, K_B
from ._radial_dist import (
    fd_system_nonuniform_grid,
    boltzmann_radial_potential_linear_density_ebeam,
)

logger = logging.getLogger(__name__)


class BackgroundGas(NamedTuple):
    """
    Use the static `get()` factory methods to create instances of this class.

    Simple datacontainer for a background gas for advanced simulations.
    A background gas only acts as a charge exchange partner to the Targets in the simulation.

    See Also
    --------
    ebisim.simulation.BackgroundGas.get
    """

    name: str
    ip: float
    n0: float

    @classmethod
    def get(cls, element: Union[Element, str, int], p: float, T: float = 300.0) -> BackgroundGas:
        """
        Factory method for defining a background gas.

        Parameters
        ----------
        element :
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.
        p :
            <mbar>
            Gas pressure.
        T :
            <K>
            Gas temperature, by default 300 K (Room temperature)

        Returns
        -------
        ebisim.simulation.BackgroundGas
            Ready to use BackgroundGas specification.
        """
        element = Element.as_element(element)
        return cls(
            element.name,
            element.ip,
            (p * 100) / (K_B * T)  # Convert from mbar to Pa and compute density at Temp
        )


_BACKGROUNDGAS_DOC = dict(
    name="Name of the element.",
    ip="<eV> Ionisation potential of this Gas.",
    n0="<1/m^3> Gas number density.",
)
patch_namedtuple_docstrings(BackgroundGas, _BACKGROUNDGAS_DOC)


class Device(NamedTuple):
    """
    Use the static `get()` factory methods to create instances of this class.

    Objects of this class are used to pass important EBIS/T parameters into the simulation.
    """

    current: float
    e_kin: float
    r_e: float
    length: Optional[float]
    j: float
    v_ax: float
    v_ax_sc: float
    v_ra: float
    b_ax: float
    r_dt: float
    r_dt_bar: float
    fwhm: float
    rad_grid: np.ndarray
    rad_fd_l: np.ndarray
    rad_fd_d: np.ndarray
    rad_fd_u: np.ndarray
    rad_phi_uncomp: np.ndarray
    rad_phi_ax_barr: np.ndarray
    rad_re_idx: int

    @classmethod
    def get(cls, *,
            current: float, e_kin: float, r_e: float, v_ax: float, b_ax: float, r_dt: float,
            length: Optional[float] = None, v_ra: Optional[float] = None,
            j: Optional[float] = None, fwhm: Optional[float] = None,
            n_grid: int = 400, r_dt_bar: Optional[float] = None) -> Device:
        """
        Factory method for defining a device.
        All arguments are keyword only to reduce the chance of mixing them up.

        Parameters
        ----------
        current :
            <A>
            Electron beam current.
        e_kin :
            <eV>
            Uncorrected electron beam energy.
        r_e :
            <m>
            Electron beam radius.
        v_ax :
            <V>
            Axial barrier bias.
        b_ax :
            <T>
            Axial magnetic flux density in the trap.
        r_dt :
            <m>
            Drift tube radius.
        length :
            <m>
            Trap length -> Currently not used in simulations.
        v_ra :
            <V>
            Override for radial trap depth.
            Only effective if ModelOptions.RADIAL_DYNAMICS=False.
        j :
            <A/cm^2>
            Override for current density.
        fwhm :
            <eV>
            Override for the electron beam energy spread.
            Only effective if ModelOptions.RADIAL_DYNAMICS=False.
        n_grid :
            Approximate number of nodes for the radial mesh.
        r_dt_bar :
            Radius of the barrier drift tubes.
            If not passed, assuming equal radius as trap drift tube.

        Returns
        -------
        ebisim.simulation.Device
            The populated device object.
        """
        # Enforcing typing here seems to help with spooky segfaults,
        # which are probably caused by calling the Boltzmann Poission solver with varying types
        # At some point I should figure this out and fix or report the underlying problem
        current = float(current)
        e_kin = float(e_kin)
        r_e = float(r_e)
        v_ax = float(v_ax)
        b_ax = float(b_ax)
        r_dt = float(r_dt)
        n_grid = int(n_grid)

        logger.debug(f"Device.get({current}, {e_kin}, {r_e}, {length}, "
                     + f"{v_ax}, {b_ax}, {r_dt}, {v_ra}, {j}, {fwhm}, {n_grid})")
        rad_grid = np.concatenate((
            np.linspace(0, r_e, n_grid//6, endpoint=False),
            np.linspace(r_e, 2*r_e, n_grid//6, endpoint=False),
            np.geomspace(2*r_e, r_dt, n_grid//6*4)
        ))
        logger.debug("Device.get: Call fd_system_nonuniform_grid.")
        rad_ldu = fd_system_nonuniform_grid(rad_grid)
        rad_re_idx = int(np.argmin((rad_grid-r_e)**2))

        logger.debug("Device.get: Trap - Call boltzmann_radial_potential_linear_density_ebeam.")
        phi, _, __ = boltzmann_radial_potential_linear_density_ebeam(
            rad_grid, current, r_e, e_kin, 0, 1, 1, ldu=rad_ldu
        )
        rad_phi_uncomp = phi

        if r_dt_bar is None:
            logger.debug(
                "Device.get: Barrier - Call boltzmann_radial_potential_linear_density_ebeam."
            )
            phi_barrier, _, __ = boltzmann_radial_potential_linear_density_ebeam(
                rad_grid, current, r_e, e_kin+v_ax, 0, 1, 1, ldu=rad_ldu
            )
            v_ax_sc = phi_barrier[0]
        else:
            _rad_grid = np.concatenate((
                np.linspace(0, r_e, n_grid//6, endpoint=False),
                np.linspace(r_e, 2*r_e, n_grid//6, endpoint=False),
                np.geomspace(2*r_e, r_dt_bar, n_grid//6*4)
            ))
            logger.debug(
                "Device.get: Barrier - Call boltzmann_radial_potential_linear_density_ebeam."
            )
            phi_barrier, _, __ = boltzmann_radial_potential_linear_density_ebeam(
                _rad_grid, current, r_e, e_kin+v_ax, 0, 1, 1,
            )
            v_ax_sc = phi_barrier[0]

        if j is None:
            j = current/r_e**2/PI*1e-4
        if fwhm is None:
            fwhm = 1/2 * current/4/PI/EPS_0/plasma.electron_velocity(e_kin+phi.min())
        if v_ra is None:
            v_ra = -phi.min()

        logger.debug("Device.get: Create and return Device instance.")
        return cls(
            current=float(current),
            e_kin=float(e_kin),
            r_e=float(r_e),
            length=float(length) if length is not None else None,
            j=float(j),
            v_ax=float(v_ax),
            v_ax_sc=float(v_ax_sc),
            v_ra=float(v_ra),
            b_ax=float(b_ax),
            r_dt=float(r_dt),
            r_dt_bar=float(r_dt_bar if r_dt_bar else r_dt),
            fwhm=float(fwhm),
            rad_grid=rad_grid,
            rad_fd_l=rad_ldu[0],
            rad_fd_d=rad_ldu[1],
            rad_fd_u=rad_ldu[2],
            rad_phi_uncomp=rad_phi_uncomp,
            rad_phi_ax_barr=phi_barrier + v_ax,
            rad_re_idx=rad_re_idx
        )

    def __str__(self) -> str:
        return f"Device: I = {self.current:.2e} A, E = {self.e_kin:.2e} eV"


_DEVICE_DOC = dict(
    current="<A> Beam current.",
    e_kin="<eV> Uncorrected beam energy.",
    r_e="<m> Beam radius.",
    length="<m> Trap length.",
    j="<A/cm^2> Current density.",
    v_ax="<V> Axial barrier voltage.",
    v_ax_sc="<V> Axial barrier space charge correction.",
    v_ra="<V> Radial barrier voltage.",
    b_ax="<T> Axial magnetic field density.",
    r_dt="<m> Drift tube radius.",
    r_dt_bar="<m> Drift tube radius of the barrier drift tubes.",
    fwhm="<eV> Electron beam energy spread.",
    rad_grid="<m> Radial grid for finite difference computations.",
    rad_fd_l="Lower diagonal vector of finite difference scheme.",
    rad_fd_d="Diagonal vector of finite difference scheme.",
    rad_fd_u="Upper diagonal vector of finite difference scheme.",
    rad_phi_uncomp="<V> Radial potential of the electron beam.",
    rad_phi_ax_barr="<V> Radial potential of the electron beam in the barrier tube",
    rad_re_idx="Index of the radial grid point closest to r_e",
)
patch_namedtuple_docstrings(Device, _DEVICE_DOC)


class ModelOptions(NamedTuple):
    """
    An instance of ModelOptions can be used to turn on or off certain effects
    in an advanced simulation.
    """

    EI: bool = True
    RR: bool = True
    CX: bool = True
    DR: bool = False
    SPITZER_HEATING: bool = True
    COLLISIONAL_THERMALISATION: bool = True
    ESCAPE_AXIAL: bool = True
    ESCAPE_RADIAL: bool = True
    RECOMPUTE_CROSS_SECTIONS: bool = False
    RADIAL_DYNAMICS: bool = False
    IONISATION_HEATING: bool = True
    OVERRIDE_FWHM: bool = False


_MODELOPTIONS_DOC = dict(
    EI="Switch for electron impact ionisation, default True.",
    RR="Switch for radiative recombination, default True.",
    CX="Switch for charge exchange, default True.",
    DR="Switch for dielectronic recombination, default False.",
    SPITZER_HEATING="Switch for Spitzer- or electron-heating, default True.",
    COLLISIONAL_THERMALISATION="Switch for ion-ion thermalisation, default True.",
    ESCAPE_AXIAL="Switch for axial escape from the trap, default True.",
    ESCAPE_RADIAL="Switch for radial escape from the trap, default True.",
    RECOMPUTE_CROSS_SECTIONS="""\
Switch deciding whether EI, RR, and DR cross
sections are recomputed on each call of the differential equation system. Advisable if electron beam
energy changes over time and sharp transitions are expected, e.g. DR or ionisation thresholds for a
given shell. Default False.""",
    RADIAL_DYNAMICS="""\
Switch for effects of radial ion cloud extent.
May be computationally very intensive""",
    IONISATION_HEATING="Switch for ionisation heating/recombination cooling",
    OVERRIDE_FWHM="If set use FWHM from device definition instead of computed value.",
)
patch_namedtuple_docstrings(ModelOptions, _MODELOPTIONS_DOC)

DEFAULT_MODEL_OPTIONS = ModelOptions()  #: Default simulation options


class AdvancedModel(NamedTuple):
    """
    The advanced model class is the base for ebisim.simulation.advanced_simulation.
    It acts as a fast datacontainer for the underlying rhs function which represents the right hand
    side of the differential equation system.
    Since it is jitcompiled using numba, care is required during instatiation.

    Parameters
    ----------
    device : ebisim.simulation.Device
        Container describing the EBIS/T and specifically the electron beam.
    targets : numba.typed.List[ebisim.simulation.Target]
        List of ebisim.simulation.Target for which charge breeding is simulated.
    bg_gases : numba.typed.List[ebisim.simulation.BackgroundGas], optional
        List of ebisim.simulation.BackgroundGas which act as CX partners, by default None.
    options : ebisim.simulation.ModelOptions, optional
        Switches for effects considered in the simulation, see default values of
        ebisim.simulation.ModelOptions.
    """

    device: Device
    targets: Any  # _T_TARGET_LIST = numba.types.ListType(_T_TARGET)
    bg_gases: Any  # _T_BG_GAS_LIST = numba.types.ListType(_T_BG_GAS)
    options: ModelOptions
    lb: np.ndarray
    ub: np.ndarray
    nq: int
    q: np.ndarray
    a: np.ndarray
    eixs: np.ndarray
    rrxs: np.ndarray
    drxs: np.ndarray
    cxxs_bggas: Any  # numba.types.ListType(_T_F8_ARRAY)
    cxxs_trgts: Any  # numba.types.ListType(_T_F8_ARRAY)

    @classmethod
    def get(cls, device: Device, targets: List[Element],
            bg_gases: Optional[List[BackgroundGas]] = None,
            options: ModelOptions = DEFAULT_MODEL_OPTIONS) -> AdvancedModel:

        # Types needed by numba
        _T_BG_GAS = numba.typeof(BackgroundGas.get("He", 1e-8))
        _T_F8_ARRAY = numba.float64[:]

        bg_gases = bg_gases or []
        if not bg_gases:
            bg_gases_nblist = numba.typed.List.empty_list(_T_BG_GAS)
        else:
            bg_gases_nblist = numba.typed.List(bg_gases)

        # Determine array bounds for different targets in state vector
        lb = np.zeros(len(targets), dtype=np.int32)
        ub = np.zeros(len(targets), dtype=np.int32)
        offset = 0
        for i, trgt in enumerate(targets):
            lb[i] = offset
            ub[i] = lb[i] + trgt.z + 1
            offset = ub[i]

        # Compute total number of charge states for all targets (len of "n" or "kT" state vector)
        nq = int(ub[-1])

        # Define vectors listing the charge state and mass for each state
        q = np.zeros(nq, dtype=np.int32)
        a = np.zeros(nq, dtype=np.int32)
        for i, trgt in enumerate(targets):
            q[lb[i]:ub[i]] = np.arange(trgt.z + 1, dtype=np.int32)
            a[lb[i]:ub[i]] = np.full(trgt.z + 1, trgt.a, dtype=np.int32)

        # Initialise cross section vectors
        _eixs = np.zeros(nq)
        _rrxs = np.zeros(nq)
        _drxs = np.zeros(nq)
        for i, trgt in enumerate(targets):
            _eixs[lb[i]:ub[i]] = xs.eixs_vec(trgt, device.e_kin)
            _rrxs[lb[i]:ub[i]] = xs.rrxs_vec(trgt, device.e_kin)
            _drxs[lb[i]:ub[i]] = xs.drxs_vec(trgt, device.e_kin, device.fwhm)

        # Precompute CX cross sections (invariant)
        _cxxs_bggas = numba.typed.List.empty_list(_T_F8_ARRAY)
        _cxxs_trgts = numba.typed.List.empty_list(_T_F8_ARRAY)
        for gas in bg_gases:
            _cxxs_bggas.append(xs.cxxs(q, gas.ip))
        for trgt in targets:
            _cxxs_trgts.append(xs.cxxs(q, trgt.ip))
        return cls(
            device=device,
            targets=numba.typed.List(targets),
            bg_gases=bg_gases_nblist,
            options=options,
            lb=lb,
            ub=ub,
            nq=nq,
            q=q,
            a=a,
            eixs=_eixs,
            rrxs=_rrxs,
            drxs=_drxs,
            cxxs_bggas=_cxxs_bggas,
            cxxs_trgts=_cxxs_trgts
        )

    def __str__(self) -> str:
        return (f"AdvancedModel: {self.device!s}, {self.targets!s}, "
                + f"{self.bg_gases!s}, Options: [...]")
