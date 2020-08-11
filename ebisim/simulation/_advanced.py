"""
This module contains the advanced simulation method and related resources.
"""
import logging
from collections import namedtuple, OrderedDict
import numpy as np
import scipy.integrate
import scipy.interpolate
import numba
from joblib import Parallel, delayed

from .. import xs
from .. import plasma
from ..elements import Element
from ..physconst import Q_E, M_P, PI, EPS_0, K_B
from ..physconst import MINIMAL_N_1D, MINIMAL_KBT
from ._result import Result, Rate
from ._radial_dist import (
    fd_system_nonuniform_grid,
    boltzmann_radial_potential_linear_density_ebeam,
    # heat_capacity
)

logger = logging.getLogger(__name__)
#Hack for making Enums hashable by numba - hash will differ from CPython
#This is to make Enums work as numba.typed.Dict keys
logger.debug("Patching numba.types.EnumMember __hash__.")
@numba.extending.overload_method(numba.types.EnumMember, '__hash__')
def enum_hash(val): # pylint: disable=unused-argument
    def impl(val):
        return hash(val.value)
    return impl

logger.debug("Defining Target.")
class Target(namedtuple("Target", Element._fields + ("n", "kT", "cx"))):
    """
    Use the static `get_ions()` or `get_gas()` factory methods to create instances of this class.

    Targets are used in advanced simulations;
    they are an extended version of ebisim.elements.Element.

    There are four extra fields for a more convenient setup of advanced simulations:
    n and kT are vectors holding initial conditions.
    cx is a boolean flag determining whether the neutral particles contribute to charge exchange.

    See Also
    --------
    ebisim.simulation.Target.get_ions
    ebisim.simulation.Target.get_gas
    """
    __slots__ = ()

    @classmethod
    def get_gas(cls, element, p, r_dt, T=300.0, cx=True):
        """
        Factory method for defining a neutral gas injection Target.
        A gas target is a target with constant density in charge state 0.

        Parameters
        ----------
        element : ebisim.elements.Element or str or int
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.
        p : float
            <mbar>
            Gas pressure.
        r_dt : float
            <m>
            Drift tube radius, required to compute linear density.
        T : float, optional
            <K>
            Gas temperature, by default 300 K (Room temperature)
        cx : bool, optional
            see class documentation, by default True

        Returns
        -------
        ebisim.simulation.Target
            Ready to use Target specification.
        """
        element = Element.as_element(element)
        _n = np.full(element.z + 1, MINIMAL_N_1D, dtype=np.float64)
        _kT = np.full(element.z + 1, MINIMAL_KBT, dtype=np.float64)
        _n[0] = (p * 100) / (K_B * T) * PI * r_dt**2#Convert from mbar to Pa and compute density
        if _n[0] < MINIMAL_N_1D:
            raise ValueError("The resulting density is smaller than the internal minimal value.")
        _kT[0] = K_B * T / Q_E
        return cls(*element, n=_n, kT=_kT, cx=cx)

    @classmethod
    def get_ions(cls, element, nl, kT=10, q=1, cx=True):
        """
        Factory method for defining a pulsed ion injection Target.
        An ion target has a given density in the charge state of choice q.

        Parameters
        ----------
        element : ebisim.elements.Element or str or int
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.
        nl : float
            <1/m>
            Linear density of the initial charge state (ions per unit length).
        kT_per_q : float, optional
            <eV>
            Temperature / kinetic energy of the injected ions,
            by default 10 eV
        q : int, optional
            Initial charge state, by default 1
        cx : bool, optional
            see class documentation, by default True

        Returns
        -------
        ebisim.simulation.Target
            Ready to use Target specification.
        """
        if nl < MINIMAL_N_1D:
            raise ValueError("The density is smaller than the internal minimal value.")
        element = Element.as_element(element)
        _n = np.full(element.z + 1, MINIMAL_N_1D, dtype=np.float64)
        _kT = np.full(element.z + 1, MINIMAL_KBT, dtype=np.float64)
        _n[q] = nl
        _kT[q] = kT
        return cls(*element, n=_n, kT=_kT, cx=cx)

    def __repr__(self):
        return f"Target({Element.as_element(self)})"

#Patching in docstrings
logger.debug("Patching Target docstrings.")
for f in Element._fields:
    setattr(getattr(Target, f), "__doc__", getattr(getattr(Element, f), "__doc__"))
Target.n.__doc__ = """<1/m> Array holding the initial linear density of each charge state."""
Target.kT.__doc__ = """<eV> Array holding the initial temperature of each charge state."""
Target.cx.__doc__ = """Boolean flag determining whether neutral particles of this target are
considered as charge exchange partners."""


logger.debug("Defining BackgroundGas.")
class BackgroundGas(namedtuple("BackgroundGas", "name, ip, n0")):
    """
    Use the static `get()` factory methods to create instances of this class.

    Simple datacontainer for a background gas for advanced simulations.
    A background gas only acts as a charge exchange partner to the Targets in the simulation.

    See Also
    --------
    ebisim.simulation.BackgroundGas.get
    """
    __slots__ = ()

    @classmethod
    def get(cls, element, p, T=300.0):
        """
        Factory method for defining a background gas.

        Parameters
        ----------
        element : ebisim.elements.Element or str or int
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.
        p : float
            <mbar>
            Gas pressure.
        T : float, optional
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
            (p * 100) / (K_B * T) #Convert from mbar to Pa and compute density at Temp
        )

#Patching in docstrings
logger.debug("Patching BackgroundGas docstrings.")
BackgroundGas.name.__doc__ = """str Name of the element."""
BackgroundGas.ip.__doc__ = """float <eV> Ionisation potential of this Gas."""
BackgroundGas.n0.__doc__ = """float <1/m^3> Gas number density."""

logger.debug("Defining Device.")
_DEVICE = OrderedDict(
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
class Device(namedtuple("Device", _DEVICE.keys())):
    """
    Use the static `get()` factory methods to create instances of this class.

    Objects of this class are used to pass important EBIS/T parameters into the simulation.
    """
    __slots__ = ()

    @classmethod
    def get(
            cls, current, e_kin, r_e, length, v_ax, b_ax, r_dt,
            v_ra=None, j=None, fwhm=None, n_grid=200, r_dt_bar=None
        ):
        """
        Factory method for defining a device.

        Parameters
        ----------
        current : float
            <A>
            Electron beam current.
        e_kin : float
            <eV>
            Uncorrected electron beam energy.
        r_e : float
            <m>
            Electron beam radius.
        length : float
            <m>
            Trap length.
        v_ax : float
            <V>
            Axial barrier bias.
        b_ax : float
            <T>
            Axial magnetic flux density in the trap.
        r_dt : float
            <m>
            Drift tube radius.
        v_ra : float, optional
            <V>
            Override for radial trap depth, by default None.
            Only effective if ModelOptions.RADIAL_DYNAMICS=False.
        j : float, optional
            <A/cm^2>
            Override for current density, by default None.
        fwhm : float, optional
            <eV>
            Override for the electron beam energy spread, by default None.
            Only effective if ModelOptions.RADIAL_DYNAMICS=False.
        n_grid : int, optional
            Approximate number of nodes for the radial mesh, by default 200.
        r_dt_bar : float, optional
            Radius of the barrier drift tubes.
            If not passed, assuming equal radius as trap drift tube.

        Returns
        -------
        ebisim.simulation.Device
            The populated device object.
        """
        logger.debug(f"Device.get({current}, {e_kin}, {r_e}, {length}, "\
                     f"{v_ax}, {b_ax}, {r_dt}, {v_ra}, {j}, {fwhm}, {n_grid})")
        rad_grid = np.concatenate((
            np.linspace(0, r_e, n_grid//6, endpoint=False),
            np.linspace(r_e, 2*r_e, n_grid//6, endpoint=False),
            np.geomspace(2*r_e, r_dt, n_grid//6*4)
        ))
        logger.debug("Device.get: Call fd_system_nonuniform_grid.")
        rad_ldu = fd_system_nonuniform_grid(rad_grid)
        rad_re_idx = np.argmin((rad_grid-r_e)**2)

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
            length=float(length),
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

    def __repr__(self):
        return f"Device.get({self.current}, {self.e_kin}, {self.r_e}, {self.length}, "\
                f"{self.v_ax}, {self.b_ax}, {self.r_dt}, {self.v_ra}, {self.j}, {self.fwhm}, "\
                f"{len(self.rad_grid)})"

logger.debug("Patching Device docstrings.")
for _k, _v in _DEVICE.items():
    setattr(getattr(Device, _k), "__doc__", _v)


logger.debug("Defining ModelOptions.")
_MODEL_OPTIONS_DEFAULTS = OrderedDict(
    EI=True, RR=True, CX=True, DR=False,
    SPITZER_HEATING=True, COLLISIONAL_THERMALISATION=True,
    ESCAPE_AXIAL=True, ESCAPE_RADIAL=True,
    RECOMPUTE_CROSS_SECTIONS=False, RADIAL_DYNAMICS=False, IONISATION_HEATING=True,
    OVERRIDE_FWHM=False
)
ModelOptions = namedtuple(
    "ModelOptions", _MODEL_OPTIONS_DEFAULTS.keys(), defaults=_MODEL_OPTIONS_DEFAULTS.values()
)
DEFAULT_MODEL_OPTIONS = ModelOptions()
#Patching in docstrings
logger.debug("Patching ModelOptions docstrings.")
ModelOptions.__doc__ = """An instance of ModelOptions can be used to turn on or off certain effects
in an advanced simulation."""
ModelOptions.EI.__doc__ = "Switch for electron impact ionisation, default True."
ModelOptions.RR.__doc__ = "Switch for radiative recombination, default True."
ModelOptions.CX.__doc__ = "Switch for charge exchange, default True."
ModelOptions.DR.__doc__ = "Switch for dielectronic recombination, default False."
ModelOptions.SPITZER_HEATING.__doc__ = "Switch for Spitzer- or electron-heating, default True."
ModelOptions.COLLISIONAL_THERMALISATION.__doc__ = "Switch for ion-ion thermalisation, default True."
ModelOptions.ESCAPE_AXIAL.__doc__ = "Switch for axial escape from the trap, default True."
ModelOptions.ESCAPE_RADIAL.__doc__ = "Switch for radial escape from the trap, default True."
ModelOptions.RECOMPUTE_CROSS_SECTIONS.__doc__ = """Switch deciding whether EI, RR, and DR cross
sections are recomputed on each call of the differential equation system. Advisable if electron beam
energy changes over time and sharp transitions are expected, e.g. DR or ionisation thresholds for a
given shell. Default False."""
ModelOptions.RADIAL_DYNAMICS.__doc__ = """Switch for effects of radial ion cloud extent.
May be computationally very intensive"""
ModelOptions.IONISATION_HEATING.__doc__ = """Switch for ionisation heating/recombination cooling"""
ModelOptions.OVERRIDE_FWHM.__doc__ = """If set use FWHM from device definition instead of computed
value."""


# Typedefs for AdvancedModel
logger.debug("Defining numba types for AdvancedModel typing.")
logger.debug("Defining numba types: _T_DEVICE.")
_T_DEVICE = numba.typeof(Device.get(1, 8000, 1e-4, 0.8, 500, 2, 0.005))
logger.debug("Defining numba types: _T_TARGET.")
_T_TARGET = numba.typeof(Target.get_ions("He", 1000., 1., 1))
logger.debug("Defining numba types: _T_BG_GAS.")
_T_BG_GAS = numba.typeof(BackgroundGas.get("He", 1e-8))
logger.debug("Defining numba types: _T_MODEL_OPTIONS.")
_T_MODEL_OPTIONS = numba.typeof(DEFAULT_MODEL_OPTIONS)
logger.debug("Defining numba types: _T_TARGET_LIST.")
_T_TARGET_LIST = numba.types.ListType(_T_TARGET)
logger.debug("Defining numba types: _T_BG_GAS_LIST.")
_T_BG_GAS_LIST = numba.types.ListType(_T_BG_GAS)
logger.debug("Defining numba types: _T_F8_ARRAY.")
_T_F8_ARRAY = numba.float64[:] #Cannot be called in jitted code so need to predefine
logger.debug("Defining numba types: _T_I4_ARRAY.")
_T_I4_ARRAY = numba.int32[:]
logger.debug("Defining numba types: _T_RATE_ENUM.")
_T_RATE_ENUM = numba.typeof(Rate.EI)


logger.debug("Defining AdvancedModel.")
_ADVMDLSPEC = OrderedDict(
    device=_T_DEVICE,
    targets=_T_TARGET_LIST,
    bg_gases=_T_BG_GAS_LIST,
    options=_T_MODEL_OPTIONS,
    lb=_T_I4_ARRAY,
    ub=_T_I4_ARRAY,
    nq=numba.int32,
    q=_T_I4_ARRAY,
    a=_T_I4_ARRAY,
    eixs=_T_F8_ARRAY,
    rrxs=_T_F8_ARRAY,
    drxs=_T_F8_ARRAY,
    cxxs_bggas=numba.types.ListType(_T_F8_ARRAY),
    cxxs_trgts=numba.types.ListType(_T_F8_ARRAY),
)
# @numba.experimental.jitclass(_ADVMDLSPEC)
class AdvancedModel(namedtuple("AdvancedModel", _ADVMDLSPEC.keys())):
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
    @classmethod
    def get(cls, device, targets, bg_gases=None, options=DEFAULT_MODEL_OPTIONS):
        # Bind parameters
        bg_gases = bg_gases if bg_gases is not None else numba.typed.List.empty_list(_T_BG_GAS)

        # Determine array bounds for different targets in state vector
        lb = np.zeros(len(targets), dtype=np.int32)
        ub = np.zeros(len(targets), dtype=np.int32)
        offset = 0
        for i, trgt in enumerate(targets):
            lb[i] = offset
            ub[i] = lb[i] + trgt.z + 1
            offset = ub[i]

        # Compute total number of charge states for all targets (len of "n" or "kT" state vector)
        nq = ub[-1]

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
            targets=targets,
            bg_gases=bg_gases,
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

@numba.njit(cache=True, nogil=True)
def _chunked_adv_rhs(model, t, y, rates=None):
    if y.ndim == 1:
        return _adv_rhs(model, t, y, rates)
    elif y.ndim == 2:
        ret = np.zeros_like(y)
        for k in range(y.shape[1]):
            ret[:, k] = _adv_rhs(model, t, y[:, k], None)
        return ret

@numba.njit(cache=True, nogil=True)
def _adv_rhs(model, _t, y, rates=None):
    """
    The right hand side of the differential equation set.

    Parameters
    ----------
    _t : float
        <s> Time, currently no effect.
    y : numpy.ndarray
        <1/m^3> and <eV>
        Joint array of ion densities and temperatures.
        Array must have the following structure:
        z+1 elements holding the density for each Target in self.targets (same order)
        followed by
        z+1 elements holding the temperature for each Target in self.targets (same order)
    rates: numba.typed.Dict[numpy.ndarray], optional
        If a dictionary object is passed into rates it will be populated with the arrays
        holding the reaction rates.


    Returns
    -------
    numpy.ndarray
        dy/dt
    """
    # pylint: disable=bad-whitespace
    # Split y into useful parts
    n   = y[:model.nq]
    kT  = y[model.nq:]
    # Clip low values?
    # n = np.maximum(n, MINIMAL_N_1D) #-> linear density
    n_r = n[:]
    # n = np.maximum(n, 0) #-> linear density
    # n[n < .001*MINIMAL_N_1D] = 0
    n[n < MINIMAL_N_1D**2] = MINIMAL_N_1D**2
    # n[n < 0] = 0

    # kT = np.maximum(kT, 0)
    kT = np.maximum(kT, MINIMAL_KBT)

    # Transposed helperarrays
    q_T = np.atleast_2d(model.q).T
    a_T = np.atleast_2d(model.a).T
    n_T = np.atleast_2d(n).T
    kT_T = np.atleast_2d(kT).T

    # Preallocate output arrays
    dn  = np.zeros_like(n)
    dkT = np.zeros_like(kT)

    # Radial dynamics
    if model.options.RADIAL_DYNAMICS:
        # Solve radial problem
        phi, _n3d, _shapes = boltzmann_radial_potential_linear_density_ebeam(
            model.device.rad_grid, model.device.current, model.device.r_e, model.device.e_kin,
            n_T, kT_T, q_T,
            ldu=(model.device.rad_fd_l, model.device.rad_fd_d, model.device.rad_fd_u)
        )

    else:
        phi = model.device.rad_phi_uncomp

    ix = model.device.rad_re_idx
    r = model.device.rad_grid

    # Boltzmann distribution shape functions
    shapes = np.exp(-q_T * (phi - phi.min())/kT_T) #Works for neutrals

    # Radial integrals
    i_rs_re = np.trapz(shapes[:, :ix+1] * r[:ix+1], r[:ix+1])
    i_rsp_re = np.trapz(shapes[:, :ix+1] * r[:ix+1] * (phi[:ix+1]-phi.min()), r[:ix+1])
    i_rs_rd = np.trapz(shapes * r, r)
    i_rrs_rd = np.trapz(shapes * r * r, r)

    # On axis 3d density
    n3d = n_T / 2 / PI / np.atleast_2d(i_rs_rd).T * np.atleast_2d(shapes[:, 0]).T
    n3d = n3d.T[0] # Adjust shape

    # Compute overlap factors
    ion_rad = i_rrs_rd / i_rs_rd
    fei = i_rs_re/i_rs_rd
    fij = (ion_rad/np.atleast_2d(ion_rad).T)**2
    fij = np.minimum(fij, 1.0)

    # Compute effective trapping voltages
    v_ax = (model.device.v_ax + model.device.v_ax_sc) - phi.min()
    v_ra = -phi.min()

    # Characteristic beam energies
    _sc_mean = 2*np.trapz(r[:ix+1]*phi[:ix+1], r[:ix+1])/model.device.r_e**2
    e_kin = model.device.e_kin + _sc_mean
    if not model.options.OVERRIDE_FWHM:
        e_kin_fwhm = 2.355*np.sqrt(
            2*np.trapz(r[:ix+1]*(phi[:ix+1]-_sc_mean)**2, r[:ix+1])/model.device.r_e**2
        )
    else:
        e_kin_fwhm = model.device.fwhm

    # Ionisation heating (mean)
    if model.options.IONISATION_HEATING:
        # iheat = 2/3*(2 / self.device.r_e**2 * i_rsp_re)
        iheat = 2/3 * i_rsp_re / i_rs_re
    else:
        iheat = np.zeros(model.nq)


    # Compute some electron beam quantities
    je = model.device.j / Q_E * 1e4 # electron number current density
    ve = plasma.electron_velocity(e_kin)
    ne = je/ve # Electron number density


    # Collision rates
    rij = plasma.ion_coll_rate(
        np.atleast_2d(n3d).T, n3d,
        kT_T, kT,
        a_T, model.a,
        q_T, model.q
    )
    ri  = np.sum(rij, axis=-1)
    # Thermal ion velocities
    v_th = np.sqrt(8 * Q_E * kT/(PI * model.a * M_P)) # Thermal velocities
    # v_z = np.sqrt(kT/(self.a*M_P))
    # f_ax = v_z/(2*self.device.length) # Axial roundtrip frequency
    # f_ra = v_z/(2*self.device.r_dt) #Radial single pass frequency

    # update cross sections?
    if model.options.RECOMPUTE_CROSS_SECTIONS:
        eixs = np.zeros(model.nq)
        rrxs = np.zeros(model.nq)
        drxs = np.zeros(model.nq)
        for i, trgt in enumerate(model.targets):
            eixs[model.lb[i]:model.ub[i]] = xs.eixs_vec(trgt, e_kin)
            rrxs[model.lb[i]:model.ub[i]] = xs.rrxs_vec(trgt, e_kin)
            drxs[model.lb[i]:model.ub[i]] = xs.drxs_vec(trgt, e_kin, e_kin_fwhm)
    else:
        eixs = model.eixs
        rrxs = model.rrxs
        drxs = model.drxs

    # EI
    if model.options.EI:
        R_ei      = eixs * n * je * fei
        dn       -= R_ei
        dn[1:]   += R_ei[:-1]
        dkT[1:]  += R_ei[:-1] / n_r[1:] * (kT[:-1] - kT[1:])
        dkT[1:]  += R_ei[:-1] / n_r[1:] * iheat[:-1]


    # RR
    if model.options.RR:
        R_rr      = rrxs * n * je * fei
        dn       -= R_rr
        dn[:-1]  += R_rr[1:]
        dkT[:-1] += R_rr[1:] / n_r[:-1] * (kT[1:] - kT[:-1])
        dkT[:-1]  -= R_rr[1:] / n_r[:-1] * iheat[1:]


    # DR
    if model.options.DR:
        R_dr      = drxs * n * je * fei
        dn       -= R_dr
        dn[:-1]  += R_dr[1:]
        dkT[:-1] += R_dr[1:] / n_r[:-1] * (kT[1:] - kT[:-1])
        dkT[:-1]  -= R_dr[1:] / n_r[:-1] * iheat[1:]


    # CX
    if model.options.CX:
        R_cx      = np.zeros_like(n)
        for g, gas in enumerate(model.bg_gases):
            R_cx += model.cxxs_bggas[g] * gas.n0 * n * v_th
        for j, jtrgt in enumerate(model.targets):
            if jtrgt.cx: # Only compute cx with target gas if wished by user
                R_cx += model.cxxs_trgts[j] * n3d[model.lb[j]] * n * v_th
        dn       -= R_cx
        dn[:-1]  += R_cx[1:]
        dkT[:-1] += R_cx[1:] / n_r[:-1] * (kT[1:] - kT[:-1])
        dkT[:-1]  -= R_cx[1:] / n_r[:-1] * iheat[1:]


    # Electron heating / Spitzer heating
    if model.options.SPITZER_HEATING:
        _dkT_eh   = plasma.spitzer_heating(n3d, ne, kT, e_kin, model.a, model.q) * fei
        dkT      += _dkT_eh


    # Ion-ion heat transfer (collisional thermalisation)
    if model.options.COLLISIONAL_THERMALISATION:
        _dkT_ct   = np.sum(
            fij*plasma.collisional_thermalisation(
                kT_T, kT, a_T, model.a, rij
            ), axis=-1
        )
        dkT      += _dkT_ct


    # Axial escape
    if model.options.ESCAPE_AXIAL:
        w_ax      = plasma.trapping_strength_axial(kT, model.q, v_ax)
        R_ax_co      = plasma.collisional_escape_rate(ri, w_ax) * n
        free_ax, tfact_ax = plasma.roundtrip_escape(w_ax)
        R_ax_rt   = free_ax * n * ri#f_ax
        for k in model.lb:
            R_ax_rt[k] = 0
            R_ax_co[k] = 0
        # R_ax_co[n < 10*MINIMAL_N_1D] = 0
        # R_ax_rt[n < 10*MINIMAL_N_1D] = 0
        dn       -= R_ax_co + R_ax_rt
        dkT      -= R_ax_co / n_r * w_ax * kT + R_ax_rt / n_r * (tfact_ax - 1) * kT


    # Radial escape
    if model.options.ESCAPE_RADIAL:
        w_ra      = plasma.trapping_strength_radial(
            kT, model.q, model.a, v_ra, model.device.b_ax, model.device.r_dt
        )
        R_ra_co      = plasma.collisional_escape_rate(ri, w_ra) * n
        free_ra, tfact_ra = plasma.roundtrip_escape(w_ra)
        R_ra_rt   = free_ra * n * ri#f_ra
        for k in model.lb:
            R_ra_rt[k] = 0
            R_ra_co[k] = 0
        # R_ra_co[n < 10*MINIMAL_N_1D] = 0
        # R_ra_rt[n < 10*MINIMAL_N_1D] = 0
        dn       -= R_ra_co + R_ra_rt
        dkT      -= R_ra_co / n_r * w_ra * kT + R_ra_rt / n_r * (tfact_ra - 1) * kT


    #TODO: Expansion cooling

    #Check if neutrals are depletable or if there is continuous neutral injection
    for k in model.lb:
        # Kill all neutral rates - seems to improve stability
        dn[k] = 0.0
        dkT[k] = 0.0


    if rates is not None:
        if model.options.EI:
            rates[Rate.EI] = R_ei
            # rates[Rate.T_EI] = R_ei * kT/n
        if model.options.RR:
            rates[Rate.RR] = R_rr
            # rates[Rate.T_RR] = R_rr * kT/n
        if model.options.CX:
            rates[Rate.CX] = R_cx
            # rates[Rate.T_CX] = R_cx * kT/n
        if model.options.DR:
            rates[Rate.DR] = R_dr
            # rates[Rate.T_DR] = R_dr * kT/n
        if model.options.ESCAPE_AXIAL:
            rates[Rate.W_AX]    = w_ax
            rates[Rate.AX_CO] = R_ax_co
            rates[Rate.T_AX_CO] = R_ax_co * (kT + w_ax * kT)/n_r
            rates[Rate.AX_RT] = R_ax_rt
            rates[Rate.T_AX_RT] = R_ax_rt * (tfact_ax - 1) * kT/n_r
            # rates["free_ax"] = free_ax
        if model.options.ESCAPE_RADIAL:
            rates[Rate.W_RA]    = w_ra
            rates[Rate.RA_CO] = R_ra_co
            rates[Rate.T_RA_CO] = R_ra_co * (kT + w_ra * kT)/n_r
            rates[Rate.RA_RT] = R_ra_rt
            rates[Rate.T_RA_RT] = R_ra_rt * (tfact_ra - 1) * kT/n_r
            # rates["free_ra"] = free_ra
        if model.options.COLLISIONAL_THERMALISATION:
            rates[Rate.T_COLLISIONAL_THERMALISATION] = _dkT_ct
        if model.options.SPITZER_HEATING:
            rates[Rate.T_SPITZER_HEATING] = _dkT_eh
        if model.options.IONISATION_HEATING:
            rates[Rate.IONISATION_HEAT] = iheat
        # rates[Rate.CV] = heat_capacity(self.device.rad_grid, phi, q_T, kT_T).T[0]
        # rates[Rate.CHARGE_COMPENSATION] = comp
        rates[Rate.F_EI] = fei
        rates[Rate.E_KIN_MEAN] = np.atleast_1d(np.array(e_kin))
        rates[Rate.E_KIN_FWHM] = np.atleast_1d(np.array(e_kin_fwhm))
        rates[Rate.V_RA] = np.atleast_1d(np.array(v_ra))
        rates[Rate.V_AX] = np.atleast_1d(np.array(v_ax))
        rates[Rate.COLLISION_RATE_SELF] = np.diag(rij).copy()
        rates[Rate.COLLISION_RATE_TOTAL] = ri

    dn[n < MINIMAL_N_1D] = np.maximum(dn[n < MINIMAL_N_1D], 0)
    # dkT[kT < 1 * MINIMAL_KBT] = np.maximum(dkT[kT < 1 * MINIMAL_KBT], 0)
    # dkT[n < .1 * MINIMAL_N_1D] = 0
    # dn[n < MINIMAL_N_1D] = 0
    return np.concatenate((dn, dkT))


logger.debug("Defining advanced_simulation.")
def advanced_simulation(device, targets, t_max, bg_gases=None, options=None, rates=False,
                        solver_kwargs=None, verbose=True, n_threads=1):
    """
    Interface for performing advanced charge breeding simulations.

    For a list of effects refer to `ebisim.simulation.ModelOptions`.

    Parameters
    ----------
    device : ebisim.simulation.Device
        Container describing the EBIS/T and specifically the electron beam.
    targets : ebisim.simulation.Target or list[ebisim.simulation.Target]
        Target(s) for which charge breeding is simulated.
    t_max : float
        <s>
        Simulated breeding time
    bg_gases : ebisim.simulation.BackgroundGas or list[ebisim.simulation.BackgroundGas], optional
        Background gas(es) which act as CX partners, by default None.
    rates : bool
        If true a 'second run' is performed to store the rates, this takes extra time and can
        create quite a bit of data.
    options : ebisim.simulation.ModelOptions, optional
        Switches for effects considered in the simulation, see default values of
        ebisim.simulation.ModelOptions.
    solver_kwargs : None or dict, optional
        If supplied these keyword arguments are unpacked in the solver call.
        Refer to the documentation of scipy.integrate.solve_ivp for more information.
        By default None.
    verbose : bool, optional
        Print a little progress indicator and some status messages, by default True.
    n_threads : int
        How many threads to use (mostly for jacbion estimation which can evaluate the RHS
        in parallel with different inputs.)

    Returns
    -------
    ebisim.simulation.Result or tuple[ebisim.simulation.Result]
        An instance of the Result class, holding the simulation parameters, timesteps and
        charge state distribution including the species temperature.
    """
    logger.info("Preparing advanced simulation.")
    logger.info(f"device = {device}.")
    logger.info(f"targets = {targets}.")
    logger.info(f"t_max = {t_max}.")
    logger.info(f"bg_gases = {bg_gases}.")
    logger.info(f"options = {options}.")
    logger.info(f"rates = {rates}.")
    logger.info(f"solver_kwargs = {solver_kwargs}.")
    logger.info(f"verbose = {verbose}.")
    if options is None:
        options = DEFAULT_MODEL_OPTIONS
    if isinstance(targets, Target):
        targets = [targets]
    targets = numba.typed.List(targets)

    if bg_gases is None:
        bg_gases = numba.typed.List.empty_list(_T_BG_GAS)
    else:
        if isinstance(bg_gases, BackgroundGas):
            bg_gases = [bg_gases]
        bg_gases = numba.typed.List(bg_gases)

    logger.debug("Initialising AdvancedModel object.")
    model = AdvancedModel.get(device, targets, bg_gases, options)

    _n0 = np.concatenate([t.n for t in targets])
    _kT0 = []
    for t in targets: # Make sure that initial temperature is not unreasonably small
        kT = t.kT.copy()
        minkT = np.maximum(device.fwhm * np.arange(t.z+1), MINIMAL_KBT)
        kT[t.n < 1.00001 * MINIMAL_N_1D] = \
            np.maximum(kT[t.n < 1.00001 * MINIMAL_N_1D], minkT[t.n < 1.00001 * MINIMAL_N_1D])
        if np.logical_and(np.not_equal(kT, t.kT), t.n > 1.00001 * MINIMAL_N_1D).any():
            logger.warning(f"Initial temperature vector adjusted for {t}, new: {kT}")
        _kT0.append(kT)
    _kT0 = np.concatenate(_kT0)
    n_kT_initial = np.concatenate([_n0, _kT0])

    # prepare solver options
    if not solver_kwargs:
        solver_kwargs = {}
    solver_kwargs.setdefault("method", "Radau")

    # save adjusted call parameters for passing on to Result
    param = locals().copy()

    with Parallel(n_jobs=n_threads, prefer="threads") as parallel:
        def mt(model, t, y, rates):
            if rates is not None:
                return _chunked_adv_rhs(model, t, y, rates)

            nc = 1 if y.ndim == 1 else y.shape[1]
            cl = n_threads * [nc//n_threads,]
            for _k in range(n_threads):
                if _k < nc%n_threads:
                    cl[_k] += 1
            jobs = []
            for _k in range(n_threads):
                if cl[_k] > 0:
                    jobs.append(
                        delayed(_chunked_adv_rhs)(model, t, y[:, sum(cl[:_k]):sum(cl[:_k+1])])
                    )
            res = parallel(jobs)
            return np.concatenate(res, axis=-1)


        if verbose:
            logger.debug("Wrapping rhs in progress meter.")
            k_old = k = 0
            print("")
            def rhs(t, y, rates=None):
                nonlocal k
                nonlocal k_old
                k += 1 if len(y.shape) == 1 else y.shape[1]
                if k-k_old > 99:
                    print("\r", f"Integration: {k} calls, t = {t:.4e} s", end="")
                    k_old = k
                return mt(model, t, y, rates)
        else:
            rhs = lambda t, y, rates=None: mt(model, t, y, rates)


        logger.debug("Starting integration.")
        res = scipy.integrate.solve_ivp(
            rhs, (0, t_max), n_kT_initial, vectorized=True, **solver_kwargs
        )
        if verbose:
            print("\rIntegration finished:", k, "calls                    ")
            print(res.message)
            print(f"Calls: {k} of which ~{res.nfev} normal ({res.nfev/k:.2%}) and " \
                f"~{res.y.shape[0]*res.njev} for jacobian approximation "\
                f"({res.y.shape[0]*res.njev/k:.2%})")

        if rates:
            logger.debug("Assembling rate arrays.")
            # Recompute rates for final solution (this cannot be done parasitically due to
            # the solver approximating the jacobian and calling rhs with bogus values).
            nt = res.t.size

            #Poll once to get the available rates
            # extractor = list(_rates.values())[0]
            extractor = numba.typed.Dict.empty(
                key_type=numba.typeof(Rate.EI),
                value_type=numba.types.float64[:]
            )
            _ = rhs(res.t[0], res.y[:, 0], extractor)
            rates = {}
            for k in extractor:
                if len(extractor[k].shape) == 1:
                    rates[k] = np.zeros((extractor[k].size, nt))

            # Poll all steps
            for idx in range(nt):
                if verbose and idx%100 == 0:
                    print("\r", f"Rates: {idx+1} / {nt}", end="")
                # if res.t[idx] in _rates: #Technically this data should already exist
                #     extractor = _rates[res.t[idx]]
                # else: #In case it does not -> recompute
                #     _ = model.rhs(res.t[idx], res.y[:, idx], extractor)
                _ = rhs(res.t[idx], res.y[:, idx], extractor)
                for key, val in extractor.items():
                    if len(val.shape) == 1:
                        rates[key][:, idx] = val

            if verbose:
                print("\rRates finished:", nt, "rates")

    out = []
    for i, trgt in enumerate(model.targets):
        logger.debug(f"Assembling result of target #{i}.")
        if rates:
            irates = {}
            for key in rates.keys():
                _ir = rates[key]
                if _ir.shape[0] != 1:
                    irates[key] = _ir[model.lb[i]:model.ub[i]] #Per CS
                else:
                    irates[key] = _ir #scalar
        else:
            irates = None
        out.append(
            Result(
                param=param,
                t=res.t,
                N=res.y[model.lb[i]:model.ub[i]],
                kbT=res.y[model.nq + model.lb[i]:model.nq + model.ub[i]],
                res=res,
                target=trgt,
                device=device,
                rates=irates,
                model=model,
                id_=i
            )
        )
    if len(out) == 1:
        return out[0]
    return tuple(out)
