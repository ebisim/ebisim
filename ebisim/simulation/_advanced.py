"""
This module contains the advanced simulation method and related resources.
"""

from collections import namedtuple, OrderedDict
import numpy as np
import scipy.integrate
import scipy.interpolate
import numba

from .. import xs
from .. import plasma
from ..elements import Element
from ..physconst import Q_E, M_P, PI, EPS_0, K_B
from ..physconst import MINIMAL_DENSITY, MINIMAL_KBT
from ._result import Result


class Target(namedtuple("Target", Element._fields + ("n", "kT", "cni", "cx"))):
    """
    Use the static `get_ions()` or `get_gas()` factory methods to create instances of this class.

    Targets are used in advanced simulations;
    they are an extended version of ebisim.elements.Element.

    There are four extra fields for a more convenient setup of advanced simulations:
    n and kT are vectors holding initial conditions.
    cni is a boolean flag determining whether the neutral particles are continuously injected,
    if yes, then neutrals cannot be depleted or heated.
    cx is a boolean flag determining whether the neutral particles contribute to charge exchange.

    See Also
    --------
    ebisim.simulation.Target.get_ions
    ebisim.simulation.Target.get_gas
    """
    __slots__ = ()

    @classmethod
    def get_gas(cls, element, p, T=300.0, cx=True):
        """
        Factory method for defining a gas injection Target.
        A gas target is a target with constant density in charge state 0, i.e. continuous neutral
        injection (cni=True).

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
        cx : bool, optional
            see class documentation, by default True

        Returns
        -------
        ebisim.simulation.Target
            Ready to use Target specification.
        """
        element = Element.as_element(element)
        _n = np.full(element.z + 1, MINIMAL_DENSITY, dtype=np.float64)
        _kT = np.full(element.z + 1, MINIMAL_KBT, dtype=np.float64)
        _n[0] = (p * 100) / (K_B * T) #Convert from mbar to Pa and compute density at Temp
        _kT[0] = K_B * T / Q_E
        return cls(*element, n=_n, kT=_kT, cni=True, cx=cx)

    @classmethod
    def get_ions(cls, element, n, kT=0.025, q=1, cx=True):
        """
        Factory method for defining a pulsed ion injection Target.
        An ion target has a given density in the charge state of choice q.

        Parameters
        ----------
        element : ebisim.elements.Element or str or int
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.
        n : float
            <1/m^3>
            Density of the initial charge state.
        kT : float, optional
            <eV>
            Temperature / kinetic energy of the injected ions,
            by default 0.025 eV (Room temperature)
        q : int, optional
            Initial charge state, by default 1
        cx : bool, optional
            see class documentation, by default True

        Returns
        -------
        ebisim.simulation.Target
            Ready to use Target specification.
        """
        element = Element.as_element(element)
        _n = np.full(element.z + 1, MINIMAL_DENSITY, dtype=np.float64)
        _kT = np.full(element.z + 1, MINIMAL_KBT, dtype=np.float64)
        _n[q] = n
        _kT[q] = kT
        return cls(*element, n=_n, kT=_kT, cni=False, cx=cx)

#Patching in docstrings
for f in Element._fields:
    setattr(getattr(Target, f), "__doc__", getattr(getattr(Element, f), "__doc__"))
Target.n.__doc__ = """<1/m^3> Array holding the initial density of each charge state."""
Target.kT.__doc__ = """<eV> Array holding the initial temperature of each charge state."""
Target.cni.__doc__ = """Boolean flag determining whether neutral particles of this target are
continuously injected, and hence cannot be depleted or heated."""
Target.cx.__doc__ = """Boolean flag determining whether neutral particles of this target are
considered as charge exchange partners."""


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
BackgroundGas.name.__doc__ = """str Name of the element."""
BackgroundGas.ip.__doc__ = """float <eV> Ionisation potential of this Gas."""
BackgroundGas.n0.__doc__ = """float <1/m^3> Gas number density."""

Device = namedtuple(
    "Device",
    ("j", "e_kin", "current", "r_e", "drfwhm", "v_ax", "v_ra", "b_ax", "r_dt")
)


_MODEL_OPTIONS_DEFAULTS = OrderedDict(
    EI=True, RR=True, CX=True, DR=False,
    SPITZER_HEATING=True, COLLISIONAL_THERMALISATION=True,
    ESCAPE_AXIAL=True, ESCAPE_RADIAL=True,
    RECOMPUTE_CROSS_SECTIONS=False
)
ModelOptions = namedtuple(
    "ModelOptions", _MODEL_OPTIONS_DEFAULTS.keys(), defaults=_MODEL_OPTIONS_DEFAULTS.values()
)
DEFAULT_MODEL_OPTIONS = ModelOptions()
#Patching in docstrings
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


# Typedefs for AdvancedModel
_T_DEVICE = numba.typeof(Device(0., 0., 0., 0., 0., 0., 0., 0., 0.))
_T_TARGET = numba.typeof(Target.get_ions("He", 0., 0., 1))
_T_BG_GAS = numba.typeof(BackgroundGas.get("He", 1e-8))
_T_MODEL_OPTIONS = numba.typeof(DEFAULT_MODEL_OPTIONS)
_T_TARGET_LIST = numba.types.ListType(_T_TARGET)
_T_BG_GAS_LIST = numba.types.ListType(_T_BG_GAS)
_T_F8_ARRAY = numba.float64[:] #Cannot be called in jitted code so need to predefine
_T_I4_ARRAY = numba.int32[:]


_ADVMDLSPEC = OrderedDict(
    device=_T_DEVICE,
    targets=_T_TARGET_LIST,
    bg_gases=_T_BG_GAS_LIST,
    options=_T_MODEL_OPTIONS,
    lb=_T_I4_ARRAY,
    ub=_T_I4_ARRAY,
    nq=numba.int32,
    _q=_T_I4_ARRAY,
    _a=_T_I4_ARRAY,
    _eixs=_T_F8_ARRAY,
    _rrxs=_T_F8_ARRAY,
    _drxs=_T_F8_ARRAY,
    _cxxs_bggas=numba.types.ListType(_T_F8_ARRAY),
    _cxxs_trgts=numba.types.ListType(_T_F8_ARRAY),
)
@numba.experimental.jitclass(_ADVMDLSPEC)
class AdvancedModel:
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
    def __init__(self, device, targets, bg_gases=None, options=DEFAULT_MODEL_OPTIONS):
        # Bind parameters
        self.device = device
        self.targets = targets
        self.bg_gases = bg_gases if bg_gases is not None else numba.typed.List.empty_list(_T_BG_GAS)
        self.options = options


        # Determine array bounds for different targets in state vector
        self.lb = np.zeros(len(targets), dtype=np.int32)
        self.ub = np.zeros(len(targets), dtype=np.int32)
        offset = 0
        for i, trgt in enumerate(targets):
            self.lb[i] = offset
            self.ub[i] = self.lb[i] + trgt.z + 1
            offset = self.ub[i]


        # Compute total number of charge states for all targets (len of "n" or "kT" state vector)
        self.nq = self.ub[-1]


        # Define vectors listing the charge state and mass for each state
        self._q = np.zeros(self.nq, dtype=np.int32)
        self._a = np.zeros(self.nq, dtype=np.int32)
        for i, trgt in enumerate(self.targets):
            self._q[self.lb[i]:self.ub[i]] = np.arange(trgt.z + 1, dtype=np.int32)
            self._a[self.lb[i]:self.ub[i]] = np.full(trgt.z + 1, trgt.a, dtype=np.int32)


        # Initialise cross section vectors
        self._eixs = np.zeros(self.nq)
        self._rrxs = np.zeros(self.nq)
        self._drxs = np.zeros(self.nq)
        self._update_eirrdrxs(self.device.e_kin, self.device.drfwhm)
        # Precompute CX cross sections (invariant)
        self._cxxs_bggas = numba.typed.List.empty_list(_T_F8_ARRAY)
        self._cxxs_trgts = numba.typed.List.empty_list(_T_F8_ARRAY)
        for gas in self.bg_gases:
            self._cxxs_bggas.append(xs.cxxs(self._q, gas.ip))
        for trgt in self.targets:
            self._cxxs_trgts.append(xs.cxxs(self._q, trgt.ip))



    def _update_eirrdrxs(self, e_kin, drfwhm):
        for i, trgt in enumerate(self.targets):
            if self.options.EI:
                self._eixs[self.lb[i]:self.ub[i]] = xs.eixs_vec(trgt, e_kin)
            if self.options.RR:
                self._rrxs[self.lb[i]:self.ub[i]] = xs.rrxs_vec(trgt, e_kin)
            if self.options.DR:
                self._drxs[self.lb[i]:self.ub[i]] = xs.drxs_vec(trgt, e_kin, drfwhm)


    def rhs(self, _t, y):
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


        Returns
        -------
        numpy.ndarray
            dy/dt
        """
        # pylint: disable=bad-whitespace

        # Split y into useful parts
        n   = y[:self.nq]
        kT  = y[self.nq:]
        # Clip low values?
        n = np.maximum(n, MINIMAL_DENSITY)
        # kT = np.maximum(kTn, MINIMAL_KBT)
        # Preallocate output arrays
        dn  = np.zeros_like(n)
        dkT = np.zeros_like(kT)

        # Compute some electron beam quantities
        je = self.device.j / Q_E * 1e4 # electron number current density
        ve = plasma.electron_velocity(self.device.e_kin)
        ne = je/ve # Electron number density

        # Collision rates
        # rij = plasma.ion_coll_rate(
        #     n[:, np.newaxis], n,
        #     kT[:, np.newaxis], kT,
        #     self._a[:, np.newaxis], self._a,
        #     self._q[:, np.newaxis], self._q
        # )
        rij = plasma.ion_coll_rate(
            np.atleast_2d(n).T, n,
            np.atleast_2d(kT).T, kT,
            np.atleast_2d(self._a).T, self._a,
            np.atleast_2d(self._q).T, self._q
        )
        # print(rij)
        ri  = np.sum(rij, axis=-1)
        v_th = np.sqrt(8 * Q_E * kT/(PI * self._a * M_P)) # Thermal velocities

        # TODO: Radial trapping: Trap depth, compensation, Radial extent, overlap? ionisat heat

        # update cross sections?
        if self.options.RECOMPUTE_CROSS_SECTIONS:
            self._update_eirrdrxs(self.device.e_kin, self.device.drfwhm)

        # EI
        if self.options.EI:
            R_ei      = self._eixs * n * je
            dn       -= R_ei
            dn[1:]   += R_ei[:-1]
            dkT[1:]  += R_ei[:-1] / n[1:] * (kT[:-1] - kT[1:])
            # TODO: Ionisation Heating


        # RR
        if self.options.RR:
            R_rr      = self._rrxs * n * je
            dn       -= R_rr
            dn[:-1]  += R_rr[1:]
            dkT[:-1] += R_rr[1:] / n[:-1] * (kT[1:] - kT[:-1])
            # TODO: RR cooling


        # DR
        if self.options.DR:
            R_dr      = self._drxs * n * je
            dn       -= R_dr
            dn[:-1]  += R_dr[1:]
            dkT[:-1] += R_dr[1:] / n[:-1] * (kT[1:] - kT[:-1])
            # TODO: DR cooling


        # CX
        if self.options.CX:
            R_cx      = np.zeros_like(n)
            for g, gas in enumerate(self.bg_gases):
                R_cx += self._cxxs_bggas[g] * gas.n0 * n * v_th
            for j, jtrgt in enumerate(self.targets):
                if jtrgt.cx: # Only compute cx with target gas if wished by user
                    R_cx += self._cxxs_trgts[j] * n[self.lb[j]] * n * v_th
            dn       -= R_cx
            dn[:-1]  += R_cx[1:]
            dkT[:-1] += R_cx[1:] / n[:-1] * (kT[1:] - kT[:-1])
            # TODO: CX cooling


        # Electron heating / Spitzer heating
        if self.options.SPITZER_HEATING:
            dkT      += plasma.spitzer_heating(n, ne, kT, self.device.e_kin, self._a, self._q)


        # Ion-ion heat transfer (collisional thermalisation)
        if self.options.COLLISIONAL_THERMALISATION:
            # dkT      += np.sum(
            #     plasma.collisional_thermalisation(
            #         kT[:, np.newaxis], kT, self._a[:, np.newaxis], self._a, rij
            #     ), axis=-1
            # )
            dkT      += np.sum(
                plasma.collisional_thermalisation(
                    np.atleast_2d(kT).T, kT, np.atleast_2d(self._a).T, self._a, rij
                ), axis=-1
            )


        # Axial escape
        if self.options.ESCAPE_AXIAL:
            # TODO: Check axial energy escape rate
            R_ax      = plasma.escape_rate_axial(n, kT, self._q, ri, self.device.v_ax)
            dn       -= R_ax
            dkT      -= R_ax / n * self._q * self.device.v_ax


        # Radial escape
        if self.options.ESCAPE_RADIAL:
            # TODO: Check radial energy escape rate
            R_ra      = plasma.escape_rate_radial(
                n, kT, self._q, ri, self._a, self.device.v_ra, self.device.b_ax, self.device.r_dt
            )
            dn       -= R_ra
            dkT      -= R_ra / n * self._q * (
                self.device.v_ra + self.device.r_dt * self.device.b_ax
                * np.sqrt(2 * Q_E * kT / (3 * self._a *M_P))
            )



        #Check if neutrals are depletable or if there is continuous neutral injection
        for i, trgt in enumerate(self.targets):
            if trgt.cni:
                dn[self.lb[i]] = 0.0
                dkT[self.lb[i]] = 0.0


        # TODO: Return rates on demand
        return np.concatenate((dn, dkT))


def advanced_simulation(device, targets, t_max, bg_gases=None, options=None, solver_kwargs=None):
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
    options : ebisim.simulation.ModelOptions, optional
        Switches for effects considered in the simulation, see default values of
        ebisim.simulation.ModelOptions.
    solver_kwargs : None or dict, optional
        If supplied these keyword arguments are unpacked in the solver call.
        Refer to the documentation of scipy.integrate.solve_ivp for more information.
        By default None.

    Returns
    -------
    ebisim.simulation.Result or tuple[ebisim.simulation.Result]
        An instance of the Result class, holding the simulation parameters, timesteps and
        charge state distribution including the species temperature.
    """
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

    device = Device(*tuple([float(_f) for _f in device])) #Device is easy to mess up, safety rope
    assert numba.typeof(device) == _T_DEVICE, "Numba type mismatch for device"
    assert numba.typeof(options) == _T_MODEL_OPTIONS, "Numba type mismatch for options"
    assert numba.typeof(targets) == _T_TARGET_LIST, "Numba type mismatch for Target list"
    assert numba.typeof(bg_gases) == _T_BG_GAS_LIST, "Numba type mismatch for BgGas list"

    model = AdvancedModel(device, targets, bg_gases, options)

    _n0 = np.concatenate([t.n for t in targets])
    _kT0 = np.concatenate([t.kT for t in targets])
    n_kT_initial = np.concatenate([_n0, _kT0])
    # prepare solver options
    if not solver_kwargs:
        solver_kwargs = {}
    solver_kwargs.setdefault("method", "Radau")

    # save adjusted call parameters for passing on to Result
    param = locals().copy()

    res = scipy.integrate.solve_ivp(
        model.rhs, (0, t_max), n_kT_initial, **solver_kwargs
    )

    out = []
    for i, trgt in enumerate(model.targets):
        out.append(
            Result(
                param=param,
                t=res.t,
                N=res.y[model.lb[i]:model.ub[i]],
                kbT=res.y[model.nq + model.lb[i]:model.nq + model.ub[i]],
                res=res,
                target=trgt,
                device=device
            )
        )
    if len(out) == 1:
        return out[0]
    return tuple(out)

    # # Recompute rates for final solution (this cannot be done parasitically due to
    # # the solver approximating the jacobian and calling rhs with bogus values).
    # nt = res.t.size
    # poll = numba.typed.Dict.empty(
    #     key_type=numba.types.unicode_type,
    #     value_type=numba.types.float64[:]
    # )
    # _ = _rhs(res.t[0], res.y[:, 0], device, target, rates=poll)
    # rates = {k:np.zeros((poll[k].size, nt)) for k in poll}
    # for idx in range(nt):
    #     _ = _rhs(res.t[idx], res.y[:, idx], device, target, rates=poll)
    #     for key, val in poll.items():
    #         rates[key][:, idx] = val
