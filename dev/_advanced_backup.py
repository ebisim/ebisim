"""
This module contains the advanced simulation method and related resources.
"""

from collections import namedtuple
import numpy as np
import scipy.integrate
import scipy.interpolate
import numba

from .. import xs
from .. import elements
from .. import plasma
from ..physconst import Q_E, M_P, PI, EPS_0
from ..physconst import MINIMAL_DENSITY, MINIMAL_KBT
from ._result import Result

Target = namedtuple("Target", elements.Element._fields + ("N", "kbT", "cni", "cx"))

def get_gas_target(element, N, kbT=0.025, cni=True, cx=True):
    # cast element to Element if necessary
    if not isinstance(element, elements.Element):
        element = elements.get_element(element)
    _N = np.ones(element.z + 1) * MINIMAL_DENSITY
    _kbT = np.ones(element.z + 1) * MINIMAL_KBT
    _N[0] = N
    _kbT[0] = kbT
    return Target(*element, _N, _kbT, cni, cx)

def get_ion_target(element, N, kbT=0.025, q=1):
    # cast element to Element if necessary
    if not isinstance(element, elements.Element):
        element = elements.get_element(element)
    _N = np.ones(element.z + 1) * MINIMAL_DENSITY
    _kbT = np.ones(element.z + 1) * MINIMAL_KBT
    _N[q] = N
    _kbT[q] = kbT
    return Target(*element, _N, _kbT, cni=False, cx=False)

Device = namedtuple("Device", ["j", "e_kin", "current", "r_e", "v_ax", "v_ra", "b_ax", "r_dt"])

@numba.njit(cache=True)
def rk4_step(fun, t, y, h, funargs):
    k1 = h * fun(t, y, *funargs)
    k2 = h * fun(t + h/2, y + k1/2, *funargs)
    k3 = h * fun(t + h/2, y + k2/2, *funargs)
    k4 = h * fun(t + h, y + k3, *funargs)
    return (t+h), (y + 1/6*(k1 + 2*k2 + 2*k3 + k4))

@numba.njit(cache=True)
def rk4(fun, t0, t1, h, y0, funargs):
    nt = int((t1 - t0)/h + 1)
    t = np.empty(nt)
    y = np.empty((y0.shape[0], nt))
    t[0] = t0
    y[:, 0] = y0
    for k in range(1, nt-1):
        t[k], y[:, k] = rk4_step(fun, t[k-1], y[:, k-1], h, funargs)
    t[nt-1], y[:, nt-1] = rk4_step(fun, t[nt-2], y[:, nt-2], t1-t[nt-2], funargs)
    return t, y

@numba.njit(cache=True)
def _dpot(x, y, current, e_kin, r_e, q, N, kbT):
    rho_e = -current / (plasma.electron_velocity(e_kin)*r_e**2*PI) * np.exp(-(x/r_e)**2)
    rho_p = Q_E*q * N * np.exp(-q * y[0]/kbT)
    # print(type(rho_p))
    rho_p = np.sum(rho_p)
    ypp = -(rho_e + rho_p)/EPS_0
    # ypp = -(rho_e)/EPS_0
    if y[1] > 0:
        ypp -= y[1]/x
    return np.array([y[1], ypp])

@numba.njit(cache=True)
def compute_potential(device, q, N, kbT):
    r_e = .81 * device.r_e
    sc_post = 1
    sc_pre = 0
    n = 0
    while np.abs(1 - sc_pre / sc_post) > 0.01 and n<10:
        n = n+1
    # for _ in range(3):
        sc_pre = sc_post
        # fun = lambda x, y: _dpot(x, y, device.current, device.e_kin - sc_post, r_e, q, N, kbT)
        # res = scipy.integrate.solve_ivp(fun, (0, device.r_dt), np.array([0, 0]), t_eval=rs)
        x, y = rk4(
            _dpot,
            0,
            device.r_dt,
            device.r_e/50,
            np.zeros(2),
            (device.current, device.e_kin-sc_post, device.r_e, q, N, kbT)
        )
        sc_post = y[0, -1]
    return x, y[0] - sc_post


@numba.njit(cache=True)
def _rhs(t, y, device, target, rates=None):

    # prepare rhs
    ## electrons
    ve = plasma.electron_velocity(device.e_kin)
    je = device.j / Q_E * 1e4
    je = device.current / Q_E / PI / device.r_e**2
    Ne = je / ve
    on_ax_sc = device.current/(4 * PI * EPS_0 * ve) * (2*np.log(device.r_e/device.r_dt)-1)
    ## ions
    element = target
    q = np.arange(element.z + 1)
    A = element.a
    ## cross sections
    eixs = xs.eixs_vec(element, device.e_kin)
    rrxs = xs.rrxs_vec(element, device.e_kin)
    cxxs = 1.43e-16 * q**1.17 * element.ip**-2.76
    if False:#dr_fwhm is not None:
        drxs = xs.drxs_vec(element, device.e_kin, dr_fwhm)
    else:
        drxs = np.zeros(element.z + 1)


    # rs = np.linspace(0, device.r_dt, 1000)
    # rs, phi_empty = compute_potential(device, q, np.zeros_like(q), np.ones_like(q))
    # on_ax_sc_empty = phi_empty[0]

    ### Split y vector into density and temperature
    N = y[:element.z + 1]
    kbT = y[element.z + 1:]
    kbT_clip = kbT[:]
    kbT_clip[kbT_clip < 0] = 0
    # E = N * kbT # Not currently needed

    # Ion cloud
    # print(_, phi[0])
    # on_ax_sc = phi[0]

    # Compensation
    comp = np.sum(q*N)/Ne
    # if comp > 0.0:
    #     rs, phi = compute_potential(device, q, N, kbT)
        # print(comp, _, phi[0])
    mean_ion_heat = 1/2 * device.current/(4 * PI * EPS_0 * ve) * (1-comp)
    v_trap_rad = -1 * on_ax_sc * (1-comp)
    v_trap_ax = device.v_ax + comp * on_ax_sc
    if v_trap_ax < 0:
        v_trap_ax = 1e-16
    if v_trap_rad < 0:
        v_trap_rad = 1e-16


    # precompute collision rates
    rij = plasma.ion_coll_rate_mat(N, N, kbT, kbT, A, A)
    ri = np.sum(rij, axis=1)

    ### Particle density rates
    R_ei = je * N * eixs
    R_rr = je * N * rrxs
    R_dr = je * N * drxs

    if target.cx:
        R_cx = N * N[0] * np.sqrt(8 * Q_E * kbT_clip/(PI * A * M_P)) * cxxs
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
                                np.sqrt(2 * Q_E * kbT_clip / (3 * A *M_P))))
    # Electron heating
    S_eh = plasma.electron_heating_vec(N, Ne, kbT, device.e_kin, A)
    # Energy transfer between charge states within same "species"
    S_tr = plasma.energy_transfer_vec(N, N, kbT, kbT, A, A, rij)
    # Ionisation heating
    S_ih = np.zeros_like(S_eh)
    S_ih[1:] = R_ei[:-1] * mean_ion_heat
    # S_ih[:-1] -= (R_rr + R_dr + R_cx)[1:] * mean_ion_heat


    ### Construct rhs for N (density)
    R_tot = -(R_ei + R_rr + R_dr + R_cx) - (R_ax + R_ra)
    R_tot[1:] += R_ei[:-1]
    R_tot[:-1] += R_rr[1:] + R_dr[1:] + R_cx[1:]

    ### Construct rates for energy density flow
    S_tot = -(S_ei + S_rr + S_dr + S_cx) + S_eh + S_tr + S_ih - (S_ax + S_ra)
    S_tot[1:] += S_ei[:-1]
    S_tot[:-1] += S_rr[1:] + S_dr[1:] + S_cx[1:]

    if target.cni:
        R_tot[0] = 0
        S_tot[0] = 0

    ### Deduce temperature flow -> Integrating temperature instead of energy has proven more
    ### numerically stable
    Q_tot = (S_tot - kbT * R_tot) / N
    Q_tot[N <= MINIMAL_DENSITY] = 0 ### Freeze temperature if density is low (-> not meaningful)

    if rates is not None:
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
        # rates["Comp"] = comp

    return np.concatenate((R_tot, Q_tot))


@numba.njit(parallel=True, cache=True, nogil=True)
def _rhs_vectorised(t, y, device, target, rates=None):
    out = np.empty_like(y)
    for k in range(y.shape[1]):
        out[:, k] = _rhs(t, y[:, k], device, target, rates)
    return out

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


    rhs = lambda t, y: _rhs_vectorised(t, y, device, target)
    res = scipy.integrate.solve_ivp(rhs, (0, t_max), N_kbT_initial, vectorized=True, **solver_kwargs)

    # Recompute rates for final solution (this cannot be done parasitically due to
    # the solver approximating the jacobian and calling rhs with bogus values).
    nt = res.t.size
    poll = numba.typed.Dict.empty(
        key_type=numba.types.unicode_type,
        value_type=numba.types.float64[:]
    )
    _ = _rhs(res.t[0], res.y[:, 0], device, target, rates=poll)
    rates = {k:np.zeros((poll[k].size, nt)) for k in poll}
    for idx in range(nt):
        _ = _rhs(res.t[idx], res.y[:, idx], device, target, rates=poll)
        for key, val in poll.items():
            rates[key][:, idx] = val

    return Result(
        param=param,
        t=res.t,
        N=res.y[:target.z + 1, :],
        kbT=res.y[target.z + 1:, :],
        res=res,
        rates=rates
        )
