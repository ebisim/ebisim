"""
This module contains the advanced simulation method and related resources.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Union, Optional, List, Tuple
from functools import lru_cache
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp
import numba

from .. import xs
from .. import plasma
from ._advanced_helpers import (
    Device,
    BackgroundGas,
    AdvancedModel,
    ModelOptions,
    DEFAULT_MODEL_OPTIONS
)
from ..utils import validate_namedtuple_field_types
from ..elements import Element
from ..physconst import Q_E, M_P, PI
from ..physconst import MINIMAL_N_1D, MINIMAL_KBT
from ._result import AdvancedResult, Rate
from ._radial_dist import (
    boltzmann_radial_potential_linear_density_ebeam,
    # heat_capacity
)

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _cubic_spline(x, x1, x2, y1, y2, k1, k2):
    t = (x-x1)/(x2-x1)
    a = k1*(x2-x1) - (y2-y1)
    b = -k2*(x2-x1) + (y2-y1)
    q = (1-t) * y1 + t*y2 + t*(1-t)*((1-t)*a+t*b)
    return q


@numba.njit(cache=True)
def _smooth_to_zero(x):
    N1 = MINIMAL_N_1D
    N2 = 1000*N1
    x = x.copy()
    x[x < N1] = 0
    fil = np.logical_and(N1 < x, x < N2)
    x[fil] = _cubic_spline(x[fil], N1, N2, 0., N2, 0., 1.)
    return x


# @numba.njit(cache=True)
# def _smooth_temp_rate_to_zero(n, dkT):
#     N1 = MINIMAL_N_1D/100
#     N2 = MINIMAL_N_1D
#     dkT = dkT.copy()
#     dkT[n < N1] = 0
#     fil = np.logical_and(N1 < n, n < N2)
#     dkT[fil] = dkT[fil] * _cubic_spline(n[fil], N1, N2, 0., 1., 0., 1.)
#     return dkT


@numba.njit(cache=True, nogil=True)
def _chunked_adv_rhs(model, t, y):
    yf = np.asfortranarray(y)
    ret = np.zeros_like(yf)
    for k in range(yf.shape[1]):
        ret[:, k] = _adv_rhs(model, t, yf[:, k], None)
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
    # Split y into useful parts
    n   = y[:model.nq]
    kT  = y[model.nq:]
    # Clip low values?
    n_r = n[:]
    n = _smooth_to_zero(n)

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
            ldu=(model.device.rad_fd_l, model.device.rad_fd_d, model.device.rad_fd_u),
            max_step=model.options.RADIAL_SOLVER_MAX_STEPS,
            rel_diff=model.options.RADIAL_SOLVER_REL_DIFF,
        )

    else:
        phi = model.device.rad_phi_uncomp

    ix = model.device.rad_re_idx
    r = model.device.rad_grid

    # Boltzmann distribution shape functions
    shapes = np.exp(-q_T * (phi - phi.min())/kT_T)  # Works for neutrals

    # Radial integrals
    i_rs_re = np.trapz(shapes[:, :ix+1] * r[:ix+1], r[:ix+1])
    i_rsp_re = np.trapz(shapes[:, :ix+1] * r[:ix+1] * (phi[:ix+1]-phi.min()), r[:ix+1])
    i_rs_rd = np.trapz(shapes * r, r)
    i_rrs_rd = np.trapz(shapes * r * r, r)

    # On axis 3d density
    n3d = n_T / 2 / PI / np.atleast_2d(i_rs_rd).T * np.atleast_2d(shapes[:, 0]).T
    n3d = n3d.T[0]  # Adjust shape

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
    je = model.device.j / Q_E * 1e4  # electron number current density
    ve = plasma.electron_velocity(e_kin)
    ne = je/ve  # Electron number density

    # Collision rates
    rij = plasma.ion_coll_rate(
        np.atleast_2d(n3d).T, n3d,
        kT_T, kT,
        a_T, model.a,
        q_T, model.q
    )
    ri  = np.sum(rij, axis=-1)
    # Thermal ion velocities
    v_th = np.sqrt(8 * Q_E * kT/(PI * model.a * M_P))  # Thermal velocities

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
            if jtrgt.cx:  # Only compute cx with target gas if wished by user
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
        R_ax_co = np.maximum(R_ax_co, 0.0)
        for k in model.lb:
            R_ax_co[k] = 0
        dn       -= R_ax_co
        dkT      -= 2/3*plasma.collisional_escape_rate(ri, w_ax) * w_ax * kT

    # Radial escape
    if model.options.ESCAPE_RADIAL:
        w_ra      = plasma.trapping_strength_radial(
            kT, model.q, model.a, v_ra, model.device.b_ax, model.device.r_dt
        )
        R_ra_co      = plasma.collisional_escape_rate(ri, w_ra) * n
        R_ra_co = np.maximum(R_ra_co, 0.0)
        for k in model.lb:
            R_ra_co[k] = 0
        dn       -= R_ra_co
        dkT      -= 2/3*plasma.collisional_escape_rate(ri, w_ra) * w_ra * kT

    # Kill all neutral rates - seems to improve stability - assumes nonchanging background gas
    for k in model.lb:
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
        if model.options.ESCAPE_RADIAL:
            rates[Rate.W_RA]    = w_ra
            rates[Rate.RA_CO] = R_ra_co
            rates[Rate.T_RA_CO] = R_ra_co * (kT + w_ra * kT)/n_r
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

    return np.concatenate((dn, dkT))


def advanced_simulation(device: Device, targets: Union[Element, List[Element]], t_max: float,
                        bg_gases: Union[BackgroundGas, List[BackgroundGas], None] = None,
                        options: Optional[ModelOptions] = None, rates: bool = False,
                        solver_kwargs: Optional[Dict[str, Any]] = None,
                        verbose: bool = True, n_threads: int = 1
                        ) -> Union[AdvancedResult, Tuple[AdvancedResult, ...]]:
    """
    Interface for performing advanced charge breeding simulations.

    For a list of effects refer to `ebisim.simulation.ModelOptions`.

    Parameters
    ----------
    device :
        Container describing the EBIS/T and specifically the electron beam.
    targets :
        Target(s) for which charge breeding is simulated.
    t_max :
        <s>
        Simulated breeding time
    bg_gases :
        Background gas(es) which act as CX partners.
    rates :
        If true a 'second run' is performed to store the rates, this takes extra time and can
        create quite a bit of data.
    options :
        Switches for effects considered in the simulation, see default values of
        ebisim.simulation.ModelOptions.
    solver_kwargs :
        If supplied these keyword arguments are unpacked in the solver call.
        Refer to the documentation of scipy.integrate.solve_ivp for more information.
        By default None.
    verbose :
        Print a little progress indicator and some status messages, by default True.
    n_threads :
        How many threads to use (mostly for jacbion estimation which can evaluate the RHS
        in parallel with different inputs.)

    Returns
    -------
        An instance of the AdvancedResult class, holding the simulation parameters, timesteps and
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

    # ----- Pretreat arguments
    targets = [targets] if not isinstance(targets, list) else targets

    bg_gases = bg_gases or []
    bg_gases = [bg_gases] if not isinstance(bg_gases, list) else bg_gases

    options = options or DEFAULT_MODEL_OPTIONS

    solver_kwargs = solver_kwargs or {}
    solver_kwargs.setdefault("method", "Radau")

    # ----- Generate AdvancedModel
    logger.debug("Initialising AdvancedModel object.")
    model = AdvancedModel.get(device, targets, bg_gases, options)

    # ----- Validate Tuple field types
    for i, tg in enumerate(targets):
        if not validate_namedtuple_field_types(tg):
            logger.warning(f"Unable to verify the types of Target #{i}: {tg!s}.")
    for i, bg in enumerate(bg_gases):
        if not validate_namedtuple_field_types(bg):
            logger.warning(f"Unable to verify the types of BgGas #{i}: {bg!s}.")
    if not validate_namedtuple_field_types(device):
        logger.warning(f"Unable to verify the types of {device!s}.")
    if not validate_namedtuple_field_types(options):
        logger.warning(f"Unable to verify the types of {options!s}.")
    if not validate_namedtuple_field_types(model):
        logger.warning(f"Unable to verify the types of {model!s}.")

    # ----- Generate Initial conditions
    n_kT_initial = _assemble_initial_conditions(model)

    # ----- Generate Callable
    if n_threads < 2:
        executor = None

        def rhs_int(t, y):
            return _chunked_adv_rhs(model, t, y)

    else:
        executor = ThreadPoolExecutor()

        def rhs_int(t, y):
            f = lambda ix: _chunked_adv_rhs(model, t, y[:, ix[0]:ix[1]])  # noqa: E731
            ix = _multithreading_indices(y.shape[1], n_threads)
            res = list(executor.map(f, ix))
            return np.concatenate(res, axis=-1)

    # ----- Run simulation
    if verbose:
        rhs_int = _IntegrationProgressWrapper(rhs_int)
    logger.debug("Starting integration.")
    res = solve_ivp(
        rhs_int, (0, t_max), n_kT_initial, vectorized=True, **solver_kwargs
    )
    if isinstance(rhs_int, _IntegrationProgressWrapper):
        rhs_int.finalize_integration_report(res)

    # ----- Extract rates if demanded
    ratebuffer = _gather_rates(model, res, verbose) if rates else None

    # ----- Result assembly
    if executor is not None:
        executor.shutdown()
    return _assemble_results(model, res, ratebuffer)


class _IntegrationProgressWrapper:
    def __init__(self, rhs):
        self.rhs = rhs
        self.k = 0
        self.k_last_emit = 0

    def __call__(self, t: float, y: np.ndarray):
        self._log_progress(t, y)
        return self.rhs(t, y)

    def _log_progress(self, t: float, y: np.ndarray) -> None:
        inc = 1 if len(y.shape) == 1 else y.shape[1]
        self.k = self.k + inc
        if self.k - self.k_last_emit > 99:
            print(f"Integration: {self.k} calls, t = {t:.4e} s", end="\r")
            self.k_last_emit = self.k

    def finalize_integration_report(self, res: Any) -> None:
        """
        Call after integration finishes to finalize the progress display

        Parameters
        ----------
        res :
            The solve_ivp result object
        """
        print("\rIntegration finished:", self.k, "calls                    ")
        print(res.message)
        print(f"Calls: {self.k} of which ~{res.nfev} normal ({res.nfev/self.k:.2%}) and "
              + f"~{res.y.shape[0]*res.njev} for jacobian approximation "
              + f"({res.y.shape[0]*res.njev/self.k:.2%})")


class _RatesProgressWrapper:
    def __init__(self, rhs, steps):
        self.rhs = rhs
        self.k = 0
        self.steps = steps

    def __call__(self, model: AdvancedModel, t: float, y: np.ndarray, rates: Any) -> np.ndarray:
        self._log_progress()
        return self.rhs(model, t, y, rates)

    def _log_progress(self) -> None:
        self.k += 1
        if not self.k % 100:
            print(f"Rates: {self.k} / {self.steps}", end="\r")

    def finalize_rate_report(self) -> None:
        """
        Call after rate extraction finishes to finalize the rate progress display
        """
        print("Rates finished:", self.steps, "rates")


@lru_cache(maxsize=None)
def _multithreading_indices(n_cols: int, n_threads: int) -> Tuple[Tuple[int, int], ...]:
    """
    Computes a balanced distribution of column indices for the multithreaded
    vectorised version of the RHS function

    Parameters
    ----------
    n_cols :
        Number of columns to distribute
    n_threads :
        Max number of available threads

    Returns
    -------
        Tuple of start / stop index pairs
    """
    cl = n_threads * [n_cols//n_threads, ]
    for _k in range(n_threads):
        if _k < (n_cols % n_threads):
            cl[_k] += 1
    indices = []
    for _k in range(n_threads):
        if cl[_k] > 0:
            indices.append((sum(cl[:_k]), sum(cl[:_k+1])))
    return tuple(indices)


def _assemble_initial_conditions(model: AdvancedModel) -> np.ndarray:
    _kT0 = []
    for t in model.targets:  # Make sure that initial temperature is not unreasonably small
        if t.kT is None or t.n is None:
            raise ValueError(f"{t!s} does not provide initial conditions (n, kT).")
        kT = t.kT.copy()
        # I tried to reduce the value of minkT and it caused crashes, I have no solid
        # argument for a value here, but it is obvius that the simulation stability is very
        # sensitive to the temperature/radial well ratio. This must be normalisation issues
        # since it presents itself as np.nan or np.inf being produced during the solution
        # of the rate equations
        minkT = np.maximum(model.device.fwhm * np.arange(t.z+1), MINIMAL_KBT)
        filter_ = t.n < 1.00001 * MINIMAL_N_1D
        kT[filter_] = np.maximum(kT[filter_], minkT[filter_])
        if np.not_equal(kT, t.kT).any():
            logger.warning(
                f"Initial temperature vector adjusted for {t!s}. "
                + "This only affects charge states with densities at the minimum limit."
                )
        _kT0.append(kT)
    _n0 = np.concatenate([t.n for t in model.targets])
    _kT0 = np.concatenate(_kT0)
    return np.concatenate([_n0, _kT0])


def _gather_rates(model, res, verbose):
    logger.debug("Assembling rate arrays.")
    # Recompute rates for final solution (this cannot be done parasitically due to
    # the solver approximating the jacobian and calling rhs with bogus values).
    nt = res.t.size

    rhs = _RatesProgressWrapper(_adv_rhs, nt) if verbose else _adv_rhs

    # Poll once to get the available rates
    extractor = numba.typed.Dict.empty(
        key_type=numba.typeof(Rate.EI),
        value_type=numba.types.float64[::1]
    )
    _ = rhs(model, res.t[0], res.y[:, 0], extractor)
    ratebuffer = {}
    for k in extractor:
        if len(extractor[k].shape) == 1:
            ratebuffer[k] = np.zeros((extractor[k].size, nt))

    # Poll all steps
    for idx in range(nt):
        _ = rhs(model, res.t[idx], res.y[:, idx], extractor)
        for key, val in extractor.items():
            if len(val.shape) == 1:
                ratebuffer[key][:, idx] = val

    if isinstance(rhs, _RatesProgressWrapper):
        rhs.finalize_rate_report()

    return ratebuffer


def _assemble_results(model, res, ratebuffer):
    out = []
    for i, trgt in enumerate(model.targets):
        logger.debug(f"Assembling result of target #{i}.")
        irates = {}
        if ratebuffer is not None:
            for key in ratebuffer.keys():
                _ir = ratebuffer[key]
                if _ir.shape[0] != 1:
                    irates[key] = _ir[model.lb[i]:model.ub[i]]  # Per CS
                else:
                    irates[key] = _ir  # scalar

        out.append(
            AdvancedResult(
                t=res.t,
                N=res.y[model.lb[i]:model.ub[i]],
                kbT=res.y[model.nq + model.lb[i]:model.nq + model.ub[i]],
                res=res,
                target=trgt,
                device=model.device,
                rates=irates or None,
                model=model,
                id_=i
            )
        )
    if len(out) == 1:
        return out[0]
    return tuple(out)
