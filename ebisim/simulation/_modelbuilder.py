"""
This module contains a model builder. It can be used to generate the RHS of a charge breeding
simulation including certain effects.
"""

from collections import namedtuple, OrderedDict
import numpy as np
import numba

from .. import xs
from .. import elements
from .. import plasma
from ..physconst import Q_E, M_P, PI
from ..physconst import MINIMAL_DENSITY, MINIMAL_KBT


Target = namedtuple("Target", elements.Element._fields + ("n", "kT", "cni", "cx"))
Device = namedtuple(
    "Device",
    ("j", "e_kin", "current", "r_e", "drfwhm", "v_ax", "v_ra", "b_ax", "r_dt")
)
BgGas = namedtuple("BgGas", "ip, n0")

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


class AdvancedModel:
    def __init__(self, device, targets, bg_gases=None, options=DEFAULT_MODEL_OPTIONS):
        # Bind parameters
        self.device = device
        self.targets = targets
        self.bg_gases = bg_gases
        self.options = options

        # Define vectors listing the charge state and mass for each state
        self._q = np.concatenate([np.arange(trgt.z + 1, dtype=np.int32) for trgt in self.targets])
        self._a = np.concatenate(
            [np.full(trgt.z + 1, trgt.a, dtype=np.int32) for trgt in self.targets]
        )
        # Compute total number of charge states for all targets (len of "n" or "kT" state vector)
        self._nq = self._q.size

        # Initialise cross section vectors
        self._eixs = np.zeros(self._nq)
        self._rrxs = np.zeros(self._nq)
        self._drxs = np.zeros(self._nq)
        self._update_eirrdrxs(self.device.e_kin, self.device.drfwhm)
        # Precompute CX cross sections (invariant)
        self._cxxs_bggas = [xs.cxxs(self._q, gas.ip) for gas in self.bg_gases]
        self._cxxs_trgts = [xs.cxxs(self._q, trgt.ip) for trgt in self.targets]

        # Determine array bounds for different targets in state vector
        self.lb = np.zeros(len(targets), dtype=np.int32)
        self.ub = np.zeros(len(targets), dtype=np.int32)
        offset = 0
        for i, trgt in enumerate(targets):
            self.lb[i] = offset
            self.ub[i] = self.lb[i] + trgt.z + 1
            offset = self.ub[i]

    def _update_eirrdrxs(self, e_kin, drfwhm):
        for i, trgt in enumerate(self.targets):
            if self.options.EI:
                self._eixs[self.lb[i]:self.ub[i]] = xs.eixs_vec(trgt, e_kin)
            if self.options.RR:
                self._rrxs[self.lb[i]:self.ub[i]] = xs.rrxs_vec(trgt, e_kin)
            if self.options.DR:
                self._drxs[self.lb[i]:self.ub[i]] = xs.drxs_vec(trgt, e_kin, drfwhm)

    def rhs(self, _t, y):
        # pylint: disable=bad-whitespace

        # Split y into useful parts
        n   = y[:self._nq]
        kT  = y[self._nq:]
        # Preallocate output arrays
        dn  = np.zeros(self._nq)
        dkT = np.zeros(self._nq)

        # Compute some electron beam quantities
        je = self.device.j / Q_E # electron number current density
        ve = plasma.electron_velocity(self.device.e_kin)
        ne = je/ve # Electron number density

        # Collision rates
        rij = plasma.ion_coll_rate(
            n[:, np.newaxis], n,
            kT[:, np.newaxis], kT,
            self._a[:, np.newaxis], self._a,
            self._q[:, np.newaxis], self._q
        )
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
            R_cx      = np.zeros(self._nq)
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
            dkT      += np.sum(
                plasma.collisional_thermalisation(
                    kT[:, np.newaxis], kT, self._a[:, np.newaxis], self._a, rij
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
                n, kT, self._q, ri, self._a, self.device.v_rad, self.device.b_ax, self.device.r_dt
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

    def jac(self, t, y):
        pass
