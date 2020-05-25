"""
This module contains a model builder. It can be used to generate the RHS of a charge breeding
simulation including certain effects.
"""

from collections import namedtuple
import numpy as np
import numba

from .. import xs
from .. import elements
from .. import plasma
from ..physconst import Q_E, M_P, PI
from ..physconst import MINIMAL_DENSITY, MINIMAL_KBT

Target = namedtuple("Target", elements.Element._fields + ("N", "kbT", "cni", "cx"))
Device = namedtuple("Device", ["j", "e_kin", "current", "r_e", "v_ax", "v_ra", "b_ax", "r_dt"])



class AdvancedModel:
    def __init__(self, device, targets, bg_gases=None):
        # Bind parameters
        self.device = device
        self.targets = targets
        self.bg_gases = bg_gases

        # Define vectors listing the charge state and mass for each state
        self._q = np.concatenate([np.arange(trgt.z + 1, dtype=np.int32) for trgt in self.targets])
        self._a = np.concatenate(
            [np.full(trgt.z + 1, trgt.a, dtype=np.int32) for trgt in self.targets]
        )
        # Compute total number of charge states for all targets (len of "n" or "kT" state vector)
        self._nq = self._q.size


        self._eixs = np.zeros(self._nq)
        self._rrxs = np.zeros(self._nq)
        self._drxs = np.zeros(self._nq)
        self._update_eirrdrxs(self.device.e_kin, _DRFWHM)


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
            self._eixs[self.lb[i]:self.ub[i]] = xs.eixs_vec(trgt, e_kin)
            self._rrxs[self.lb[i]:self.ub[i]] = xs.rrxs_vec(trgt, e_kin)
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
        rij = plasma.ion_coll_rate_mat(n, n, kT, kT, self._a, self._a)
        ri  = np.sum(rij, axis=-1)
        # Thermal velocities
        v_th = np.sqrt(8 * Q_E * kT/(PI * self._a * M_P))

        # TODO: Radial trapping: Trap depth, compensation, Radial extent, overlap? ionisat heat


        # EI
        R_ei      = self._eixs * n * je
        dn       -= R_ei
        dn[1:]   += R_ei[:-1]
        dkT[1:]  += R_ei[:-1] / n[1:] * (kT[:-1] - kT[1:])
        # TODO: Ionisation Heating


        # RR
        R_rr      = self._rrxs * n * je
        dn       -= R_rr
        dn[:-1]  += R_rr[1:]
        dkT[:-1] += R_rr[1:] / n[:-1] * (kT[1:] - kT[:-1])
        # TODO: RR cooling


        # DR
        R_dr      = self._drxs * n * je
        dn       -= R_dr
        dn[:-1]  += R_dr[1:]
        dkT[:-1] += R_dr[1:] / n[:-1] * (kT[1:] - kT[:-1])
        # TODO: DR cooling


        # Electron heating / Spitzer heating
        dkT      += plasma.electron_heating_vec(n, ne, kT, self.device.e_kin, trgt.A)


        # Ion-ion heat transfer (collisional thermalisation)
        dkT      += plasma.energy_transfer_vec(n, n, kT, kT, self._a, self._a, rij)


        # Axial escape
        # TODO: Check axial energy escape rate
        R_ax      = plasma.escape_rate_axial(n, kT, ri, self.device.v_ax)
        dn       -= R_ax
        dkT      -= R_ax / n * self._q * self.device.v_ax


        # Radial escape
        # TODO: Check radial energy escape rate
        R_ra      = plasma.escape_rate_radial(
            n, kT, ri, trgt.A, self.device.v_rad, self.device.b_ax, self.device.r_dt
        )
        dn       -= R_ra
        dkT      -= R_ra / n * self._q * (self.device.v_ra
            + self.device.r_dt * self.device.b_ax * np.sqrt(2 * Q_E * kT / (3 * self._a *M_P))
        )



        # Loop through targets
        for i, trgt in enumerate(self.targets):
            # Extract relevant state from y
            lb = self.lb[i]
            ub = self.ub[i]
            _n = n[lb:ub]
            _kT = kT[lb:ub]
            _ri = ri[lb:ub]
            kT[kT < MINIMAL_KBT] = MINIMAL_KBT # Make sure T is positive

            # and allocate arrays for rates and other useful stuff
            dn = np.zeros_like(n)
            dkT = np.zeros_like(kT)
            q = np.arange(trgt.z + 1)
            v_th = np.sqrt(8 * Q_E * kT/(PI * trgt.A * M_P))






            # CX
            R_cx      = np.zeros_like(n)
            for gas in self.bg_gases:
                cxxs  = 1.43e-16 * q**1.17 * gas.ip**-2.76
                R_cx += cxxs * n * v_th * _NN0
            for j, jtrgt in enumerate(self.targets):
                if jtrgt.cx: # Only compute cx with target gas if wished by user
                    cxxs  = 1.43e-16 * q**1.17 * jtrgt.ip**-2.76
                    R_cx += cxxs * n * v_th * y[self.lb[j]]
            dn       -= R_cx
            dn[:-1]  += R_cx[1:]
            dkT[:-1] += R_cx[1:] / n[:-1] * (kT[1:] - kT[:-1])
            # TODO: CX cooling





            #Check if neutrals are depletable or if there is continuous neutral injection
            if trgt.cni:
                dn[0] = 0.0
                dkT[0] = 0.0


            # Fill rates into output vectors
            dn[lb, ub] = dn[:]
            dkT[lb, ub] = dkT[:]

        # TODO: Return rates on demand
        return np.concatenate((dn, dkT))

    def jac(self, t, y):
        pass
