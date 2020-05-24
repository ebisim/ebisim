"""
This module contains a model builder. It can be used to generate the RHS of a charge breeding
simulation including certain effects.
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
Device = namedtuple("Device", ["j", "e_kin", "current", "r_e", "v_ax", "v_ra", "b_ax", "r_dt"])



class AdvancedModel:
    def __init__(self, device, targets, bg_gases=None):
        # Bind parameters
        self.device = device
        self.targets = targets
        self.bg_gases = bg_gases
        self._halflen = 0
        for trgt in self.targets:
            self._halflen += trgt.z + 1

        # Determine array bounds for differential equation
        # and define a convenience vector holding the mass
        self.nlb = np.zeros(len(targets), dtype=np.int32)
        self.nub = np.zeros(len(targets), dtype=np.int32)
        self.tlb = np.zeros(len(targets), dtype=np.int32)
        self.tub = np.zeros(len(targets), dtype=np.int32)
        self._av = np.ones(self._halflen, dtype=np.int32)
        offset = 0
        for i, trgt in enumerate(targets):
            self.nlb[i] = offset
            self.nub[i] = self.nlb[i] + trgt.z + 1
            self.tlb[i] = offset + self._halflen
            self.tub[i] = self.tlb[i] + trgt.z + 1
            offset = self.nub[i]
            self._av[self.nlb[i]:self.nub[i]] *= trgt.a


    def rhs(self, t, y):
        # pylint: disable=bad-whitespace
        # Preallocate output
        _dn = np.zeros(self._halflen)
        _dkT = np.zeros(self._halflen)

        # Compute some electron beam quantities
        je = self.device.j / Q_E # electron number current density
        ve = plasma.electron_velocity(self.device.e_kin)
        ne = je/ve # Electron density

        # Collision rate matrix
        _n = y[:self._halflen]
        _kT = y[self._halflen:]
        _rij = plasma.ion_coll_rate_mat(_n, _n, _kT, _kT, self._av, self._av)
        _ri = np.sum(_rij, axis=-1)

        # Ion-ion heat transfer (collisional thermalisation)
        _dkT += plasma.energy_transfer_vec(_n, _n, _kT, _kT, self._av, self._av, _rij)

        # Loop through targets
        for i, trgt in enumerate(self.targets):
            # Extract relevant state from y
            nlb = self.nlb[i]
            nub = self.nub[i]
            tlb = self.tlb[i]
            tub = self.tub[i]
            n = y[nlb:nub]
            kT = y[tlb:tub]
            ri = _ri[nlb:ntb]
            kT[kT < MINIMAL_KBT] = MINIMAL_KBT # Make sure T is positive

            # and allocate arrays for rates and other useful stuff
            dn = np.zeros_like(n)
            dkT = np.zeros_like(kT)
            q = np.arange(trgt.z + 1)
            v_th = np.sqrt(8 * Q_E * kT/(PI * trgt.A * M_P))


            # EI
            eixs      = xs.eixs_vec(trgt, self.device.e_kin)
            R_ei      = eixs * n * je
            dn       -= R_ei
            dn[1:]   += R_ei[:-1]
            dkT[1:]  += R_ei[:-1] / n[1:] * (kT[:-1] - kT[1:])
            #TODO: Ionisation Heating


            # RR
            rrxs      = xs.rrxs_vec(trgt, self.device.e_kin)
            R_rr      = rrxs * n * je
            dn       -= R_rr
            dn[:-1]  += R_rr[1:]
            dkT[:-1] += R_rr[1:] / n[:-1] * (kT[1:] - kT[:-1])
            #TODO: RR cooling


            # DR
            drxs      = xs.drxs_vec(trgt, self.device.e_kin, _DRFWHM)
            R_dr      = drxs * n * je
            dn       -= R_dr
            dn[:-1]  += R_dr[1:]
            dkT[:-1] += R_dr[1:] / n[:-1] * (kT[1:] - kT[:-1])
            #TODO: DR cooling


            # CX
            R_cx      = np.zeros_like(n)
            for gas in self.bg_gases:
                cxxs  = 1.43e-16 * q**1.17 * gas.ip**-2.76
                R_cx += cxxs * n * v_th * _NN0
            for j, jtrgt in enumerate(self.targets):
                if jtrgt.cx: # Only compute cx with target gas if wished by user
                    cxxs  = 1.43e-16 * q**1.17 * jtrgt.ip**-2.76
                    R_cx += cxxs * n * v_th * y[self.nlb[j]]
            dn       -= R_cx
            dn[:-1]  += R_cx[1:]
            dkT[:-1] += R_cx[1:] / n[:-1] * (kT[1:] - kT[:-1])
            #TODO: CX cooling



            # Axial escape
            # TODO: Check axial energy escape rate
            R_ax      = plasma.escape_rate_axial(n, kT, ri, self.device.v_ax)
            dn       -= R_ax
            dkT      -= R_ax / n * q * self.device.v_ax


            # Radial escape
            # TODO: Check radial energy escape rate
            R_ra      = plasma.escape_rate_radial(
                n, kT, ri, trgt.A, self.device.v_rad, self.device.b_ax, self.device.r_dt
            )
            dn       -= R_ra
            dkT      -= R_ra / n * q * (
                self.device.v_ra
                + self.device.r_dt * self.device.b_ax * np.sqrt(2 * Q_E * kT / (3 * trgt.A *M_P))
            )


            # Electron heating / Spitzer heating
            dkT += plasma.electron_heating_vec(n, ne, kT, self.device.e_kin, trgt.A)


            #Check if neutrals are depletable or if there is continuous neutral injection
            if trgt.cni:
                dn[0] = 0.0
                dkT[0] = 0.0


            # Fill rates into output vectors
            _dn[nlb, nub] = dn[:]
            _dkT[tlb, tub] = dkT[:]
        return np.concatenate((_dn, _dkT))

    def jac(self, t, y):
        pass
