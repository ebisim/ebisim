"""
This module contains tools for computing characteristic quantities of electron beams typically
found in an electron beam ion source / trap.
"""

import numpy as np
from .physconst import C_L, Q_E, M_E_EV, PI, EPS_0, K_B, M_E

def electron_velocity(e_kin):
    """
    Returns the electron velocity corresponding to a kin. energy in m/s

    Input Parameters
    e_kin - electron energy in eV
    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)

class ElectronBeam:
    """
    This class contains logic that allows estimating the space charge corrected energy
    of an electron beam and the resulting energy spread
    """
    def __init__(self, cur, b_d, r_d, b_c, r_c, t_c):
        """
        Set the constant machine parameters

        Input Parameters
        cur - The electron current in A
        b_d - magnetic flux density in the trap centre in Tesla
        r_d - drift tube radius in m
        r_c - cathode radius in m
        t_c - cathode temperatur in K
        b_c - magnetic flux density on the cathode surface
        """
        self._cur = cur
        self._b_d = b_d
        self._r_d = r_d
        self._r_c = r_c
        self._t_c = t_c
        self._b_c = b_c

    @property
    def current(self):
        """
        Can read and write a new current to the electron beam object, which will then be used
        for future computations
        """
        return self._cur

    @current.setter
    def current(self, val):
        self._cur = val


    def characteristic_potential(self, e_kin):
        """
        Returns the characteristic potential due to spacecharge in V

        Input Parameters
        e_kin - electron energy in eV
        """
        v = electron_velocity(e_kin)
        return self._cur / (4 * PI * EPS_0 * v)

    def herrmann_radius(self, e_kin):
        """
        Returns the Hermann radius of an electron beam with the given machine parameters in m

        Input Parameters
        e_kin - electron energy in eV
        """
        v = electron_velocity(e_kin)
        s1 = M_E * self._cur / (PI * EPS_0 * Q_E * v * self._b_d**2)
        s2 = 8 * K_B * self._t_c * M_E * self._r_c**2 / (Q_E**2 * self._b_d**2)
        s3 = self._b_c**2 * self._r_c**4 / (self._b_d**2)
        return np.sqrt(s1 + np.sqrt(s1**2 + s2 + s3))

    def space_charge_correction(self, e_kin, r=0):
        """
        Returns the space charge correction at a given radius r, current and electron beam energy
        This is done by iteratively computing the spacecharge correction and the hermann radius
        until the realtive change in space charge correction is < 1e-6

        Input Parameters
        e_kin - electron energy in eV
        r - distance from axis in m, defaults to 0
        """
        if r > self._r_d or r < 0:
            raise ValueError("r cannot be bigger than the drift tube radius or smaller than 0")

        # Init iterative solution
        # This is mainly required to adjust phi_0 as a function of e_kin
        # the hermann radius is essentially constant over large dynamic ranges
        sc_on_ax_new = 1
        sc_on_ax_old = 0
        while (sc_on_ax_new - sc_on_ax_old)/sc_on_ax_new > 1e-6: # Check relative difference
            # Compute kinetic energy correction
            corr_e_kin = e_kin + sc_on_ax_new
            # Compute Herrmann radius and characteristic potential
            r_e = self.herrmann_radius(corr_e_kin)
            phi_0 = self.characteristic_potential(corr_e_kin)
            # Compute space charge correction
            sc_on_ax_old = sc_on_ax_new
            sc_on_ax_new = phi_0 * (2 * np.log(r_e / self._r_d) - 1)

        # When loop is exited r_e and phi_0 should have the right values
        if r < r_e:
            multip = 2 * np.log(r_e / self._r_d) + (r / r_e)**2 - 1
        else:
            multip = 2 * np.log(r / self._r_d)
        return phi_0 * multip

class RexElectronBeam(ElectronBeam):
    """
    ElectronBeam Class with REXEBIS Parameters set by default

    Magnetic Field Trap: 2 T
    Magnetic Field Cathode: 0.2 T
    Cathode Radius: 0.8 mm
    Drift tube Radius: 5 mm
    Cathode Temp: 1600 K
    """
    def __init__(self, cur):
        """
        Set the constant machine parameters

        Input Parameters
        cur - The electron current in A
        """
        b_d = 2
        r_d = 5/1000
        r_c = .8/1000
        t_c = 1600
        b_c = .2
        super().__init__(cur, b_d, r_d, b_c, r_c, t_c)
