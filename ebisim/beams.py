"""
This module contains tools for computing characteristic quantities of electron beams typically
found in an electron beam ion source / trap.
"""

import numpy as np
from .physconst import Q_E, PI, EPS_0, K_B, M_E
from .plasma import electron_velocity


class ElectronBeam:
    """
    This class contains logic that allows estimating the space charge corrected energy
    of an electron beam and the resulting energy spread

    Parameters
    ----------
    cur : float
        <A>
        Electron beam current.
    b_d : float
        <T>
        Magnetic flux density in the centre (drift tubes) of the EBIS.
    r_d : float
        <m>
        Drift tube radius.
    b_c : float
        <T>
        Magentic flux density on the cathode surface
    r_c : float
        <m>
        Cathode radius.
    t_c : float
        <K>
        Cathode temperature.

    """
    def __init__(self, cur, b_d, r_d, b_c, r_c, t_c):
        self._cur = cur
        self._b_d = b_d
        self._r_d = r_d
        self._r_c = r_c
        self._t_c = t_c
        self._b_c = b_c

    @property
    def current(self):
        """
        Current of the electron beam.
        """
        return self._cur

    @current.setter
    def current(self, val):
        self._cur = val

    def characteristic_potential(self, e_kin):
        """
        Returns the characteristic potential due to spacecharge

        Parameters
        ----------
        e_kin : float
            <eV>
            Electron kinetic energy.

        Returns
        -------
        float
            <V>
            Characteristic potential.
        """
        v = electron_velocity(e_kin)
        return self._cur / (4 * PI * EPS_0 * v)

    def herrmann_radius(self, e_kin):
        """
        Returns the Hermann radius of an electron beam with the given machine parameters

        Parameters
        ----------
        e_kin : float
            <eV>
            Electron kinetic energy.

        Returns
        -------
        float
            <m>
            Hermann radius.

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
        until the realtive change in space charge correction is < 1e-6.

        Parameters
        ----------
        e_kin : float
            <eV>
            Electron kinetic energy.
        r : float, optional
            Distance from the axis at which to evaluate the correction, by default 0.

        Returns
        -------
        float
            <V>
            Space charge correction.

        Raises
        ------
        ValueError
            If r is larger than the drift tube radius.
        """
        if r > self._r_d or r < 0:
            raise ValueError("r cannot be bigger than the drift tube radius or smaller than 0")

        # Init iterative solution
        # This is mainly required to adjust phi_0 as a function of e_kin
        # the hermann radius is essentially constant over large dynamic ranges
        sc_on_ax_new = 1
        sc_on_ax_old = 0
        while (sc_on_ax_new - sc_on_ax_old)/sc_on_ax_new > 1e-6:  # Check relative difference
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
