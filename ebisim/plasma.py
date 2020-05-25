"""
This module contains functions for computing collission rates and related plasma parameters.
"""

import math
import numba
import numba.types as nt
import numpy as np

from .physconst import M_E, M_P, PI, EPS_0, Q_E, C_L, M_E_EV
from .physconst import MINIMAL_DENSITY


# @numba.njit(cache=True)
@numba.vectorize([nt.float64(nt.float64)], cache=True, nopython=True)
def electron_velocity(e_kin):
    """
    Computes the electron velocity corresponding to a kinetic energy.

    Input Parameters
    e_kin - electron energy in eV

    Parameters
    ----------
    e_kin : float or int or numpy.ndarray
        <eV>
        Kinetic energy of the electron.

    Returns
    -------
    float or numpy.ndarray
        <m/s>
        Speed of the electron.

    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)


# @numba.njit(cache=True)
@numba.vectorize([nt.float64(nt.float64, nt.float64, nt.float64, nt.float64, nt.int64, nt.int64)], cache=True, nopython=True)
def clog_ei(Ni, Ne, kbTi, kbTe, Ai, qi):
    """
    The coulomb logarithm for ion electron collisions.

    Parameters
    ----------
    Ni : float
        <1/m^3>
        Ion density.
    Ne : float
        <1/m^3>
        Electron density.
    kbTi : float
        <eV>
        Ion temperature.
    kbTe : float
        <eV>
        Electron temperature.
    Ai : float or int
        Ion mass number.
    qi : float or int
        Ion charge state.

    Returns
    -------
    float
        Ion electron coulomb logarithm.

    """
    # Ni *= 1e-6 # go from 1/m**3 to 1/cm**3 # BE CAREFUL WITH EXTENDING THIS TO ARRAYS (MUTATING)
    # Ne *= 1e-6
    Ni, Ne = Ni *1e-6, Ne*1e-6
    Mi = Ai * M_P
    if   qi*qi*10 >= kbTe >= kbTi * M_E / Mi:
        return 23. - np.log(Ne**0.5 * qi * kbTe**-1.5)
    elif kbTe >= qi*qi*10 >= kbTi * M_E / Mi:
        return 24. - np.log(Ne**0.5 / kbTe)
    elif kbTe <= kbTi * M_E / Mi:
        return 16. - np.log(Ni**0.5 * kbTi**-1.5 * qi * qi * Ai)
    # The next case should not usually arise in any realistic situation but the solver may probe it
    # Hence it is purely a rough guess
    elif qi*qi*10 <= kbTi * M_E / Mi <= kbTe:
        return 24. - np.log(Ne**0.5 / kbTe)
    return 10


# @numba.njit(cache=True)
@numba.vectorize([nt.float64(nt.float64, nt.float64, nt.float64, nt.float64, nt.int64, nt.int64, nt.int64, nt.int64)], cache=True, nopython=True)
def clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    """
    The coulomb logarithm for ion ion collisions.

    Parameters
    ----------
    Ni : float
        <1/m^3>
        Ion species "i" density.
    Nj : float
        <1/m^3>
        Ion species "j" density.
    kbTi : float
        <eV>
        Ion species "i" temperature.
    kbTj : float
        <eV>
        Ion species "j" temperature.
    Ai : float or int
        Ion species "i" mass number.
    Aj : float or int
        Ion species "j" mass number.
    qi : float or int
        Ion species "i" charge state.
    qj : float or int
        Ion species "j" charge state.

    Returns
    -------
    float
        Ion ion coulomb logarithm.

    """
    # Ni *= 1e-6 # go from 1/m**3 to 1/cm**3
    # Nj *= 1e-6
    Ni, Nj = Ni *1e-6, Nj*1e-6
    A = qi * qj * (Ai + Aj) / (Ai * kbTj + Aj * kbTi)
    B = Ni * qi * qi / kbTi + Nj * qj * qj / kbTj
    clog = 23 - np.log(A * B**0.5)
    if clog < 0:
        clog = 0
    return clog


# @numba.njit(cache=True)
@numba.vectorize([nt.float64(nt.float64, nt.float64, nt.float64, nt.float64, nt.int64, nt.int64)], cache=True, nopython=True)
def coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi):
    """
    Computes the Coulomb cross section for elastic electron ion collisions

    Parameters
    ----------
    Ni : float
        <1/m^3>
        Ion density.
    Ne : float
        <1/m^3>
        Electron density.
    kbTi : float
        <eV>
        Ion temperature.
    Ee : float or int
        <eV>
        Electron kinetic energy.
    Ai : float or int
        Ion mass number.
    qi : float or int
        Ion charge state.

    Returns
    -------
    float
        <m^2>
        Coulomb cross section.

    """
    if qi == 0:
        return 0
    v_e = electron_velocity(Ee)
    clog = clog_ei(Ni, Ne, kbTi, Ee, Ai, qi)
    return 4 * PI * (qi * Q_E * Q_E / (4 * PI * EPS_0 * M_E))**2 * clog / v_e**4


# @numba.njit(cache=True)
@numba.vectorize([nt.float64(nt.float64, nt.float64, nt.float64, nt.float64, nt.int64, nt.int64, nt.int64, nt.int64)], cache=True, nopython=True)
def ion_coll_rate(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    """
    Collision rates for ions species "i" and target species "j"

    Parameters
    ----------
    Ni : float
        <1/m^3>
        Ion species "i" density.
    Nj : float
        <1/m^3>
        Ion species "j" density.
    kbTi : float
        <eV>
        Ion species "i" temperature.
    kbTj : float
        <eV>
        Ion species "j" temperature.
    Ai : float or int
        Ion species "i" mass number.
    Aj : float or int
        Ion species "j" mass number.
    qi : float or int
        Ion species "i" charge state.
    qj : float or int
        Ion species "j" charge state.

    Returns
    -------
    float
        <1/s>
        Ion ion collision rate.

    See Also
    --------
    ebisim.plasma.ion_coll_rate_mat : Similar method for all charge states.

    """
    # Artifically clamp collision rate to zero if either density is very small
    # This is a reasonable assumption and prevents instabilities when calling the coulomb logarithm
    if Ni <= MINIMAL_DENSITY or Nj <= MINIMAL_DENSITY or kbTi <= 0 or kbTj <= 0:
        return 0
    if qi == 0 or qj == 0:
        return 1e-10
    clog = clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj)
    kbTi_SI = kbTi * Q_E
    Mi = Ai * M_P
    const = 4 / 3 / (4 * PI * EPS_0)**2 * np.sqrt(2 * PI)
    return const * Nj * (qi * qj * Q_E * Q_E / Mi)**2 * (Mi/kbTi_SI)**1.5 * clog


@numba.njit(cache=True)
def ion_coll_rate_mat(Ni, Nj, kbTi, kbTj, Ai, Aj):
    """
    Matrix holding collision rates for ions species "i" and target species "j"

    Parameters
    ----------
    Ni : numpy.ndarray
        <1/m^3>
        Vector of ion species "i" densities.
    Nj : numpy.ndarray
        <1/m^3>
        Vector of ion species "j" densities.
    kbTi : numpy.ndarray
        <eV>
        Vector of ion species "i" temperatures.
    kbTj : numpy.ndarray
        <eV>
        Vector of ion species "j" temperatures.
    Ai : float or int
        Ion species "i" mass number.
    Aj : float or int
        Ion species "j" mass number.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Ion ion collision rates stored in a 2D array, where
        r_ij[qi, qj] = ion_coll_rate(Ni[qi], Nj[qj], kbTi[qi], kbTj[qj], Ai, Aj, qi, qj)

    See Also
    --------
    ebisim.plasma.ion_coll_rate : Similar method for two single charge states.

    """
    ni = Ni.size
    nj = Nj.size
    r_ij = np.zeros((ni, nj))
    for qi in range(1, ni):
        for qj in range(1, nj):
            r_ij[qi, qj] = ion_coll_rate(Ni[qi], Nj[qj], kbTi[qi], kbTj[qj], Ai, Aj, qi, qj)
    return r_ij


@numba.njit(cache=True)
def electron_heating_vec(Ni, Ne, kbTi, Ee, Ai):
    """
    Computes the heating rates due to elastic electron ion collisions ('Spitzer Heating')

    Parameters
    ----------
    Ni : numpy.ndarray
        <1/m^3>
        Vector of ion densities.
    Ne : float
        <1/m^3>
        Electron density.
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    Ee : float or int
        <eV>
        Electron kinetic energy.
    Ai : float or int
        Ion mass number.

    Returns
    -------
    numpy.ndarray
        <eV/m^3/s>
        Vector of electron heating rate for each charge state.

    """
    n = Ni.size
    heat = np.zeros(n)
    const = Ne * electron_velocity(Ee) * 2 * M_E / (Ai * M_P) * Ee
    for qi in range(1, n):
        if Ni[qi] > MINIMAL_DENSITY:
            heat[qi] = const * Ni[qi] * coulomb_xs(Ni[qi], Ne, kbTi[qi], Ee, Ai, qi)
    return heat


@numba.njit(cache=True)
def energy_transfer_vec(Ni, Nj, kbTi, kbTj, Ai, Aj, rij):
    """
    Computes the collisional energy transfer rates for species "i" with respect to species "j".

    Parameters
    ----------
    Ni : numpy.ndarray
        <1/m^3>
        Vector of ion species "i" densities.
    Nj : numpy.ndarray
        <1/m^3>
        Vector of ion species "j" densities.
    kbTi : numpy.ndarray
        <eV>
        Vector of ion species "i" temperatures.
    kbTj : numpy.ndarray
        <eV>
        Vector of ion species "j" temperatures.
    Ai : float or int
        Ion species "i" mass number.
    Aj : float or int
        Ion species "j" mass number.
    rij : numpy.ndarray
        <1/s>
        Collision rate matrix for the ions (cf. ion_coll_mat).

    Returns
    -------
    numpy.ndarray
        <eV/m^3/s>
        Vector of energy transfer rate for each charge state.

    """
    ni = Ni.size
    nj = Nj.size
    trans_i = np.zeros(ni)
    for qi in range(1, ni):
        for qj in range(1, nj):
            if kbTi[qi] <= 0 or kbTj[qj] <= 0:
                trans_i[qi] += 0
            else:
                trans_i[qi] += 2 * rij[qi, qj] * Ni[qi] * Ai/Aj * (kbTj[qj] - kbTi[qi]) / \
                               (1 + (Ai * kbTj[qj]) / (Aj * kbTi[qi]))**1.5
    return trans_i


@numba.njit(cache=True)
def loss_frequency_axial(kbTi, V):
    """
    Computes the axial trap (loss) frequencies.

    Parameters
    ----------
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    V : float or int
        <V>
        Trap depth.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Vector of axial ion trap (loss) frequencies for each charge state.

    """
    valid = kbTi > 0
    q = np.arange(kbTi.size)/1 # divide by one to make float
    w = q * V
    w[valid] = w[valid] / kbTi[valid]
    w[np.logical_not(valid)] = np.inf
    w[0] = np.inf # fake value for neutrals -> essentially infinite trap
    return w


@numba.njit(cache=True)
def loss_frequency_radial(kbTi, Ai, V, B, r_dt):
    """
    Radial trap (loss) frequencies.
    kbTi - electron energy /temperature in eV
    Ai - ion mass in amu
    V - Trap depth in V
    B - Axial magnetic field in T
    r_dt - drift tube in m


    Parameters
    ----------
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    Ai : float or int
        Ion mass number.
    V : float or int
        <V>
        Trap depth.
    B : float or int
        <T>
        Axial magnetic flux density.
    r_dt : float or int
        <m>
        Drift tube radius.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Vector of radial ion trap (loss) frequencies for each charge state.

    """
    valid = kbTi > 0
    q = np.arange(kbTi.size)/1 # divide by one to make float
    w = q
    w[valid] = w[valid] * (V + B * r_dt * np.sqrt(2 * kbTi[valid] * Q_E /(3*M_P*Ai))) / kbTi[valid]
    w[np.logical_not(valid)] = np.inf
    w[0] = np.inf # fake value for neutrals -> essentially infinite trap
    return w


@numba.njit(cache=True)
def escape_rate_axial(Ni, kbTi, ri, V):
    """
    Computes the axial ion escape rates.

    Parameters
    ----------
    Ni : numpy.ndarray
        <1/m^3>
        Vector of ion densities.
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    ri : numpy.ndarray
        <1/s>
        Vector of total ion ion collision rates for each charge state.
    V : float or int
        <V>
        Trap depth.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Vector of axial ion loss rates for each charge state.

    """
    w = loss_frequency_axial(kbTi, V)
    return escape_rate(Ni, ri, w)


@numba.njit(cache=True)
def escape_rate_radial(Ni, kbTi, ri, Ai, V, B, r_dt):
    """
    Computes the radial ion escape rates.

    Parameters
    ----------
    Ni : numpy.ndarray
        <1/m^3>
        Vector of ion densities.
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    ri : numpy.ndarray
        <1/s>
        Vector of total ion ion collision rates for each charge state.
    Ai : float or int
        Ion mass number.
    V : float or int
        <V>
        Trap depth.
    B : float or int
        <T>
        Axial magnetic flux density.
    r_dt : float or int
        <m>
        Drift tube radius.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Vector of radial ion loss rates for each charge state.

    """
    w = loss_frequency_radial(kbTi, Ai, V, B, r_dt)
    return escape_rate(Ni, ri, w)


@numba.njit(cache=True)
def escape_rate(Ni, ri, w):
    """
    Generic escape rate - to be called by axial and radial escape

    Parameters
    ----------
    Ni : numpy.ndarray
        <1/m^3>
        Vector of ion densities.
    ri : numpy.ndarray
        <1/s>
        Vector of total ion ion collision rates for each charge state.
    w : numpy.ndarray
        <1/s>
        Vector of trap (loss) frequencies.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Vector of ion loss rates for each charge state.

    """
    esc = 3 / np.sqrt(2) * Ni * ri * np.exp(-w) / w
    esc[esc < 0] = 0 # this cleans neutrals and any other faulty stuff
    esc[Ni < MINIMAL_DENSITY] = 0
    return esc
