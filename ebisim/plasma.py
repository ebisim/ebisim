"""
This module contains functions for computing collission rates and related plasma parameters.
"""

from numba import njit, vectorize, float64, int64
import numpy as np

from .physconst import M_E, M_P, PI, EPS_0, Q_E, C_L, M_E_EV
from .physconst import MINIMAL_DENSITY


@njit(cache=True)
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


@vectorize([float64(float64, float64, float64, float64, float64, int64)], cache=True, nopython=True)
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
    qi : int
        Ion charge state.

    Returns
    -------
    float
        Ion electron coulomb logarithm.

    """
    Ni = Ni * 1e-6 # go from 1/m**3 to 1/cm**3
    Ne = Ne * 1e-6
    qqten = qi * qi * 10
    redkbTi = kbTi * M_E / (Ai * M_P)
    if   redkbTi <= kbTe  <= qqten:
        return 23. - np.log(Ne**0.5 * qi * kbTe**-1.5)
    elif redkbTi <= qqten <= kbTe:
        return 24. - np.log(Ne**0.5 / kbTe)
    elif kbTe <= redkbTi:
        return 16. - np.log(Ni**0.5 * kbTi**-1.5 * qi * qi * Ai)
    # The next case should not usually arise in any realistic situation but the solver may probe it
    # Hence it is purely a rough guess
    else: #(if qqten <= redkbTi <= kbTe)
        return 24. - np.log(Ne**0.5 / kbTe)


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, int64, int64)],
    cache=True, nopython=True
)
def clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    r"""
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
    qi : int
        Ion species "i" charge state.
    qj : int
        Ion species "j" charge state.

    Returns
    -------
    float
        Ion ion coulomb logarithm.

    Notes
    -----
    .. math::

        \lambda_{ij} = \lambda_{ji} = 23 - \ln \left(
            \frac{q_i q_j(\mu_i+\mu_j)}{\mu_i T_j+\mu_j T_i} \left(
                \frac{n_i q_i^2}{T_i} + \frac{n_j q_j^2}{T_j}
            \right)^{1/2}
        \right)

    """
    A = qi * qj * (Ai + Aj) / (Ai * kbTj + Aj * kbTi)
    B = (Ni * qi * qi / kbTi + Nj * qj * qj / kbTj) * 1e-6 # go from 1/m**3 to 1/cm**3
    return 23 - np.log(A * B**0.5)


@vectorize([float64(float64, float64, float64, float64, float64, int64)], cache=True, nopython=True)
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
    qi : int
        Ion charge state.

    Returns
    -------
    float
        <m^2>
        Coulomb cross section.

    """
    if qi == 0:
        return 0.
    v_e = electron_velocity(Ee)
    clog = clog_ei(Ni, Ne, kbTi, Ee, Ai, qi)
    return 4 * PI * (qi * Q_E * Q_E / (4 * PI * EPS_0 * M_E))**2 * clog / v_e**4


@vectorize(
    [float64(float64, float64, float64, float64, float64, float64, int64, int64)],
    cache=True, nopython=True
)
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
    qi : int
        Ion species "i" charge state.
    qj : int
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
        return 0.
    if qi == 0 or qj == 0:
        return 0.
    clog = clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj)
    kbTi_SI = kbTi * Q_E
    Mi = Ai * M_P
    const = 4 / 3 / (4 * PI * EPS_0)**2 * np.sqrt(2 * PI)
    return const * Nj * (qi * qj * Q_E * Q_E / Mi)**2 * (Mi/kbTi_SI)**1.5 * clog


@vectorize(
    [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def spitzer_heating(Ni, Ne, kbTi, Ee, Ai, qi):
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
        <eV/s>
        Vector of electron heating rate (temperature increase) for each charge state.

    """
    if Ni < MINIMAL_DENSITY:
        return 0.
    return Ne * electron_velocity(Ee) * 2 * M_E / (Ai * M_P) * Ee \
           * coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi)


@vectorize(
    [float64(float64, float64, float64, float64, float64)],
    cache=True, nopython=True
)
def collisional_thermalisation(kbTi, kbTj, Ai, Aj, rij):
    """
    Computes the collisional energy transfer rates for species "i" with respect to species "j".

    Parameters
    ----------
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
        <eV/s>
        Vector of temperature change rate for each charge state.

    """
    if kbTi <= 0 or kbTj <= 0:
        return 0
    else:
        return 2 * rij * Ai/Aj * (kbTj - kbTi) / \
                        (1 + (Ai * kbTj) / (Aj * kbTi))**1.5


@vectorize([float64(float64, int64, float64)], cache=True, nopython=True)
def loss_frequency_axial(kbTi, qi, V):
    """
    Computes the axial trap (loss) frequencies.

    Parameters
    ----------
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    qi : int
        Ion species "i" charge state.
    V : float or int
        <V>
        Trap depth.

    Returns
    -------
    numpy.ndarray
        <1/s>
        Vector of axial ion trap (loss) frequencies for each charge state.

    """
    if qi == 0 or kbTi <= 0:
        return np.inf # fake value for neutrals -> essentially infinite trap
    else:
        return qi * V / kbTi


@vectorize([float64(float64, int64, float64, float64, float64, float64)], cache=True, nopython=True)
def loss_frequency_radial(kbTi, qi, Ai, V, B, r_dt):
    """
    Radial trap (loss) frequencies.

    Parameters
    ----------
    kbTi : numpy.ndarray
        <eV>
        Vector of ion temperatures.
    qi : int
        Ion species "i" charge state.
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
    if qi == 0 or kbTi <= 0:
        return np.inf # fake value for neutrals -> essentially infinite trap
    else:
        return qi * (V + B * r_dt * np.sqrt(2 * kbTi * Q_E /(3*M_P*Ai))) / kbTi


@vectorize([float64(float64, float64, float64)], cache=True, nopython=True)
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
    if esc < 0 or Ni < MINIMAL_DENSITY:
        esc = 0.
    return esc


@vectorize([float64(float64, float64, int64, float64, float64)], cache=True, nopython=True)
def escape_rate_axial(Ni, kbTi, qi, ri, V):
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
    qi : float or int
        Ion species "i" charge state.
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
    w = loss_frequency_axial(kbTi, qi, V)
    return escape_rate(Ni, ri, w)


@vectorize(
    [float64(float64, float64, int64, float64, float64, float64, float64, float64)],
    cache=True, nopython=True
)
def escape_rate_radial(Ni, kbTi, qi, ri, Ai, V, B, r_dt):
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
    qi : int
        Ion species "i" charge state.
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
    w = loss_frequency_radial(kbTi, qi, Ai, V, B, r_dt)
    return escape_rate(Ni, ri, w)
