"""
This module contains functions for computing collission rates and related plasma parameters.
"""

from numba import njit, vectorize#, float64, int64
import numpy as np

from .physconst import M_E, M_P, PI, EPS_0, Q_E, C_L, M_E_EV
from .physconst import MINIMAL_DENSITY


@njit(cache=True)
def _erfc_approx(x):
    """
    Approximation of the complementary error function erfc on the interval (0, inf]
    http://users.auth.gr/users/9/3/028239/public_html/pdf/Q_Approxim.pdf
    https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    """
    return (1 - np.exp(-1.98*x))*np.exp(-x**2) / 2.0117351207777605 / x


@njit(cache=True)
def electron_velocity(e_kin):
    """
    Computes the electron velocity corresponding to a kinetic energy.

    Parameters
    ----------
    e_kin : float or numpy.ndarray
        <eV>
        Kinetic energy of the electron.

    Returns
    -------
    float or numpy.ndarray
        <m/s>
        Speed of the electron.

    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)


@vectorize(
    # [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def clog_ei(Ni, Ne, kbTi, kbTe, Ai, qi):
    """
    The coulomb logarithm for ion electron collisions.

    Parameters
    ----------
    Ni : float or numpy.ndarray
        <1/m^3>
        Ion density.
    Ne : float or numpy.ndarray
        <1/m^3>
        Electron density.
    kbTi : float or numpy.ndarray
        <eV>
        Ion temperature.
    kbTe : float or numpy.ndarray
        <eV>
        Electron temperature.
    Ai : float or numpy.ndarray
        Ion mass number.
    qi : int or numpy.ndarray
        Ion charge state.

    Returns
    -------
    float or numpy.ndarray
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
    # [float64(float64, float64, float64, float64, float64, float64, int64, int64)],
    cache=True, nopython=True
)
def clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    r"""
    The coulomb logarithm for ion ion collisions.

    Parameters
    ----------
    Ni : float or numpy.ndarray
        <1/m^3>
        Ion species "i" density.
    Nj : float or numpy.ndarray
        <1/m^3>
        Ion species "j" density.
    kbTi : float or numpy.ndarray
        <eV>
        Ion species "i" temperature.
    kbTj : float or numpy.ndarray
        <eV>
        Ion species "j" temperature.
    Ai : float or numpy.ndarray
        Ion species "i" mass number.
    Aj : float or numpy.ndarray
        Ion species "j" mass number.
    qi : int or numpy.ndarray
        Ion species "i" charge state.
    qj : int or numpy.ndarray
        Ion species "j" charge state.

    Returns
    -------
    float or numpy.ndarray
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


@vectorize(
    # [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi):
    """
    Computes the Coulomb cross section for elastic electron ion collisions

    Parameters
    ----------
    Ni : float or numpy.ndarray
        <1/m^3>
        Ion density.
    Ne : float or numpy.ndarray
        <1/m^3>
        Electron density.
    kbTi : float or numpy.ndarray
        <eV>
        Ion temperature.
    Ee : float or numpy.ndarray
        <eV>
        Electron kinetic energy.
    Ai : float or numpy.ndarray
        Ion mass number.
    qi : int or numpy.ndarray
        Ion charge state.

    Returns
    -------
    float or numpy.ndarray
        <m^2>
        Coulomb cross section.

    """
    if qi == 0:
        return 0.
    v_e = electron_velocity(Ee)
    clog = clog_ei(Ni, Ne, kbTi, Ee, Ai, qi)
    return 4 * PI * (qi * Q_E * Q_E / (4 * PI * EPS_0 * M_E))**2 * clog / v_e**4


@vectorize(
    # [float64(float64, float64, float64, float64, float64, float64, int64, int64)],
    cache=True, nopython=True
)
def ion_coll_rate(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    """
    Collision rates for ions species "i" and target species "j"

    Parameters
    ----------
    Ni : float or numpy.ndarray
        <1/m^3>
        Ion species "i" density.
    Nj : float or numpy.ndarray
        <1/m^3>
        Ion species "j" density.
    kbTi : float or numpy.ndarray
        <eV>
        Ion species "i" temperature.
    kbTj : float or numpy.ndarray
        <eV>
        Ion species "j" temperature.
    Ai : float or numpy.ndarray
        Ion species "i" mass number.
    Aj : float or numpy.ndarray
        Ion species "j" mass number.
    qi : int or numpy.ndarray
        Ion species "i" charge state.
    qj : int or numpy.ndarray
        Ion species "j" charge state.

    Returns
    -------
    float or numpy.ndarray
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
    # [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def spitzer_heating(Ni, Ne, kbTi, Ee, Ai, qi):
    """
    Computes the heating rates due to elastic electron ion collisions ('Spitzer Heating')

    Parameters
    ----------
    Ni : float or numpy.ndarray
        <1/m^3>
        Vector of ion densities.
    Ne : float or numpy.ndarray
        <1/m^3>
        Electron density.
    kbTi : float or numpy.ndarray
        <eV>
        Vector of ion temperatures.
    Ee : float or numpy.ndarray
        <eV>
        Electron kinetic energy.
    Ai : float or numpy.ndarray
        Ion mass number.

    Returns
    -------
    float or numpy.ndarray
        <eV/s>
        Vector of electron heating rate (temperature increase) for each charge state.

    """
    if Ni < MINIMAL_DENSITY:
        return 0.
    return Ne * electron_velocity(Ee) * 2 * M_E / (Ai * M_P) * Ee \
           * coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi)


@vectorize(
    # [float64(float64, float64, float64, float64, float64)],
    cache=True, nopython=True
)
def collisional_thermalisation(kbTi, kbTj, Ai, Aj, rij):
    """
    Computes the collisional energy transfer rates for species "i" with respect to species "j".

    Parameters
    ----------
    kbTi : float or numpy.ndarray
        <eV>
        Vector of ion species "i" temperatures.
    kbTj : float or numpy.ndarray
        <eV>
        Vector of ion species "j" temperatures.
    Ai : float or numpy.ndarray
        Ion species "i" mass number.
    Aj : float or numpy.ndarray
        Ion species "j" mass number.
    rij : float or numpy.ndarray
        <1/s>
        Collision rate matrix for the ions (cf. ion_coll_mat).

    Returns
    -------
    float or numpy.ndarray
        <eV/s>
        Vector of temperature change rate for each charge state.

    """
    if kbTi <= 0 or kbTj <= 0:
        return 0
    else:
        return 2 * rij * Ai/Aj * (kbTj - kbTi) / \
                        (1 + (Ai * kbTj) / (Aj * kbTi))**1.5


@vectorize(
    # [float64(float64, int64, float64)],
    cache=True, nopython=True
)
def trapping_strength_axial(kbTi, qi, V):
    """
    Computes the axial trapping strenghts.

    Parameters
    ----------
    kbTi : float or numpy.ndarray
        <eV>
        Vector of ion temperatures.
    qi : int or numpy.ndarray
        Ion species "i" charge state.
    V : float or numpy.ndarray
        <V>
        Trap depth.

    Returns
    -------
    float or numpy.ndarray
        <1/s>
        Vector of axial ion trapping strenghts for each charge state.

    """
    # if qi == 0 or kbTi <= 0:
    # if kbTi <= 0:
    #     return np.inf # fake value for neutrals -> essentially infinite trap
    # else:
    w = qi * V / kbTi
    if w < 1:
        return 0.
    return w



@vectorize(
    # [float64(float64, int64, float64, float64, float64, float64)],
    cache=True, nopython=True
)
def trapping_strength_radial(kbTi, qi, Ai, V, B, r_dt):
    """
    Radial trapping strenghts.

    Parameters
    ----------
    kbTi : float or numpy.ndarray
        <eV>
        Vector of ion temperatures.
    qi : int or numpy.ndarray
        Ion species "i" charge state.
    Ai : float or numpy.ndarray
        Ion mass number.
    V : float or numpy.ndarray
        <V>
        Trap depth.
    B : float or numpy.ndarray
        <T>
        Axial magnetic flux density.
    r_dt : float or numpy.ndarray
        <m>
        Drift tube radius.

    Returns
    -------
    float or numpy.ndarray
        <1/s>
        Vector of radial ion trapping strenghts for each charge state.

    """
    # if qi == 0 or kbTi <= 0:
    # if kbTi <= 0:
    #     return np.inf # fake value for neutrals -> essentially infinite trap
    # else:
    w = qi * (V + B * r_dt * np.sqrt(2 * kbTi * Q_E /(3*M_P*Ai))) / kbTi
    if w < 1:
        return 0.
    return w


@vectorize(
    # [float64(float64, float64, float64)],
    cache=True, nopython=True
)
def collisional_escape_rate(ri, w):
    """
    Generic escape rate - to be called by axial and radial escape

    Parameters
    ----------
    ri : float or numpy.ndarray
        <1/s>
        Vector of total ion ion collision rates for each charge state.
    w : float or numpy.ndarray
        <1/s>
        Vector of trap (loss) frequencies.

    Returns
    -------
    float or numpy.ndarray
        <1/s>
        Vector of ion loss rates for each charge state.

    """
    if w == 0:
        esc = ri
    else:
        esc = 3 / np.sqrt(2) * ri * np.exp(-w) / w
    # if esc < 0:
    #     esc = 0.
    return esc


_INVSQRTPI = 1/np.sqrt(PI)
@vectorize()
def roundtrip_escape(w):
    """
    Computes the fraction of ions that have enough kinetic energy to overcome the trap represented
    by w. Also returns the "temperature factor" which tells how much hotter those ions are
    then the complete population average

    Parameters
    ----------
    w : numpy.ndarray
        The axial or radial trapping strength

    Returns
    -------
    free_fraction : numpy.ndarray
        Fraction of ions with enough energy to pass trapping barrier
    temperature_factor : numpy.ndarray
        Factor by which the escaping ions are hotter than the whole ensemble
    """
    w = w * 3 #Degrees of freedom pushes cut-on (minimal) energy three times higher
    _sqrtw = np.sqrt(w)
    _erfc = _erfc_approx(_sqrtw)
    free_fraction = _erfc + 2*_INVSQRTPI*_sqrtw*np.exp(-w)
    free_fraction[w == 0] = 1
    temperature_factor = (1.5*_erfc + _INVSQRTPI*_sqrtw*np.exp(-w)*(2*w+3))/free_fraction
    temperature_factor[free_fraction == 0] = 1
    temperature_factor[w == 0] = 1
    return free_fraction, temperature_factor


# @vectorize(
#     # [float64(float64, float64, int64, float64, float64)],
#     cache=True, nopython=True
# )
# def escape_rate_axial(Ni, kbTi, qi, ri, V):
#     """
#     Computes the axial ion escape rates.

#     Parameters
#     ----------
#     Ni : float or numpy.ndarray
#         <1/m^3>
#         Vector of ion densities.
#     kbTi : float or numpy.ndarray
#         <eV>
#         Vector of ion temperatures.
#     qi : int or int
#         Ion species "i" charge state.
#     ri : float or numpy.ndarray
#         <1/s>
#         Vector of total ion ion collision rates for each charge state.
#     V : float or numpy.ndarray
#         <V>
#         Trap depth.

#     Returns
#     -------
#     float or numpy.ndarray
#         <1/s>
#         Vector of axial ion loss rates for each charge state.

#     """
#     w = loss_frequency_axial(kbTi, qi, V)
#     return escape_rate(Ni, ri, w)


# @vectorize(
#     # [float64(float64, float64, int64, float64, float64, float64, float64, float64)],
#     cache=True, nopython=True
# )
# def escape_rate_radial(Ni, kbTi, qi, ri, Ai, V, B, r_dt):
#     """
#     Computes the radial ion escape rates.

#     Parameters
#     ----------
#     Ni : float or numpy.ndarray
#         <1/m^3>
#         Vector of ion densities.
#     kbTi : float or numpy.ndarray
#         <eV>
#         Vector of ion temperatures.
#     qi : int or numpy.ndarray
#         Ion species "i" charge state.
#     ri : float or numpy.ndarray
#         <1/s>
#         Vector of total ion ion collision rates for each charge state.
#     Ai : float or numpy.ndarray
#         Ion mass number.
#     V : float or numpy.ndarray
#         <V>
#         Trap depth.
#     B : float or numpy.ndarray
#         <T>
#         Axial magnetic flux density.
#     r_dt : float or numpy.ndarray
#         <m>
#         Drift tube radius.

#     Returns
#     -------
#     float or numpy.ndarray
#         <1/s>
#         Vector of radial ion loss rates for each charge state.

#     """
#     w = loss_frequency_radial(kbTi, qi, Ai, V, B, r_dt)
#     return escape_rate(Ni, ri, w)
