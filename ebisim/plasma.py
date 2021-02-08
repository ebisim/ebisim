"""
This module contains functions for computing collission rates and related plasma parameters.
"""
import logging
from numba import njit, vectorize  # , float64, int64
import numpy as np

from .physconst import M_E, M_P, PI, EPS_0, Q_E, C_L, M_E_EV
from .physconst import MINIMAL_N_3D

logger = logging.getLogger(__name__)

logger.debug("Defining _erfc_appprox.")


@njit(cache=True)
def _erfc_approx(x):
    """
    Approximation of the complementary error function erfc on the interval (0, inf]
    http://users.auth.gr/users/9/3/028239/public_html/pdf/Q_Approxim.pdf
    https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
    """
    return (1 - np.exp(-1.98*x))*np.exp(-x**2) / 2.0117351207777605 / x


logger.debug("Defining electron_velocity.")


@njit(cache=True)
def electron_velocity(e_kin):
    r"""
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

    Notes
    -----
    .. math::
        v_e = c\sqrt{1-\left(\dfrac{m_e c^2}{m_e c^2 + E_e}\right)^2}
    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)


logger.debug("Defining clog_ei.")


@vectorize(
    # [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def clog_ei(Ni, Ne, kbTi, kbTe, Ai, qi):
    r"""
    The coulomb logarithm for ion electron collisions [CLOGEI]_.

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

    References
    ----------
    .. [CLOGEI] "NRL Plasma Formulary",
           J. D. Huba,
           Naval Research Laboratory (2019),
           https://www.nrl.navy.mil/ppd/sites/www.nrl.navy.mil.ppd/files/pdfs/NRL_Formulary_2019.pdf

    Notes
    -----
    .. math::
            \lambda_{ei} = 23 - \ln\left(N_e^{1/2} (T_e)^{-3/2} q_i\right)
                 \quad\text{if } T_i m_e/m_i < T_e < 10 q_i^2\text{ eV}

    .. math::
            \lambda_{ei} = 24 - \ln\left(N_e^{1/2} (T_e)^{-1}\right)
                 \quad\text{if } T_i m_e/m_i < 10 q_i^2\text{ eV} < T_e

    .. math::
            \lambda_{ei} = 16 - \ln\left(N_i^{1/2} (T_i)^{-3/2} q_i^2 \mu_i\right)
                 \quad\text{if } T_e < T_i m_e/m_i

    .. math::
            \left[\lambda_{ei} = 24 - \ln\left(N_e^{1/2} (T_e)^{-1/2}\right)
                 \quad\text{else (fallback)} \right]

    In these formulas N and T are given in cm^-3 and eV respectively.
    As documented, the function itself expects the density to be given in 1/m^3.

    """
    Ni = Ni * 1e-6  # go from 1/m**3 to 1/cm**3
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
    else:  # (if qqten <= redkbTi <= kbTe)
        return 24. - np.log(Ne**0.5 / kbTe)


logger.debug("Defining clog_ii.")


@vectorize(
    # [float64(float64, float64, float64, float64, float64, float64, int64, int64)],
    cache=True, nopython=True
)
def clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    r"""
    The coulomb logarithm for ion ion collisions [CLOGII]_.

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

    References
    ----------
    .. [CLOGII] "NRL Plasma Formulary",
           J. D. Huba,
           Naval Research Laboratory (2019),
           https://www.nrl.navy.mil/ppd/sites/www.nrl.navy.mil.ppd/files/pdfs/NRL_Formulary_2019.pdf

    Notes
    -----
    .. math::
        \lambda_{ij} = \lambda_{ji} = 23 - \ln \left(
            \frac{q_i q_j(\mu_i+\mu_j)}{\mu_i T_j+\mu_j T_i} \left(
                \frac{n_i q_i^2}{T_i} + \frac{n_j q_j^2}{T_j}
            \right)^{1/2}
        \right)

    In these formulas N and T are given in cm^-3 and eV respectively.
    As documented, the function itself expects the density to be given in 1/m^3.

    """
    A = qi * qj * (Ai + Aj) / (Ai * kbTj + Aj * kbTi)
    B = (Ni * qi * qi / kbTi + Nj * qj * qj / kbTj) * 1e-6  # go from 1/m**3 to 1/cm**3
    return 23 - np.log(A * B**0.5)


logger.debug("Defining coulomb_xs.")


@vectorize(
    # [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi):
    r"""
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

    Notes
    -----
    .. math::
        \sigma_i = 4 \pi \left( \dfrac{q_i q_e^2}{4\pi\epsilon_0 m_e} \right)^2
        \dfrac{\ln \Lambda_{ei}}{v_e^4}

    """
    if qi == 0:
        return 0.
    v_e = electron_velocity(Ee)
    clog = clog_ei(Ni, Ne, kbTi, Ee, Ai, qi)
    return 4 * PI * (qi * Q_E * Q_E / (4 * PI * EPS_0 * M_E))**2 * clog / v_e**4


logger.debug("Defining ion_coll_rate.")


@vectorize(
    # [float64(float64, float64, float64, float64, float64, float64, int64, int64)],
    cache=True, nopython=True
)
def ion_coll_rate(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    r"""
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

    Notes
    -----
    .. math::
        \nu_{ij} = \dfrac{1}{(4\pi\epsilon_0)^2}\dfrac{4\sqrt{2\pi}}{3}N_j\left(
            \dfrac{q_i q_j q_e^2}{m_i}
        \right)^2 \left(\dfrac{m_i}{k_B T_i}\right)^{3/2} \ln \Lambda_{ij}

    """
    # Artifically clamp collision rate to zero if either density is very small
    # This is a reasonable assumption and prevents instabilities when calling the coulomb logarithm
    if Ni <= MINIMAL_N_3D or Nj <= MINIMAL_N_3D or kbTi <= 0 or kbTj <= 0:
        return 0.
    if qi == 0 or qj == 0:
        return 0.
    clog = clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj)
    kbTi_SI = kbTi * Q_E
    Mi = Ai * M_P
    const = 4 / 3 / (4 * PI * EPS_0)**2 * np.sqrt(2 * PI)
    return np.maximum(const * Nj * (qi * qj * Q_E * Q_E / Mi)**2 * (Mi/kbTi_SI)**1.5 * clog, 0)


logger.debug("Defining spitzer_heating.")


@vectorize(
    # [float64(float64, float64, float64, float64, float64, int64)],
    cache=True, nopython=True
)
def spitzer_heating(Ni, Ne, kbTi, Ee, Ai, qi):
    r"""
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

    Notes
    -----
    .. math::
        \left(\dfrac{d k_B T_i}{d t}\right)^{\text{Spitzer}} =
        \dfrac{2}{3} N_e v_e \sigma_i 2 \dfrac{m_e}{m_i} E_e

    where sigma_i is the cross section for Coulomb collisions (cf. ebisim.plasma.coulomb_xs).

    """
    if Ni < MINIMAL_N_3D:
        return 0.
    return np.maximum(0, 2/3 * Ne * electron_velocity(Ee) * 2 * M_E / (Ai * M_P) * Ee \
                         * coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi))


logger.debug("Defining collisional_thermalisation.")


@vectorize(
    # [float64(float64, float64, float64, float64, float64)],
    cache=True, nopython=True
)
def collisional_thermalisation(kbTi, kbTj, Ai, Aj, nuij):
    r"""
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
    nuij : float or numpy.ndarray
        <1/s>
        Collision rate matrix for the ions (cf. ion_coll_mat).

    Returns
    -------
    float or numpy.ndarray
        <eV/s>
        Vector of temperature change rate for each charge state.

    Notes
    -----
    .. math::
        \left(\dfrac{d k_B T_i}{d t}\right)_j =
        2 \nu_{ij} \dfrac{m_i}{m_j} \dfrac{k_B T_j - k_B T_i}{(1 + m_i k_B T_j / m_j k_B T_i)^{3/2}}

    """
    if kbTi <= 0 or kbTj <= 0:
        return 0
    else:
        return 2 * nuij * Ai/Aj * (kbTj - kbTi) / \
                        (1 + (Ai * kbTj) / (Aj * kbTi))**1.5


logger.debug("Defining trapping_strength_axial.")


@vectorize(
    # [float64(float64, int64, float64)],
    cache=True, nopython=True
)
def trapping_strength_axial(kbTi, qi, V):
    r"""
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

    Notes
    -----
    .. math::
        \omega_i^{ax} = \dfrac{q_i V_{ax}}{k_B T_i}

    """
    # if qi == 0 or kbTi <= 0:
    # if kbTi <= 0:
    #     return np.inf # fake value for neutrals -> essentially infinite trap
    # else:
    w = qi * V / kbTi
    # if w < 1e-10:
    #     return 0.
    return w


logger.debug("Defining trapping_strength_radial.")


@vectorize(
    # [float64(float64, int64, float64, float64, float64, float64)],
    cache=True, nopython=True
)
def trapping_strength_radial(kbTi, qi, Ai, V, B, r_dt):
    r"""
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

    Notes
    -----
    .. math::
        \omega_i^{rad} =
        \dfrac{q_i \left(V_{rad} + B r_{dt} \sqrt{2 k_B T_i/(3 m_i)} \right)}{k_B T_i}

    """
    # if qi == 0 or kbTi <= 0:
    # if kbTi <= 0:
    #     return np.inf # fake value for neutrals -> essentially infinite trap
    # else:
    w = qi * (V + B * r_dt * np.sqrt(2 * kbTi * Q_E /(3*M_P*Ai))) / kbTi
    # if w < 1e-10:
    #     return 0.
    return w


logger.debug("Defining collisional_escape_rate.")


@vectorize(
    # [float64(float64, float64, float64)],
    cache=True, nopython=True
)
def collisional_escape_rate(nui, w):
    r"""
    Generic escape rate - to be called by axial and radial escape

    Parameters
    ----------
    nui : float or numpy.ndarray
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

    Notes
    -----
    .. math::
        \dfrac{3}{\sqrt{2}} \nu_i \dfrac{e^{-\omega_i}}{\omega_i}

    """
    if w <= 0:
        esc = nui
    else:
        esc = 3 / np.sqrt(2) * nui * np.exp(-w) / w
    # if esc < 0:
    #     esc = 0.
    return esc


logger.debug("Defining roundtrip_escape.")
_INVSQRTPI = 1/np.sqrt(PI)


@njit(cache=True)
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
    w = w * 3  # Degrees of freedom pushes cut-on (minimal) energy three times higher
    _sqrtw = np.sqrt(w)
    _erfc = _erfc_approx(_sqrtw)
    free_fraction = _erfc + 2*_INVSQRTPI*_sqrtw*np.exp(-w)
    free_fraction[w <= 0] = 1
    temperature_factor = (1.5*_erfc + _INVSQRTPI*_sqrtw*np.exp(-w)*(2*w+3))/free_fraction
    temperature_factor[free_fraction == 0] = 1
    temperature_factor[w <= 0] = 1
    return free_fraction, temperature_factor


# logger.debug("Defining escape_rate_axial.")
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


# logger.debug("Defining escape_rate_radial.")
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
