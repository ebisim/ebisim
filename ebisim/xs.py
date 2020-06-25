"""
This module contains functions to compute the cross sections for various ionisation and
recombination processes.
"""

import logging
logger = logging.getLogger(__name__)
import math
import numpy as np
import numba

from .physconst import RY_EV, ALPHA, PI, COMPT_E_RED, M_E_EV

logger.debug("Defining _normpdf.")
@numba.njit(cache=True)
def _normpdf(x, mu, sigma):
    """
    The pdf of the normal distribution f_mu,sigma(x).


    Parameters
    ----------
    x : numpy.ndarray
        Function argument
    mu : numpy.ndarray
        Mean of the distribution
    sigma : numpy.ndarray
        Standard deviation of the distribution

    Returns
    -------
    numpy.ndarray
        Value of the normal PDF evaluated elementwise on the input arrays
    """
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (2 * PI * sigma**2)**0.5


logger.debug("Defining cxxs.")
@numba.njit(cache=True)
def cxxs(q, ip):
    """
    Single charge exchange cross section according to the Mueller Salzborn formula

    Parameters
    ----------
    q : int
        Charge state of the colliding ion
    ip : float
        <eV>
        Ionisation potential of the collision partner (neutral gas)

    Returns
    -------
    float
        <m^2>
        Charge exchange cross section
    """
    return 1.43e-16 * q**1.17 * ip**-2.76

logger.debug("Defining eixs_vec.")
@numba.njit(cache=True)
def eixs_vec(element, e_kin):
    """
    Electron ionisation cross section according to a simplified version of the models given in
    [Lotz1967]_.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.

    Returns
    -------
    numpy.ndarray
        <m^2>
        The cross sections for each individual charge state, where the array-index corresponds
        to the charge state, i.e. out[q] ~ cross section of q+ ion.

    References
    ----------
    .. [Lotz1967] "An empirical formula for the electron-impact ionization cross-section",
           W. Lotz,
           Zeitschrift Für Physik, 206(2), 205–211 (1967),
           https://doi.org/10.1007/BF01325928

    See Also
    --------
    ebisim.xs.eixs_mat : Similar method with different output format.

    """
    css = element.e_bind.shape[0]
    shells = element.e_bind.shape[1]
    avail_factors = element.ei_lotz_a.shape[0]
    xs_vec = np.zeros(css + 1)
    t = e_kin / M_E_EV
    for cs in range(css):
        xs = 0
        for shell in range(shells):
            e = element.e_bind[cs, shell]
            n = element.e_cfg[cs, shell]
            if n > 0 and e_kin > e:
                i = e / M_E_EV
                grys_fact = (2+i)/(2+t) * ((1+t) / (1+i))**2 \
                            * (((i+t) * (2+t) * (1+i)**2) / (t * (2+t) * (1+i)**2 + i * (2+i)))**1.5
                if cs < avail_factors:
                    a = element.ei_lotz_a[cs, shell]
                    b = element.ei_lotz_b[cs, shell]
                    c = element.ei_lotz_c[cs, shell]
                    xs += grys_fact * a * n * math.log(e_kin / e) / (e_kin * e) \
                          * (1 - b*np.exp(-c*(e_kin/e - 1)))
                else:
                    xs += grys_fact * 4.5e-18 * n * math.log(e_kin / e) / (e_kin * e)
        xs_vec[cs] = xs
    return xs_vec


logger.debug("Defining eixs_mat.")
@numba.njit(cache=True)
def eixs_mat(element, e_kin):
    """
    Electron ionisation cross section.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.

    Returns
    -------
    numpy.array
        <m^2>
        The cross sections for each individual charge state, arranged in a matrix suitable
        for implementation of a rate equation like dN/dt = j * xs_matrix dot N.
        out[q, q]   = - cross section of q+ ion
        out[q+1, q] = + cross section of (q+1)+ ion

    See Also
    --------
    ebisim.xs.eixs_vec : Similar method with different output format.

    """
    xs = eixs_vec(element, e_kin)
    return np.diag(xs[:-1], -1) - np.diag(xs)


logger.debug("Defining rrxs_vec.")
@numba.njit(cache=True)
def rrxs_vec(element, e_kin):
    """
    Radiative recombination cross section according to [Kim1983]_.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.

    Returns
    -------
    numpy.ndarray
        <m^2>
        The cross sections for each individual charge state, where the array-index corresponds
        to the charge state, i.e. out[q] ~ cross section of q+ ion.

    References
    ----------
    .. [Kim1983] "Direct radiative recombination of electrons with atomic ions:
           Cross sections and rate coefficients",
           Young Soon Kim and R. H. Pratt,
           Phys. Rev. A 27, 2913 (1983),
           https://journals.aps.org/pra/abstract/10.1103/PhysRevA.27.2913

    See Also
    --------
    ebisim.xs.rrxs_mat : Similar method with different output format.

    """
    chi = 2 * element.rr_z_eff**2 * RY_EV / e_kin
    xs = 8 * PI * ALPHA / (3 * np.sqrt(3)) * COMPT_E_RED**2 * \
            chi * np.log(1 + chi/(2 * element.rr_n_0_eff**2))
    xs[0] = 0
    return xs


logger.debug("Defining rrxs_mat.")
@numba.njit(cache=True)
def rrxs_mat(element, e_kin):
    """
    Radiative recombination cross section.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.

    Returns
    -------
    numpy.array
        <m^2>
        The cross sections for each individual charge state, arranged in a matrix suitable
        for implementation of a rate equation like dN/dt = j * xs_matrix dot N.
        out[q, q]   = - cross section of q+ ion
        out[q, q+1] = + cross section of (q+1)+ ion

    See Also
    --------
    ebisim.xs.rrxs_vec : Similar method with different output format.

    """
    xs = rrxs_vec(element, e_kin)
    return np.diag(xs[1:], 1) - np.diag(xs)


logger.debug("Defining drxs_vec.")
@numba.njit(cache=True)
def drxs_vec(element, e_kin, fwhm):
    """
    Dielectronic recombination cross section.
    The cross sections are estimated by weighing the strength of each transition with the
    profile of a normal Gaussian distribution. This simulates the effective spreading of the
    resonance peaks due to the energy spread of the electron beam

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.
    fwhm : float
        <eV>
        Energy spread to apply for the resonance smearing, expressed in terms of
        full width at half maximum.

    Returns
    -------
    numpy.ndarray
        <m^2>
        The cross sections for each individual charge state, where the array-index corresponds
        to the charge state, i.e. out[q] ~ cross section of q+ ion.


    See Also
    --------
    ebisim.xs.drxs_mat : Similar method with different output format.

    """
    xs_vec = np.zeros(element.z + 1)
    if element.dr_cs.size > 0:
        sig = fwhm/2.35482 # 2.35482approx.(2*np.sqrt(2*np.log(2)))
        tmp = element.dr_strength * _normpdf(e_kin, element.dr_e_res, sig)*1e-24
        for k in range(element.dr_cs.size):
            # cs = int(element.dr_cs[k])
            xs_vec[element.dr_cs[k]] = xs_vec[element.dr_cs[k]] + tmp[k]
    return xs_vec


logger.debug("Defining drxs_mat.")
@numba.njit(cache=True)
def drxs_mat(element, e_kin, fwhm):
    """
    Dielectronic recombination cross section.
    The cross sections are estimated by weighing the strength of each transition with the
    profile of a normal Gaussian distribution. This simulates the effective spreading of the
    resonance peaks due to the energy spread of the electron beam

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : float
        <eV>
        Kinetic energy of the impacting electron.
    fwhm : float
        <eV>
        Energy spread to apply for the resonance smearing, expressed in terms of
        full width at half maximum.

    Returns
    -------
    numpy.array
        <m^2>
        The cross sections for each individual charge state, arranged in a matrix suitable
        for implementation of a rate equation like dN/dt = j * xs_matrix dot N.
        out[q, q]   = - cross section of q+ ion
        out[q, q+1] = + cross section of (q+1)+ ion

    See Also
    --------
    ebisim.xs.drxs_vec : Similar method with different output format.

    """
    xs = drxs_vec(element, e_kin, fwhm)
    return np.diag(xs[1:], 1) - np.diag(xs)


logger.debug("Defining precompute_rr_quantities.")
@numba.njit(cache=True)
def precompute_rr_quantities(e_cfg, shell_n):
    """
    Precomputes the effective valence shell and nuclear charge for all charge states,
    as required for the computation of radiative recombinations cross sections.
    According to the procedure described in [Kim1983a]_.

    This function is primarily meant for internal use inside the ebisim.get_element() function.

    Parameters
    ----------
    e_cfg : numpy.ndarray
        Matrix holding the number of electrons in each shell.
        The row index corresponds to the charge state, the columns to different subshells
    shell_n : numpy.ndarray
        Array holding the main quantum number n corresponding to each shell listed in e_cfg

    Returns
    -------
    rr_z_eff : numpy.ndarray
        Array holding the effective nuclear charge for each charge state,
        where the array-index corresponds to the charge state.
    rr_n_0_eff : numpy.ndarray
        Array holding the effective valence shell number for each charge state,
        where the array-index corresponds to the charge state.

    References
    ----------
    .. [Kim1983a] "Direct radiative recombination of electrons with atomic ions:
           Cross sections and rate coefficients",
           Young Soon Kim and R. H. Pratt,
           Phys. Rev. A 27, 2913 (1983),
           https://journals.aps.org/pra/abstract/10.1103/PhysRevA.27.2913

    See Also
    --------
    ebisim.xs.rrxs_vec
    ebisim.xs.rrxs_mat

    """
    z = e_cfg.shape[0]
    shell_n = shell_n[:e_cfg.shape[1]] # Crop shell_n to the shells described in e_cfg

    n_0 = np.zeros(z + 1)
    occup = np.zeros(z + 1)

    # Determine, for each charge state, the valence shell (n_0),
    # and the number of electrons in it (occup)
    # Fully ionised
    n_0[z] = 1
    occup[z] = 0
    # All other charge states
    for cs in range(z):
        conf = e_cfg[cs, :]
        n_0[cs] = np.max(shell_n[np.nonzero(conf)])
        occup[cs] = np.sum(np.extract((shell_n == n_0[cs]), conf))

    w_n0 = (2 * n_0**2 - occup) / (2 * n_0**2)
    rr_n_0_eff = n_0 + (1 - w_n0) - 0.3
    rr_z_eff = (z + np.arange(z + 1)) / 2

    return rr_z_eff, rr_n_0_eff


logger.debug("Defining lookup_lotz_factors.")
def lookup_lotz_factors(e_cfg, shellorder):
    """
    Analyses the shell structure of each charge state and looks up the correct factors for
    the Lotz formula.

    This function is primarily meant for internal use inside the ebisim.get_element() function
    and the results are consumed during the Electron Ionisation (EI) cross section computations.

    Parameters
    ----------
    e_cfg : numpy.ndarray
        Matrix holding the number of electrons in each shell.
        The row index corresponds to the charge state, the columns to different subshells
    shellorder : numpy.ndarray
        Tuple containing the names of all shells in the same order as they appear in 'e_cfg'

    Returns
    -------
    ei_lotz_a : numpy.ndarray
        Array holding 'Lotz' factor 'a' for each occupied shell in 'e_cfg' up to a certain
        charge state.
    ei_lotz_b : numpy.ndarray
        Array holding 'Lotz' factor 'b' for each occupied shell in 'e_cfg' up to a certain
        charge state.
    ei_lotz_b : numpy.ndarray
        Array holding 'Lotz' factor 'c' for each occupied shell in 'e_cfg' up to a certain
        charge state.

    See Also
    --------
    ebisim.xs.eixs_vec
    ebisim.xs.eixs_mat

    """
    z = e_cfg.shape[0]
    cols = e_cfg.shape[1]

    if z > 20: # No specific data available, use factors for neutral to 1+ ionisation
        ei_lotz_a = np.zeros((1, cols))
        ei_lotz_b = np.zeros((1, cols))
        ei_lotz_c = np.zeros((1, cols))

        for i in range(cols):
            shell = shellorder[i]
            n = int(shell[0]) # main quantum number
            l = shell[1] # angular momentum character
            s = shell[2] if len(shell) > 2 else None
            shell_stub = shell[:2]

            # We need to determine, for each column, whether the electrons in another column
            # also need to be counted because they have the same n and l
            if l == "s":
                i2 = None
            elif s == "-":
                if shell_stub == "7p": #7p+ does not exist in currently used data
                    i2 = None
                else:
                    i2 = shellorder.index(shell_stub + "+")
            elif s == "+":
                i2 = shellorder.index(shell_stub + "-")

            n_e = e_cfg[0, i]
            if i2 and i2 < cols:
                n_e += e_cfg[0, i2]

            if n_e == 0:
                a = b = c = 0
            else:
                if (l in ["s", "p"] and n > 3) or (l == "d" and n > 4) or (l == "f"):
                    nstr = "n"
                else:
                    nstr = "" + str(int(n))
                a, b, c = _LOTZ_NEUTRAL_TABLE[nstr + l + str(int(n_e))]

            ei_lotz_a[0, i] = a * 1.0e-18
            ei_lotz_b[0, i] = b
            ei_lotz_c[0, i] = c

    else:
        table = _LOTZ_ADVANCED_TABLE[z]
        ncs = len(table.keys())
        ei_lotz_a = np.ones((ncs, cols)) * 4.5e-18
        ei_lotz_b = np.zeros((ncs, cols))
        ei_lotz_c = np.zeros((ncs, cols))

        for cs in range(ncs):
            for i in range(cols):
                shell_stub = shellorder[i][:2]

                if shell_stub in table[cs]:
                    a, b, c = table[cs][shell_stub]
                    ei_lotz_a[cs, i] = a * 1.0e-18
                    ei_lotz_b[cs, i] = b
                    ei_lotz_c[cs, i] = c

    ei_lotz_a.setflags(write=False)
    ei_lotz_b.setflags(write=False)
    ei_lotz_c.setflags(write=False)
    return ei_lotz_a, ei_lotz_b, ei_lotz_c


logger.debug("Defining eixs_energyscan.")
@numba.njit(cache=True)
def eixs_energyscan(element, e_kin=None, n=1000):
    """
    Creates an array of EI cross sections for varying electron energies.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : None or numpy.ndarray, optional
        <eV>
        If e_kin is None, the range of sampling energies is chosen based on the binding enrgies of
        the element and energies are sampled on a logscale. If e_kin is an array with 2 elements,
        they are interpreted as the minimum and maximum sampling energy.
        If e_kin is an array with more than two values, the energies are taken as the sampling
        energies directly, by default None.
    n : int, optional
        The number of energy sampling points, if the sampling locations are not supplied by the
        user, by default 1000.

    Returns
    -------
    e_samp : numpy.ndarray
        <eV>
        Array holding the sampling energies
    xs_scan : numpy.ndarray
        <m^2>
        Array holding the cross sections, where the row index corresponds to the charge state
        and the columns correspond to the different sampling energies

    See Also
    --------
    ebisim.xs.rrxs_energyscan
    ebisim.xs.drxs_energyscan

    """
    e_samp = _eirr_e_samp(element, e_kin, n)
    xs_scan = np.zeros((element.z + 1, len(e_samp)))
    for ind, ek in enumerate(e_samp):
        xs_scan[:, ind] = eixs_vec(element, ek)
    return e_samp, xs_scan


logger.debug("Defining rrxs_energyscan.")
@numba.njit(cache=True)
def rrxs_energyscan(element, e_kin=None, n=1000):
    """
    Creates an array of RR cross sections for varying electron energies.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : None or numpy.ndarray, optional
        <eV>
        If e_kin is None, the range of sampling energies is chosen based on the binding enrgies of
        the element and energies are sampled on a logscale. If e_kin is an array with 2 elements,
        they are interpreted as the minimum and maximum sampling energy.
        If e_kin is an array with more than two values, the energies are taken as the sampling
        energies directly, by default None.
    n : int, optional
        The number of energy sampling points, if the sampling locations are not supplied by the
        user, by default 1000.

    Returns
    -------
    e_samp : numpy.ndarray
        <eV>
        Array holding the sampling energies
    xs_scan : numpy.ndarray
        <m^2>
        Array holding the cross sections, where the row index corresponds to the charge state
        and the columns correspond to the different sampling energies

    See Also
    --------
    ebisim.xs.eixs_energyscan
    ebisim.xs.drxs_energyscan

    """
    e_samp = _eirr_e_samp(element, e_kin, n)
    xs_scan = np.zeros((element.z + 1, len(e_samp)))
    for ind, ek in enumerate(e_samp):
        xs_scan[:, ind] = rrxs_vec(element, ek)
    return e_samp, xs_scan


logger.debug("Defining _eirr_e_samp.")
@numba.njit(cache=True)
def _eirr_e_samp(element, e_kin, n):
    """
    Generates a resonable energy interval for EI and RR cross section scans based on user input
    and element binding energies

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    e_kin : None or numpy.ndarray
        <eV>
        If e_kin is None, the range of sampling energies is chosen based on the binding enrgies of
        the element and energies are sampled on a logscale. If e_kin is an array with 2 elements,
        they are interpreted as the minimum and maximum sampling energy.
        If e_kin is an array with more than two values, the energies are taken as the sampling
        energies directly.
    n : int
        The number of energy sampling points, if the sampling locations are not supplied by the
        user.

    Returns
    -------
    numpy.ndarray
        <eV>
        Array of sampling energies
    """
    if e_kin is None:
        e_min = 100.0
        for eb in element.e_bind.flatten():
            if eb > 0 and eb < e_min:
                e_min = eb
        e_min = 1.0 if (e_min < 10.0) else 10.0 # Go to next smaller magnitude
        e_max = 10 * element.e_bind.max()
        e_max = 10**np.ceil(np.log10(e_max))
        e_samp = 10**np.linspace(np.log10(e_min), np.log10(e_max), n)
    elif len(e_kin) == 2:
        e_min = e_kin[0]
        e_max = e_kin[1]
        e_samp = 10**np.linspace(np.log10(e_min), np.log10(e_max), n)
    else:
        e_samp = e_kin
    return e_samp


logger.debug("Defining drxs_energyscan.")
@numba.njit(cache=True)
def drxs_energyscan(element, fwhm, e_kin=None, n=1000):
    """
    Creates an array of DR cross sections for varying electron energies.

    Parameters
    ----------
    element : ebisim.Element
        An ebisim.Element object that holds the required physical information for cross section
        calculations.
    fwhm : float
        <eV>
        Energy spread to apply for the resonance smearing, expressed in terms of
        full width at half maximum.
    e_kin : None or numpy.ndarray, optional
        <eV>
        If e_kin is None, the range of sampling energies is chosen based on the binding enrgies of
        the element and energies are sampled on a logscale. If e_kin is an array with 2 elements,
        they are interpreted as the minimum and maximum sampling energy.
        If e_kin is an array with more than two values, the energies are taken as the sampling
        energies directly, by default None.
    n : int, optional
        The number of energy sampling points, if the sampling locations are not supplied by the
        user, by default 1000.

    Returns
    -------
    e_samp : numpy.ndarray
        <eV>
        Array holding the sampling energies
    xs_scan : numpy.ndarray
        <m^2>
        Array holding the cross sections, where the row index corresponds to the charge state
        and the columns correspond to the different sampling energies

    See Also
    --------
    ebisim.xs.eixs_energyscan
    ebisim.xs.rrxs_energyscan

    """
    if e_kin is None:
        e_min = element.dr_e_res.min() - 3 * fwhm
        e_max = element.dr_e_res.max() + 3 * fwhm
        e_samp = 10**np.linspace(np.log10(e_min), np.log10(e_max), n)
    elif len(e_kin) == 2:
        e_min = e_kin[0]
        e_max = e_kin[1]
        e_samp = 10**np.linspace(np.log10(e_min), np.log10(e_max), n)
    else:
        e_samp = e_kin
    xs_scan = np.zeros((element.z + 1, len(e_samp)))
    for ind, ek in enumerate(e_samp):
        xs_scan[:, ind] = drxs_vec(element, ek, fwhm)
    return e_samp, xs_scan


##### Tables with factors for LOTZ formula
#: Dictionary with the Lotz formula factors for different shells, a in units of <1.0e-14cm**2/eV>
#: These values are relevant for computing the ionisation of neutral atoms only
_LOTZ_NEUTRAL_TABLE = {
    "1s1" : (4.00, 0.60, 0.56),
    "1s2" : (4.00, 0.75, 0.50),
    "2p1" : (3.80, 0.60, 0.40),
    "2p2" : (3.50, 0.70, 0.30),
    "2p3" : (3.20, 0.80, 0.25),
    "2p4" : (3.00, 0.85, 0.22),
    "2p5" : (2.80, 0.90, 0.20),
    "2p6" : (2.60, 0.92, 0.19),
    "3d1" : (3.70, 0.60, 0.40),
    "3d2" : (3.40, 0.70, 0.30),
    "3d3" : (3.10, 0.80, 0.25),
    "3d4" : (2.80, 0.85, 0.20),
    "3d5" : (2.50, 0.90, 0.18),
    "3d6" : (2.20, 0.92, 0.17),
    "3d7" : (2.00, 0.93, 0.16),
    "3d8" : (1.80, 0.94, 0.15),
    "3d9" : (1.60, 0.95, 0.14),
    "3d10": (1.40, 0.96, 0.13),
    "2s1" : (4.00, 0.30, 0.60),
    "2s2" : (4.00, 0.50, 0.60),
    "3p1" : (4.00, 0.35, 0.60),
    "3p2" : (4.00, 0.40, 0.60),
    "3p3" : (4.00, 0.45, 0.60),
    "3p4" : (4.00, 0.50, 0.50),
    "3p5" : (4.00, 0.55, 0.45),
    "3p6" : (4.00, 0.60, 0.40),
    "4d1" : (4.00, 0.30, 0.60),
    "4d2" : (3.80, 0.45, 0.50),
    "4d3" : (3.50, 0.60, 0.40),
    "4d4" : (3.20, 0.70, 0.30),
    "4d5" : (3.00, 0.80, 0.25),
    "4d6" : (2.80, 0.85, 0.20),
    "4d7" : (2.60, 0.90, 0.18),
    "4d8" : (2.40, 0.92, 0.17),
    "4d9" : (2.20, 0.93, 0.16),
    "4d10": (2.00, 0.94, 0.15),
    "3s1" : (4.00, 0.00, 0.00),
    "3s2" : (4.00, 0.30, 0.60),
    "np1" : (4.00, 0.00, 0.00),
    "np2" : (4.00, 0.00, 0.00),
    "np3" : (4.00, 0.20, 0.60),
    "np4" : (4.00, 0.30, 0.60),
    "np5" : (4.00, 0.40, 0.60),
    "np6" : (4.00, 0.50, 0.50),
    "nd1" : (4.00, 0.00, 0.00),
    "nd2" : (4.00, 0.20, 0.60),
    "nd3" : (3.80, 0.30, 0.60),
    "nd4" : (3.60, 0.45, 0.50),
    "nd5" : (3.40, 0.60, 0.40),
    "nd6" : (3.20, 0.70, 0.30),
    "nd7" : (3.00, 0.80, 0.25),
    "nd8" : (2.80, 0.85, 0.20),
    "nd9" : (2.60, 0.90, 0.18),
    "nd10": (2.40, 0.92, 0.17),
    "ns1" : (4.00, 0.00, 0.00),
    "ns2" : (4.00, 0.00, 0.00),
    "nf1" : (3.70, 0.60, 0.40),
    "nf2" : (3.40, 0.70, 0.30),
    "nf3" : (3.10, 0.80, 0.25),
    "nf4" : (2.80, 0.85, 0.20),
    "nf5" : (2.50, 0.90, 0.18),
    "nf6" : (2.20, 0.92, 0.17),
    "nf7" : (2.00, 0.93, 0.16),
    "nf8" : (1.80, 0.94, 0.15),
    "nf9" : (1.60, 0.95, 0.14),
    "nf10": (1.40, 0.96, 0.13),
    "nf11": (1.30, 0.96, 0.12),
    "nf12": (1.20, 0.97, 0.12),
    "nf13": (1.10, 0.97, 0.11),
    "nf14": (1.00, 0.97, 0.11)
}

#: Dictionary with the Lotz formula factors for different shells, a in units of <1.0e-14cm**2/eV>
#: These values are relevant for computing the ionisation elements up to Z=20 and rank over
#: the _LOTZ_NEUTRAL_TABLE
#: The nested dictionary is arranged as [Z][cs][shell]
_LOTZ_ADVANCED_TABLE = {
    1:{
        0:{"1s":(4.00, 0.60, 0.56)}
    },
    2:{
        0:{"1s":(4.00, 0.75, 0.46)},
        1:{"1s":(4.40, 0.38, 0.60)}
    },
    3:{
        0:{"2s":(4.00, 0.70, 2.4), "1s":(4.20, 0.60, 0.60)},
        1:{"1s":(4.00, 0.48, 0.60)},
        2:{"1s":(4.50, 0.20, 0.60)}
    },
    4:{
        0:{"2s":(4.00, 0.70, 0.50), "1s":(4.20, 0.60, 0.60)},
        1:{"2s":(4.40, 0.00, 0.00), "1s":(4.40, 0.40, 0.60)},
        2:{"1s":(4.50, 0.30, 0.60)},
        3:{"1s":(4.50, 0.00, 0.00)}
    },
    5:{
        0:{"2p":(3.80, 0.70, 0.40), "2s":(4.00, 0.70, 0.50)},
        1:{"2s":(4.40, 0.40, 0.60), "1s":(4.40, 0.40, 0.60)},
        2:{"2s":(4.50, 0.00, 0.00), "1s":(4.50, 0.20, 0.60)},
        3:{"1s":(4.50, 0.00, 0.00)}
    },
    6:{
        0:{"2p":(3.50, 0.70, 0.40), "2s":(4.00, 0.70, 0.50)},
        1:{"2p":(4.20, 0.40, 0.60), "2s":(4.40, 0.40, 0.60)},
        2:{"2s":(4.50, 0.20, 0.60), "1s":(4.50, 0.20, 0.60)},
        3:{"2s":(4.50, 0.00, 0.00), "1s":(4.50, 0.00, 0.00)}
    },
    7:{
        0:{"2p":(3.20, 0.83, 0.22), "2s":(4.00, 0.70, 0.50)},
        1:{"2p":(3.90, 0.46, 0.62), "2s":(4.40, 0.40, 0.60)},
        2:{"2p":(4.50, 0.20, 0.60), "2s":(4.50, 0.20, 0.60)},
        3:{"2s":(4.50, 0.00, 0.00), "1s":(4.50, 0.00, 0.00)}
    },
    8:{
        0:{"2p":(2.80, 0.74, 0.24), "2s":(4.00, 0.70, 0.50)},
        1:{"2p":(3.70, 0.60, 0.60), "2s":(4.40, 0.40, 0.60)},
        2:{"2p":(4.50, 0.30, 0.60), "2s":(4.50, 0.20, 0.60)},
        3:{"2p":(4.50, 0.00, 0.00), "2s":(4.50, 0.00, 0.00)}
    },
    9:{
        0:{"2p":(2.70, 0.90, 0.20), "2s":(4.00, 0.70, 0.50)},
        1:{"2p":(3.50, 0.70, 0.50), "2s":(4.40, 0.40, 0.60)},
        2:{"2p":(4.50, 0.40, 0.60), "2s":(4.50, 0.20, 0.60)},
        3:{"2p":(4.50, 0.00, 0.00), "2s":(4.50, 0.00, 0.00)}
    },
    10:{
        0:{"2p":(2.60, 0.92, 0.19), "2s":(4.00, 0.70, 0.50)},
        1:{"2p":(3.20, 0.83, 0.48), "2s":(4.40, 0.40, 0.60)},
        2:{"2p":(4.20, 0.50, 0.60), "2s":(4.50, 0.20, 0.60)},
        3:{"2p":(4.50, 0.20, 0.60), "2s":(4.50, 0.00, 0.00)}
    },
    11:{
        0:{"3s":(4.00, 0.00, 0.00), "2p":(3.00, 0.90, 0.20), "2s":(4.00, 0.70, 0.50)},
        1:{"2p":(3.40, 0.84, 0.32), "2s":(4.40, 0.40, 0.60)},
        2:{"2p":(4.00, 0.60, 0.50), "2s":(4.50, 0.20, 0.60)},
        3:{"2p":(4.50, 0.20, 0.60), "2s":(4.50, 0.00, 0.00)}
    },
    12:{
        0:{"3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20), "2s":(4.00, 0.70, 0.50)},
        1:{"3s":(4.40, 0.00, 0.00), "2p":(3.70, 0.80, 0.40), "2s":(4.40, 0.40, 0.60)},
        2:{"2p":(4.00, 0.60, 0.50), "2s":(4.50, 0.20, 0.60)},
        3:{"2p":(4.50, 0.30, 0.60), "2s":(4.50, 0.00, 0.00)}
    },
    13:{
        0:{"3p":(4.00, 0.30, 0.60), "3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20)},
        1:{"3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40), "2s":(4.40, 0.40, 0.60)},
        2:{"3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50), "2s":(4.50, 0.20, 0.60)},
        3:{"2p":(4.50, 0.30, 0.60), "2s":(4.50, 0.00, 0.00)}
    },
    14:{
        0:{"3p":(4.00, 0.30, 0.60), "3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20)},
        1:{"3p":(4.40, 0.20, 0.60), "3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40)},
        2:{"3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50), "2s":(4.50, 0.20, 0.60)},
        3:{"3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60), "2s":(4.50, 0.00, 0.00)}
    },
    15:{
        0:{"3p":(4.00, 0.40, 0.60), "3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20)},
        1:{"3p":(4.40, 0.20, 0.60), "3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40)},
        2:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50)},
        3:{"3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60), "2s":(4.50, 0.00, 0.00)}
    },
    16:{
        0:{"3p":(4.00, 0.40, 0.60), "3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20)},
        1:{"3p":(4.40, 0.30, 0.60), "3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40)},
        2:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50)},
        3:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60)}
    },
    17:{
        0:{"3p":(4.00, 0.50, 0.50), "3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20)},
        1:{"3p":(4.40, 0.30, 0.60), "3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40)},
        2:{"3p":(4.50, 0.20, 0.60), "3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50)},
        3:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60)}
    },
    18:{
        0:{"3p":(4.00, 0.62, 0.40), "3s":(4.00, 0.40, 0.60), "2p":(3.00, 0.90, 0.20)},
        1:{"3p":(4.20, 0.30, 0.60), "3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40)},
        2:{"3p":(4.50, 0.20, 0.60), "3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50)},
        3:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60)}
    },
    19:{
        0:{"4s":(4.00, 0.00, 0.00), "3p":(4.00, 0.60, 0.40), "3s":(4.00, 0.40, 0.60)},
        1:{"3p":(4.00, 0.30, 0.60), "3s":(4.40, 0.20, 0.60), "2p":(3.70, 0.80, 0.40)},
        2:{"3p":(4.50, 0.20, 0.60), "3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50)},
        3:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60)}
    },
    20:{
        0:{"4s":(4.00, 0.40, 0.60), "3p":(4.00, 0.60, 0.40), "3s":(4.00, 0.40, 0.60)},
        1:{"4s":(4.40, 0.00, 0.00), "3p":(4.40, 0.30, 0.60), "3s":(4.40, 0.20, 0.60)},
        2:{"3p":(4.50, 0.20, 0.60), "3s":(4.50, 0.00, 0.00), "2p":(4.20, 0.60, 0.50)},
        3:{"3p":(4.50, 0.00, 0.00), "3s":(4.50, 0.00, 0.00), "2p":(4.50, 0.30, 0.60)}
    }
}
