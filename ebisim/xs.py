"""
Module containing the classes providing access to the relevant cross section for ionisation and
recombination
"""

# from functools import lru_cache
import math
import numpy as np
import numba

from .physconst import RY_EV, ALPHA, PI, COMPT_E_RED

XS_CACHE_MAXSIZE = 10000 # The maxsize for the caching of xs matrices

### The following are low level functions going into the cross section computation
### They are jit compiled by numba to increase their performance
### Unfortunately the readability of the code suffers a bit this way, but ... ce la vie
@numba.njit
def _normpdf(x, mu, sigma):
    """
    The pdf of the normal distribution
    """
    return np.exp(-(x - mu)**2 / (2 * sigma**2)) / (2 * PI * sigma**2)**0.5


@numba.njit
def eixs_vec(element, e_kin):
    css = element.e_bind.shape[0]
    shells = element.e_bind.shape[1]
    xs_vec = np.zeros(css + 1)
    for cs in range(css):
        xs = 0
        for shell in range(shells):
            e = element.e_bind[cs, shell]
            n = element.cfg[cs, shell]
            if e_kin > e and n > 0:
                xs += n * math.log(e_kin / e) / (e_kin * e)
        xs_vec[cs] = xs
    xs_vec *= 4.5e-18
    return xs_vec

@numba.njit
def eixs_mat(element, e_kin):
    xs = eixs_vec(element, e_kin)
    return np.diag(xs[:-1], -1) - np.diag(xs)


@numba.njit
def rrxs_vec(element, e_kin):
    chi = 2 * element.z_eff**2 * RY_EV / e_kin
    xs = 8 * PI * ALPHA / (3 * np.sqrt(3)) * COMPT_E_RED**2 * \
            chi * np.log(1 + chi/(2 * element.n_0_eff**2))
    xs[0] = 0
    return xs


@numba.njit
def rrxs_mat(element, e_kin):
    xs = rrxs_vec(element, e_kin)
    return np.diag(xs[1:], 1) - np.diag(xs)


@numba.njit
def drxs_vec(element, e_kin, fwhm):
    xs_vec = np.zeros(element.z + 1)
    if element.dr_cs.size > 0:
        sig = fwhm/2.35482 # 2.35482approx.(2*np.sqrt(2*np.log(2)))
        tmp = element.dr_strength * _normpdf(e_kin, element.dr_e_res, sig)*1e-24
        for k in range(element.dr_cs.size):
            # cs = int(element.dr_cs[k])
            xs_vec[element.dr_cs[k]] = xs_vec[element.dr_cs[k]] + tmp[k]
    return xs_vec


@numba.njit
def drxs_mat(element, e_kin, fwhm):
    xs = drxs_vec(element, e_kin, fwhm)
    return np.diag(xs[1:], 1) - np.diag(xs)


@numba.jit
def precompute_rr_quantities(cfg, shell_n):
    """
    Precomputes the effective valence shell and nuclear charge for all charge states,
    needed for radiative recombinations cross sections
    """
    z = cfg.shape[0]
    shell_n = shell_n[:cfg.shape[1]] # Crop shell_n to the shells described in cfg

    n_0 = np.zeros(z + 1)
    occup = np.zeros(z + 1)

    # Determine, for each charge state, the valence shell (n_0),
    # and the number of electrons in it (occup)
    # Fully ionised
    n_0[z] = 1
    occup[z] = 0
    # All other charge states
    for cs in range(z):
        conf = cfg[cs, :]
        n_0[cs] = np.max(shell_n[np.nonzero(conf)])
        occup[cs] = np.sum(np.extract((shell_n == n_0[cs]), conf))

    w_n0 = (2 * n_0**2 - occup) / (2 * n_0**2)
    n_0_eff = n_0 + (1 - w_n0) - 0.3
    z_eff = (z + np.arange(z + 1)) / 2

    n_0_eff.setflags(write=False)
    z_eff.setflags(write=False)
    return z_eff, n_0_eff
