"""
This module contains functions for calculating collission rates and related plasma parameters
"""

import math
import numba
import numpy as np

from.physconst import M_E, M_P, PI, EPS_0, Q_E, C_L, M_E_EV

# @numba.jit
def electron_velocity(e_kin):
    """
    Returns the electron velocity corresponding to a kin. energy in m/s

    Input Parameters
    e_kin - electron energy in eV
    """
    return C_L * np.sqrt(1 - (M_E_EV / (M_E_EV + e_kin))**2)

# @numba.jit
def clog_ei(Ni, Ne, kbTi, kbTe, Ai, qi):
    """
    The coulomb logarithm for ion electron collisions
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    kbTe - electron energy /temperature in eV
    Ai - ion mass in amu
    qi - ion charge
    """
    Ni *= 1e-6 # go from 1/m**3 to 1/cm**3
    Ne *= 1e-6
    Mi = Ai * M_P
    if (qi*qi*10 > kbTe) and (kbTe > kbTi * M_E / Mi):
        return 23 - math.log(Ne**0.5 * qi * kbTe**-1.5)
    elif (kbTe > kbTi * M_E / Mi):#(kbTe > qi*qi*10) and (qi*qi*10 > kbTi * M_E / Mi):
        return 24 - math.log(Ne**0.5 / kbTe)
    elif kbTe < kbTi * M_E / Mi:
        return 16 - math.log(Ni**0.5 * kbTi**-1.5 * qi * qi / Ai)
    else:
        print(kbTe, kbTi)
        raise Exception

# @numba.jit
def clog_ei_vec(Ni, Ne, kbTi, kbTe, Ai):
    """
    Vector version of clog_ei
    Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    kbTe - electron energy /temperature in eV
    Ai - ion mass in amu
    """
    n = Ni.size
    clog = np.zeros(n)
    for q in range(1, n):
        clog[q] = clog_ei(Ni[q], Ne, kbTi[q], kbTe, Ai, q)
    return clog

# @numba.jit
def clog_ii(Ni, Nj, kbTi, kbTj, Ai, Aj, qi, qj):
    """
    The coulomb logarithm for ion ion collisions
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Ai/Aj - ion mass in amu
    qi/qj - ion charge
    """
    A = qi * qj * (Ai + Aj) / (Ai * kbTj + Aj * kbTi)
    B = Ni * qi *qi / kbTi + Nj * qj * qj /kbTj
    return 23 - math.log(A * B**0.5)

# @numba.jit
def clog_ii_mat(Ni, Nj, kbTi, kbTj, Ai, Aj):
    """
    Matrix version of clog_ii
    Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
    The coulomb logarithm for ion ion collisions
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Ai/Aj - ion mass in amu
    qi/qj - ion charge
    """
    ni = Ni.size
    nj = Nj.size
    clog = np.zeros((ni, nj))
    for qj in range(1, nj):
        for qi in range(1, ni):
            clog[qi, qj] = clog_ii(Ni[qi], Nj[qj], kbTi[qi], kbTj[qj], Ai, Aj, qi, qj)
    return clog

# @numba.jit
def coulomb_xs(Ni, Ne, kbTi, Ee, Ai, qi):
    """
    Computes the coulomb cross section for electron ion elastic collisions
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    Ee - electron kinetic energy in eV
    Ai - ion mass in amu
    qi - ion charge
    """
    v_e = electron_velocity(Ee)
    clog = clog_ei(Ni, Ne, kbTi, Ee, Ai, qi)
    return 4 * PI * (qi * Q_E * Q_E / (4 * PI * EPS_0 * M_E))**2 * clog / v_e**4

# @numba.jit
def coulomb_xs_vec(Ni, Ne, kbTi, Ee, Ai):
    """
    Vector version of coulomb_xs
    Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    Ee - electron kinetic energy in eV
    Ai - ion mass in amu
    """
    n = Ni.size
    xs = np.zeros(n)
    for q in range(1, n):
        xs[q] = coulomb_xs(Ni[q], Ne, kbTi[q], Ee, Ai, q)
    return xs

# @numba.jit
def electron_heating_vec(Ni, Ne, kbTi, Ee, Ai):
    """

    """
    ve = electron_velocity(Ee)
    coul_xs = coulomb_xs_vec(Ni, Ne, kbTi, Ee, Ai)
    heat = Ne * ve * coul_xs * Ni * 2 * M_E / (Ai * M_P)
    return heat/Q_E
