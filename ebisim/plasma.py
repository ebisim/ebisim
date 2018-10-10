"""
This module contains functions for calculating collission rates and related plasma parameters
"""

import math
import numba
import numpy as np

from.physconst import M_E, M_P

@numba.jit
def clog_ei(Ni, Ne, kbTi, kbTe, Mi, qi):
    """
    The coulomb logarithm for ion electron collisions
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    kbTe - electron energy /temperature in eV
    Mi - ion mass in amu
    qi - ion charge
    """
    Ni *= 1e-6 # go from 1/m**3 to 1/cm**3
    Ne *= 1e-6
    Mi *= M_P
    if (qi*qi*10 > kbTe) and (kbTe > kbTi * M_E / Mi):
        return 23 - math.log(Ne**0.5 * qi * kbTe**-1.5)
    elif (kbTe > qi*qi*10) and (qi*qi*10 > kbTi * M_E / Mi):
        return 24 - math.log(Ne**0.5 / kbTe)
    elif kbTe < kbTi * M_E / Mi:
        return 16 - math.log(Ni**0.5 * kbTi**-1.5 * qi * qi * M_P / Mi)

@numba.jit
def clog_ei_vec(Ni, Ne, kbTi, kbTe, Mi):
    """
    Vector version of clog_ei
    Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
    Ni - ion density in 1/m^3
    Ne - electron density in 1/m^3
    kbTi - electron energy /temperature in eV
    kbTe - electron energy /temperature in eV
    Mi - ion mass in amu
    """
    n = Ni.size
    clog = np.zeros(n)
    for q in range(n):
        clog[q] = clog_ei(Ni[q], Ne, kbTi, kbTe, Mi, q)
    return clog

@numba.jit
def clog_ii(Ni, Nj, kbTi, kbTj, Mi, Mj, qi, qj):
    """
    The coulomb logarithm for ion ion collisions
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Mi/Mj - ion mass in amu
    qi/qj - ion charge
    """
    A = qi * qj * (Mi + Mj) / (Mi * kbTj + Mj * kbTi)
    B = Ni * qi *qi / kbTi + Nj * qj * qj /kbTj
    return 23 - math.log(A * B**0.5)

@numba.jit
def clog_ii_mat(Ni, Nj, kbTi, kbTj, Mi, Mj):
    """
    Matrix version of clog_ii
    Assumes Ni and kbTi are vectors of length Z +1 where the index corresponds to the charge state
    The coulomb logarithm for ion ion collisions
    Ni/Nj - ion density in 1/m^3
    kbTi/kbTj - electron energy /temperature in eV
    Mi/Mj - ion mass in amu
    qi/qj - ion charge
    """
    ni = Ni.size
    nj = Nj.size
    clog = np.zeros((ni, nj))
    for qj in range(nj):
        for qi in range(ni):
            clog[qi, qj] = clog_ii(Ni[qi], Nj[qj], kbTi[qi], kbTj[qj], Mi, Mj, qi, qj)
    return clog
