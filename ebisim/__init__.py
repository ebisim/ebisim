"""
This module contains classes and functions helping to simulate charge breeding in an EBIS
including effects of dielectronic recombination
"""

from .xs import (
    eixs_vec, eixs_mat, rrxs_vec, rrxs_mat, drxs_vec, drxs_mat,
    eixs_energyscan, rrxs_energyscan, drxs_energyscan
    )
from .problems import SimpleEBISProblem, ContinuousNeutralInjectionEBISProblem, EnergyScan
from .physconst import (
    Q_E,
    M_E,
    PI,
    EPS_0,
    K_B,
    C_L,
    ALPHA,
    HBAR,
    M_P,
    M_E_EV,
    RY_EV,
    COMPT_E_RED,
)
from .elements import Element
from .plotting import plot_ei_xs, plot_rr_xs, plot_dr_xs, plot_combined_xs
from . import beams
