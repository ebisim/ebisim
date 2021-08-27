"""
This is the top level module of ebisim.

The ebisim package offers a number of tools to perform and evaluate simulations of the charge
breeding process in an Electron Beam Ion Source / Trap.

For easier access a number of submodule members are available at this level, such that they can be
referred to as ebisim.[member] without resolving the submodule in the call.
The module level documentation lists these members with more granularity.

"""

__version__ = "0.2.0"
__author__ = "Hannes Pahl"

from .xs import (
    eixs_vec, eixs_mat, eixs_energyscan,
    rrxs_vec, rrxs_mat, rrxs_energyscan,
    drxs_vec, drxs_mat, drxs_energyscan,
    )

from .simulation import (
    BasicResult,
    AdvancedResult,
    basic_simulation,
    advanced_simulation,
    Device,
    ModelOptions,
    BackgroundGas,
    AdvancedModel,
    energy_scan,
    EnergyScanResult,
    Rate,
)

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

from .elements import Element, get_element

from .plotting import plot_eixs, plot_rrxs, plot_drxs, plot_combined_xs

from . import beams

Target = Element  # Element and Target used to be different things, now they are the same.

__all__ = [
    # xs
    "eixs_vec", "eixs_mat", "eixs_energyscan",
    "rrxs_vec", "rrxs_mat", "rrxs_energyscan",
    "drxs_vec", "drxs_mat", "drxs_energyscan",
    # simulation
    "BasicResult", "AdvancedResult", "basic_simulation", "advanced_simulation", "Target", "Device",
    "ModelOptions", "BackgroundGas", "AdvancedModel", "energy_scan", "EnergyScanResult", "Rate",
    # physconst
    "Q_E", "M_E", "PI", "EPS_0", "K_B", "C_L", "ALPHA",
    "HBAR", "M_P", "M_E_EV", "RY_EV", "COMPT_E_RED",
    # elements
    "Element", "get_element",
    # plotting
    "plot_eixs", "plot_rrxs", "plot_drxs", "plot_combined_xs",
    # beams
    "beams",
]
