"""
This subpackage contains functions and classes provide an interface to run simulations and inspect
their results.
"""

from ._result import Result

from ._basic import basic_simulation

from ._advanced import (
    advanced_simulation,
    Target,
    get_ion_target,
    get_gas_target,
    Device
)

from ._energyscan import(
    energy_scan,
    EnergyScanResult
)

__all__ = [
    "Result",
    "basic_simulation",
    "advanced_simulation",
    "Target",
    "get_ion_target",
    "get_gas_target",
    "Device",
    "energy_scan",
    "EnergyScanResult"
]