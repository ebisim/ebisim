"""
This module contains the basic simulation method.
"""
from __future__ import annotations
from typing import Optional, NamedTuple


class BasicDevice(NamedTuple):
    """
    A mock device class for holding simulations parameters for basic_simulations.
    """

    e_kin: float
    j: float
    fwhm: Optional[float]
