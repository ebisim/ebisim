"""Resource module"""
from ._element_data import (
    Z as ELEMENT_Z,
    ES as ELEMENT_ES,
    NAME as ELEMENT_NAME,
    A as ELEMENT_A,
    IP as ELEMENT_IP,
)

from ._shell_data import (
    ORDER as SHELL_ORDER,
    N as SHELL_N,
    CFG as SHELL_CFG,
    EBIND as SHELL_EBIND,
)

__all__ = [
    "ELEMENT_Z", "ELEMENT_ES", "ELEMENT_NAME", "ELEMENT_A", "ELEMENT_IP",
    "SHELL_ORDER", "SHELL_N", "SHELL_CFG", "SHELL_EBIND",
]
