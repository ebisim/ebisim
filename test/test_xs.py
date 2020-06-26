"""
Tests for the ebisim.xs module (cross sections)
"""

import ebisim.xs
import ebisim.elements
import numpy as np
# import pytest

#TODO: Implement Tests

K = ebisim.elements.get_element("K")
Cs = ebisim.elements.get_element("Cs")
Db = ebisim.elements.get_element("Db")
ELEMENTS = (K, Cs, Db)


def test_xs_numba_python_equality():
    """Check binary equality between numba and CPython for eixs rrxs drxs cxxs"""
    for element in ELEMENTS:
        for fun in ( ebisim.xs.eixs_vec, ebisim.xs.rrxs_vec):
            xsn = fun(element, 5000.0)
            xsp = fun.py_func(element, 5000.0)
            np.testing.assert_equal(xsn, xsp)

        xsn = ebisim.drxs_vec(element, 5000.0, 15.0)
        xsp = ebisim.drxs_vec.py_func(element, 5000.0, 15.0)
        np.testing.assert_equal(xsn, xsp)

    for ip in np.linspace(3, 20, 50):
        xsn = ebisim.xs.cxxs(np.arange(100), ip)
        xsp = ebisim.xs.cxxs.py_func(np.arange(100), ip)
        np.testing.assert_equal(xsn, xsp)
