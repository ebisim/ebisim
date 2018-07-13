"""
Tests for the ebisim.xs module (cross sections)
"""

import ebisim.xs
import numpy as np
# import pytest

def test_XSBase_element_properties():
    xsobj = ebisim.xs.XSBase(3)
    assert xsobj._es == "Li"
    assert xsobj._name == "Lithium"
    assert xsobj._z == 3

def test_XSBase_xs():
    xsobj = ebisim.xs.XSBase(3)
    assert xsobj.xs(1, 3) == 3001 # dummy function but needed for testing so this needs to work
    assert xsobj.xs(2, 4) == 4002
    assert xsobj.xs(19, 5) == 5019

def test_XSBase_xs_mat_IONISE():
    ebisim.xs.XSBase.XSTYPE = "IONISE" # this is a really hacky test
    expected = np.diag([-2000, -2001, -2002, -2003]) + np.diag([2000, 2001, 2002], -1)
    np.testing.assert_array_equal(expected, ebisim.xs.XSBase(3).xs_matrix(2))
    ebisim.xs.XSBase.XSTYPE = None # restore normal configuration

def test_XSBase_xs_mat_RECOMB():
    ebisim.xs.XSBase.XSTYPE = "RECOMB" # this is a really hacky test
    expected = np.diag([-3000, -3001, -3002, -3003]) + np.diag([3001, 3002, 3003], 1)
    np.testing.assert_array_equal(expected, ebisim.xs.XSBase(3).xs_matrix(3))
    ebisim.xs.XSBase.XSTYPE = None # restore normal configuration
