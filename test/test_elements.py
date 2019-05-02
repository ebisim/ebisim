"""
Tests for ebisim.elements
"""
import pytest
from ebisim.elements import (
    element_name,
    element_symbol,
    element_z,
    Element,
    ChemicalElement
)

def test_element_z():
    assert element_z("K") == 19
    assert element_z("Potassium") == 19
    assert element_z("Cs") == 55
    assert element_z("Caesium") == 55
    assert element_z("U") == 92
    assert element_z("Uranium") == 92

def test_element_symbol():
    assert element_symbol(19) == "K"
    assert element_symbol("Potassium") == "K"
    assert element_symbol(55) == "Cs"
    assert element_symbol("Caesium") == "Cs"
    assert element_symbol(92) == "U"
    assert element_symbol("Uranium") == "U"

def test_element_name():
    assert element_name(19) == "Potassium"
    assert element_name("K") == "Potassium"
    assert element_name(55) == "Caesium"
    assert element_name("Cs") == "Caesium"
    assert element_name(92) == "Uranium"
    assert element_name("U") == "Uranium"

def test_ChemicalElement():
    k = [19, "K", "Potassium", 39]
    cs = [55, "Cs", "Caesium", 133]
    u = [92, "U", "Uranium", 238]
    for elem in [k, cs, u]:
        for idx in elem[:-1]:
            ce = ChemicalElement(idx)
            assert ce.z == elem[0]
            assert ce.symbol == elem[1]
            assert ce.name == elem[2]
            assert ce.a == elem[3]

def test_Element_basic_info():
    # Test some elements
    k = [19, "K", "Potassium", 39]
    cs = [55, "Cs", "Caesium", 133]
    u = [92, "U", "Uranium", 238]
    for elem_ref in [k, cs, u]:
        for idx in elem_ref[:-1]:
            elem = Element(idx)
            assert elem.z == elem_ref[0]
            assert elem.symbol == elem_ref[1]
            assert elem.name == elem_ref[2]
            assert elem.a == elem_ref[3]
    # Check that meaningless elements throw ValueErrors
    with pytest.raises(ValueError):
        Element(0)
    with pytest.raises(ValueError):
        Element(106)
    with pytest.raises(ValueError):
        Element("h")
    with pytest.raises(ValueError):
        Element("HE")
    with pytest.raises(ValueError):
        Element("XX")
    with pytest.raises(ValueError):
        Element("Ccarbon")

def test_Element_mass_number():
    # test overwriting of the defaults of a
    for z in [12, 15, 55]:
        for a_ref in [1, 12, 54, 23, 77]:
            assert Element(z, a=a_ref).a == a_ref
    # test that meaningless a raise an Error
    with pytest.raises(ValueError):
        Element(1, a=-1)
    with pytest.raises(ValueError):
        Element(1, a=0)
    with pytest.raises(TypeError):
        Element(1, a="x")
