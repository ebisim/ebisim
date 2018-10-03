"""
Tests for ebisim.elements
"""
import ebisim.elements
# import pytest

def test_element_z():
    assert ebisim.elements.element_z("K") == 19
    assert ebisim.elements.element_z("Potassium") == 19
    assert ebisim.elements.element_z("Cs") == 55
    assert ebisim.elements.element_z("Caesium") == 55
    assert ebisim.elements.element_z("U") == 92
    assert ebisim.elements.element_z("Uranium") == 92

def test_element_symbol():
    assert ebisim.elements.element_symbol(19) == "K"
    assert ebisim.elements.element_symbol("Potassium") == "K"
    assert ebisim.elements.element_symbol(55) == "Cs"
    assert ebisim.elements.element_symbol("Caesium") == "Cs"
    assert ebisim.elements.element_symbol(92) == "U"
    assert ebisim.elements.element_symbol("Uranium") == "U"

def test_element_name():
    assert ebisim.elements.element_name(19) == "Potassium"
    assert ebisim.elements.element_name("K") == "Potassium"
    assert ebisim.elements.element_name(55) == "Caesium"
    assert ebisim.elements.element_name("Cs") == "Caesium"
    assert ebisim.elements.element_name(92) == "Uranium"
    assert ebisim.elements.element_name("U") == "Uranium"

def test_ChemicalElement():
    k = [19, "K", "Potassium"]
    cs = [55, "Cs", "Caesium"]
    u = [92, "U", "Uranium"]
    a = {"K":39, "Cs":133, "U":238}
    for elem in [k, cs, u]:
        for idx in elem:
            ce = ebisim.elements.ChemicalElement(idx)
            assert ce.z == elem[0]
            assert ce.symbol == elem[1]
            assert ce.name == elem[2]
            assert ce.a == a[elem[1]]
