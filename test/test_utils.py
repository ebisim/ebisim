"""
Tests for ebisim.utils
"""
import ebisim.utils
# import pytest

def test_open_chemical_elements_csv():
    with ebisim.utils.open_resource("ChemicalElements.csv"):
        pass
