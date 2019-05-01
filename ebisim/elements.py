"""
Module holding functions and classes for convenient handling of elements in the periodic table
May be extended for more functionality (element properties) in the future
"""

import collections
import numpy as np

from . import utils
from . import xs

_ELEM_Z, _ELEM_ES, _ELEM_NAME, _ELEM_A = utils.load_element_info()
_ELECTRON_INFO, _SHELLORDER = utils.load_electron_info()
_SHELL_N = np.array(list(map(int, [s[0] for s in _SHELLORDER])))
# _SHELLORDER is a tuple containing the names of all shells in order
# _SHELL_N is an array of the main quantum number of each shell in order

##### Helper functions for translating chemical symbols

def element_z(element):
    """
    Returns the Atomic Number of the Element, for given name or Element Symbol
    """
    if len(element) < 3:
        idx = _ELEM_ES.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_Z[idx]

def element_symbol(element):
    """
    Returns the Symbol of the Element, for given name or Atomic Number
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_ES[idx]

def element_name(element):
    """
    Returns the Name of the Element, for given Symbol or Atomic Number
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_ES.index(element)
    return _ELEM_NAME[idx]

def cast_to_ChemicalElement(element):
    """
    Checks if element is of type ChemicalElement
    If yes the element is returned unchanged
    If no a ChemicalElement object is created based on element and returned
    """
    if isinstance(element, ChemicalElement):
        return element
    return ChemicalElement(element)


class ChemicalElement(collections.namedtuple("ChemicalElement", ["z", "symbol", "name", "a"])):
    """
    Named tuple holding some essential information about a chemical element
    """
    # https://docs.python.org/3.6/library/collections.html#collections.namedtuple
    # contains documentation for named tuples
    __slots__ = () # This is a trick to suppress unnecessary dict() for this kind of class
    def __new__(cls, element_id, a=None):
        """
        Provides a convenient constructor accepting the atomic number, symbol, or name
        If a is provided is interpreted as the mass number, otherwise a resonable value is assigned
        """
        # Info on __new__ for subclasses of namedtuple
        if isinstance(element_id, int):
            z = element_id
        else:
            z = element_z(element_id)
        symbol = element_symbol(z)
        name = element_name(z)
        if a is None:
            idx = _ELEM_Z.index(z)
            a = _ELEM_A[idx]
        return super(ChemicalElement, cls).__new__(cls, z, symbol, name, a)

    def latex_isotope(self):
        """
        returns a latex formatted string describing the isotope
        """
        return "$^{%d}_{%d}$%s"%(self.a, self.z, self.symbol)


class Element(collections.namedtuple(
    "Element", ["z", "symbol", "name", "a", "cfg", "ebind", "z_eff", "n_0_eff"])):
    """
    Named tuple holding some essential information about a chemical element
    """
    # https://docs.python.org/3.6/library/collections.html#collections.namedtuple
    # contains documentation for named tuples
    __slots__ = () # This is a trick to suppress unnecessary dict() for this kind of class
    def __new__(cls, element_id, a=None):
        """
        Provides a convenient constructor accepting the atomic number, symbol, or name
        If a is provided is interpreted as the mass number, otherwise a resonable value is assigned
        """
        # Info on __new__ for subclasses of namedtuple
        if isinstance(element_id, int):
            z = element_id
        else:
            z = element_z(element_id)
        symbol = element_symbol(z)
        name = element_name(z)
        if a is None:
            idx = _ELEM_Z.index(z)
            a = _ELEM_A[idx]
        cfg = _ELECTRON_INFO[z]["cfg"]
        ebind = _ELECTRON_INFO[z]["ebind"]
        z_eff, n_0_eff = xs.precompute_rr_quantities(cfg, _SHELL_N)
        z_eff.setflags(write=False)
        n_0_eff.setflags(write=False)
        return super(Element, cls).__new__(cls, z, symbol, name, a, cfg, ebind, z_eff, n_0_eff)

    def latex_isotope(self):
        """
        returns a latex formatted string describing the isotope
        """
        return f"$^{self.a}_{self.z}${self.symbol}"

Element.z.__doc__ = "Atomic number"
Element.symbol.__doc__ = "Element symbol e.g. H, He, Li"
Element.name.__doc__ = "Element name"
Element.a.__doc__ = "Mass number"
Element.cfg.__doc__ = f"""Numpy array of electron configuration in different charge states
The index of each row corresponds to the charge state
The columns are the subshells sorted as in {_SHELLORDER}"""
Element.ebind.__doc__ = f"""Numpy array of binding energies associated with electron subshells
The index of each row corresponds to the charge state
The columns are the subshells sorted as in {_SHELLORDER}"""
