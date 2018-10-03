"""
Module holding functions and classes for convenient handling of elements in the periodic table
May be extended for more functionality (element properties) in the future
"""

import collections
from . import utils

# Load relevant data from resource folder when module is loaded
def _load_chemical_elements():
    """
    Reads atomic number Z, symbol, and name information from external file
    """
    _z = [] # Atomic Number
    _a = [] # Mass Number
    _es = [] # Element Symbol
    _name = [] # Element Name
    with utils.open_resource("ChemicalElements.csv") as f:
        f.readline() # skip header line
        for line in f:
            data = line.split(",")
            _z.append(int(data[0].strip()))
            _es.append(data[1].strip())
            _name.append(data[2].strip())
            _a.append(int(data[3].strip()))
    return (_z, _es, _name, _a)

(_ELEM_Z, _ELEM_ES, _ELEM_NAME, _ELEM_A) = _load_chemical_elements()

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

### Old version of Chemical Element, saved as a fail safe or to extend in the future
# class ChemicalElement:
#     """
#     Simple class holding some essential information about a chemical element
#     """
#     def __init__(self, element_id):
#         """
#         Initiates the object translating the input parameter and finding the other quantities

#         Input Parameters
#         element_id - Atomic Number or Element Symbol or name
#         """
#         if isinstance(element_id, int):
#             self._z = element_id
#         else:
#             self._z = element_z(element_id)
#         self._es = element_symbol(self._z)
#         self._name = element_name(self._z)

#     @property
#     def atomic_number(self):
#         """Returns the atomic number"""
#         return self._z

#     @property
#     def z(self):
#         """Returns the atomic number"""
#         return self._z

#     @property
#     def name(self):
#         """Returns the name"""
#         return self._name

#     @property
#     def symbol(self):
#         """Returns the chemical symbol"""
#         return(self)._es
