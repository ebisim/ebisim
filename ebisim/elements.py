"""
Module holding functions and classes for convenient handling of elements in the periodic table
May be extended for more functionality (element properties) in the future
"""

from . import utils

# Load relevant data from resource folder when module is loaded
def _load_chemical_elements():
    """
    Reads atomic number Z, symbol, and name information from external file
    """
    _z = [] # Atomic Number
    _es = [] # Element Symbol
    _name = [] # Element Name
    with utils.open_resource("ChemicalElements.csv") as f:
        f.readline() # skip header line
        for line in f:
            data = line.split(",")
            _z.append(int(data[0]))
            _es.append(data[1])
            _name.append(data[2].strip())
    return (_z, _es, _name)

(_ELEM_Z, _ELEM_ES, _ELEM_NAME) = _load_chemical_elements()

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


class ChemicalElement:
    """
    Simple class holding some essential information about a chemical element
    """
    def __init__(self, element_id):
        """
        Initiates the object translating the input parameter and finding the other quantities

        Input Parameters
        element_id - Atomic Number or Element Symbol or name
        """
        if isinstance(element_id, int):
            self._z = element_id
        else:
            self._z = element_z(element_id)
        self._es = element_symbol(self._z)
        self._name = element_name(self._z)

    @property
    def atomic_number(self):
        """Returns the atomic number"""
        return self._z
    
    @property
    def z(self):
        """Returns the atomic number"""
        return self._z

    @property
    def name(self):
        """Returns the name"""
        return self._name

    @property
    def symbol(self):
        """Returns the chemical symbol"""
        return(self)._es