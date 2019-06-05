"""
Module holding functions and classes for convenient handling of elements in the periodic table
May be extended for more functionality (element properties) in the future
"""

from collections import namedtuple
import numpy as np

from . import utils
from . import xs

_ELEM_Z, _ELEM_ES, _ELEM_NAME, _ELEM_A = utils.load_element_info()
_ELECTRON_INFO, _SHELLORDER = utils.load_electron_info()
_SHELL_N = np.array(list(map(int, [s[0] for s in _SHELLORDER])))
# _SHELLORDER is a tuple containing the names of all shells in order
# _SHELL_N is an array of the main quantum number of each shell in order
_DR_DATA = utils.load_dr_data()


##### Helper functions for translating chemical symbols

def element_z(element):
    """
    Returns the proton number of the given element.

    Parameters
    ----------
    element : str
        The full name or the abbreviated symbol of the element.

    Returns
    -------
    int
        Proton number
    """
    if len(element) < 3:
        idx = _ELEM_ES.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_Z[idx]


def element_symbol(element):
    """
    Returns the abbreviated symbol of the given element.

    Parameters
    ----------
    element : str or int
        The full name or the proton number of the element.

    Returns
    -------
    str
        Element symbol
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_ES[idx]


def element_name(element):
    """
    Returns the name of the given element.

    Parameters
    ----------
    element : str or int
        The abbreviated symbol or the proton number of the element.

    Returns
    -------
    str
        Element name
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_ES.index(element)
    return _ELEM_NAME[idx]


_ElementSpec = namedtuple(
    "Element", [
        "z",
        "symbol",
        "name",
        "a",
        "cfg",
        "e_bind",
        "z_eff",
        "n_0_eff",
        "dr_cs",
        "dr_e_res",
        "dr_strength"
    ]
)

class Element(_ElementSpec):
    """
    This class is derived from collections.namedtuple which facilitates use with numba-compiled
    functions.
    Instances are holding information about a chemical element.
    Some of the fields and methods exist mostly for convenience, while others hold essential
    information for the computations of cross sections and rates.
    Many ebisim functions take an instance of Element as their input and access the information
    stored within the instance for their computations.

    Parameters
    ----------
    element_id : str or int
        The full name, abbreviated symbol, or proton number of the element of interest.
    a : int or None, optional
        If provided sets the (isotopic) mass number of the Element object otherwise a reasonable
        value is chosen automatically, by default None.


    Raises
    ------
    ValueError
        If the Element could not be identified or a meaningless mass number is provided.
    """
    # https://docs.python.org/3.6/library/collections.html#collections.namedtuple
    # contains documentation for named tuples
    __slots__ = () # This is a trick to suppress unnecessary dict() for this kind of class
    def __new__(cls, element_id, a=None):
        """
        Provides a convenient constructor accepting the atomic number, symbol, or name
        If a is provided is interpreted as the mass number, otherwise a resonable value is assigned
        The __new__ construct is necessary due to the immutable nature of the underlying namedtuple.


        Parameters
        ----------
        element_id : str or int
            The full name, abbreviated symbol, or proton number of the element of interest.
        a : int or None, optional
            If provided sets the (isotopic) mass number of the Element object otherwise a reasonable
            value is chosen automatically, by default None.


        Raises
        ------
        ValueError
            If the Element could not be identified or a meaningless mass number is provided.
        """
        # Basic element info
        try:
            if isinstance(element_id, int):
                z = element_id
            else:
                z = element_z(element_id)
            symbol = element_symbol(z)
            name = element_name(z)
        except ValueError:
            raise ValueError(f"Unable to interpret element_id = {element_id}, " \
                "ebisim only supports elements up to Z = 105.")

        # Mass number
        if a is None:
            idx = _ELEM_Z.index(z)
            a = _ELEM_A[idx]
        if a <= 0:
            raise ValueError("Mass number 'a' cannot be smaller than 1.")

        # Electron configuration and shell binding energies
        cfg = _ELECTRON_INFO[z]["cfg"]
        e_bind = _ELECTRON_INFO[z]["ebind"]

        # Precomputations for radiative recombination
        z_eff, n_0_eff = xs.precompute_rr_quantities(cfg, _SHELL_N)

        # Data for computations of dielectronic recombination cross sections
        dr_cs = _DR_DATA[z]["dr_cs"]
        dr_e_res = _DR_DATA[z]["dr_e_res"]
        dr_strength = _DR_DATA[z]["dr_strength"]

        return super(Element, cls).__new__(
            cls,
            z,
            symbol,
            name,
            a,
            cfg,
            e_bind,
            z_eff,
            n_0_eff,
            dr_cs,
            dr_e_res,
            dr_strength
        )

    def latex_isotope(self):
        """
        Returns the isotope as a LaTeX formatted string.

        Returns
        -------
        str
            LaTeX formatted string describing the isotope.
        """
        return f"$^{{{self.a}}}_{{{self.z}}}${self.symbol}"

    def __repr__(self):
        return f"Element('{self.symbol}', a={self.a})"

    def __str__(self):
        return f"Element: {self.name} ({self.symbol}, Z = {self.z}, A = {self.a})"

# Monkeypatching docstrings for all the fields of Element
Element.z.__doc__ = "Atomic number"
Element.symbol.__doc__ = "Element symbol e.g. H, He, Li"
Element.name.__doc__ = "Element name"
Element.a.__doc__ = "Mass number"
Element.cfg.__doc__ = f"""Numpy array of electron configuration in different charge states.
The index of each row corresponds to the charge state.
The columns are the subshells sorted as in {_SHELLORDER}."""
Element.e_bind.__doc__ = f"""Numpy array of binding energies associated with electron subshells.
The index of each row corresponds to the charge state.
The columns are the subshells sorted as in {_SHELLORDER}."""
Element.z_eff.__doc__ = "Numpy array of effective nuclear charges for RR cross sections."
Element.n_0_eff.__doc__ = "Numpy array of effective valence shell numbers for RR cross sections."
Element.dr_cs.__doc__ = "Numpy array of charge states for DR cross sections."
Element.dr_e_res.__doc__ = "Numpy array of resonance energies for DR cross sections."
Element.dr_strength.__doc__ = "Numpy array of transition strengths for DR cross sections."
