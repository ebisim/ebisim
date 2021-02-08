"""
This module most notably implements the Element class, which serves as the main container for
physical data going into the ebisim computations.

Besides that, there are some small helper functions to translate certain element properties,
which may offer convenience to the user.
"""
import logging
from collections import namedtuple


from . import utils
from . import xs

logger = logging.getLogger(__name__)
logger.debug("Loading element data and shell configurations.")

from .resources import (  # noqa: E402
    ELEMENT_Z as _ELEM_Z,
    ELEMENT_ES as _ELEM_ES,
    ELEMENT_NAME as _ELEM_NAME,
    ELEMENT_A as _ELEM_A,
    ELEMENT_IP as _ELEM_IP,
    SHELL_ORDER as _SHELLORDER,
    SHELL_N as _SHELL_N,
    SHELL_CFG as _SHELL_CFG,
    SHELL_EBIND as _SHELL_EBIND
)
logger.debug("Loading DR data.")
_DR_DATA = utils.load_dr_data()

# ----- Helper functions for translating chemical symbols

logger.debug("Defining element_z.")


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


logger.debug("Defining element_symbol.")


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


logger.debug("Defining element_name.")


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


logger.debug("Defining _ElementSpec.")


_ElementSpec = namedtuple(
    "Element", [
        "z",
        "symbol",
        "name",
        "a",
        "ip",
        "e_cfg",
        "e_bind",
        "rr_z_eff",
        "rr_n_0_eff",
        "dr_cs",
        "dr_e_res",
        "dr_strength",
        "ei_lotz_a",
        "ei_lotz_b",
        "ei_lotz_c"
    ]
)


logger.debug("Defining Element.")


class Element(_ElementSpec):
    """
    Use the static `get()` factory method to create instances of this class.

    This class is derived from collections.namedtuple which facilitates use with numba-compiled
    functions.
    Instances are holding information about a chemical element.
    Some of the fields and methods exist mostly for convenience, while others hold essential
    information for the computations of cross sections and rates.
    Many ebisim functions take an instance of Element as their input and access the information
    stored within the instance for their computations.

    See Also
    --------
    ebisim.elements.Element.get
    """
    # https://docs.python.org/3.6/library/collections.html#collections.namedtuple
    # contains documentation for named tuples
    __slots__ = ()  # This is a trick to suppress unnecessary dict() for this kind of class

    @classmethod
    def as_element(cls, element):
        """
        If `element` is already an instance of `Element` it is returned.
        If `element` is a string or int identyfying an element an appropriate `Element` instance is
        returned.

        Parameters
        ----------
        element : ebisim.elements.Element or str or int
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.

        Returns
        -------
        ebisim.elements.Element
            An instance of Element with the physical data corresponding to the supplied element_id,
            and optionally mass number.
        """
        if isinstance(element, cls):
            return element
        elif "ebisim.simulation._advanced.Target" in str(type(element)):  # hacktime
            data = {k: v for k, v in element._asdict().items() if k in cls._fields}
            return cls(**data)
        else:
            return cls.get(element)

    def latex_isotope(self):
        """
        Returns the isotope as a LaTeX formatted string.

        Returns
        -------
        str
            LaTeX formatted string describing the isotope.
        """
        return fr"$\mathsf{{^{{{self.a}}}_{{{self.z}}}{self.symbol}}}$"

    def __repr__(self):
        return f"Element('{self.symbol}', a={self.a})"

    def __str__(self):
        return f"Element: {self.name} ({self.symbol}, Z = {self.z}, A = {self.a})"

    @classmethod
    def get(cls, element_id, a=None):
        """
        Factory method to create instances of the Element class.

        Parameters
        ----------
        element_id : str or int
            The full name, abbreviated symbol, or proton number of the element of interest.
        a : int or None, optional
            If provided sets the (isotopic) mass number of the Element object otherwise a reasonable
            value is chosen automatically, by default None.

        Returns
        -------
        ebisim.elements.Element
            An instance of Element with the physical data corresponding to the supplied element_id,
            and optionally mass number.

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
            raise ValueError(f"Unable to interpret element_id = {element_id}, "
                             + "ebisim only supports elements up to Z = 105.")

        # Mass number and ionisation potential
        idx = _ELEM_Z.index(z)
        ip = float(_ELEM_IP[idx])
        if a is None:
            a = _ELEM_A[idx]
        if a <= 0:
            raise ValueError("Mass number 'a' cannot be smaller than 1.")

        # Electron configuration and shell binding energies
        e_cfg = _SHELL_CFG[z].copy()
        e_bind = _SHELL_EBIND[z].copy()

        # Precomputations for radiative recombination
        # set write protection flags here since precompute_rr_quantities is compiled and cant do so
        rr_z_eff, rr_n_0_eff = xs.precompute_rr_quantities(e_cfg, _SHELL_N)

        # Data for computations of dielectronic recombination cross sections
        dr_cs = _DR_DATA[z]["dr_cs"].copy()
        dr_e_res = _DR_DATA[z]["dr_e_res"].copy()
        dr_strength = _DR_DATA[z]["dr_strength"].copy()

        # Precompute the factors for the Lotz formula for EI cross section
        ei_lotz_a, ei_lotz_b, ei_lotz_c = xs.lookup_lotz_factors(e_cfg, _SHELLORDER)

        # Make sure that all arrays are readonly - better safe than sorry
        e_cfg.setflags(write=False)
        e_bind.setflags(write=False)
        rr_z_eff.setflags(write=False)
        rr_n_0_eff.setflags(write=False)
        dr_cs.setflags(write=False)
        dr_e_res.setflags(write=False)
        dr_strength.setflags(write=False)
        ei_lotz_a.setflags(write=False)
        ei_lotz_b.setflags(write=False)
        ei_lotz_c.setflags(write=False)

        return cls(
            z,
            symbol,
            name,
            a,
            ip,
            e_cfg,
            e_bind,
            rr_z_eff,
            rr_n_0_eff,
            dr_cs,
            dr_e_res,
            dr_strength,
            ei_lotz_a,
            ei_lotz_b,
            ei_lotz_c
        )


logger.debug("Patching Element docstrings.")
# Monkeypatching docstrings for all the fields of Element
Element.z.__doc__ = "Atomic number"
Element.symbol.__doc__ = "Element symbol e.g. H, He, Li"
Element.name.__doc__ = "Element name"
Element.a.__doc__ = "Mass number"
Element.ip.__doc__ = "Ionisation potential"
Element.e_cfg.__doc__ = f"""Numpy array of electron configuration in different charge states.
The index of each row corresponds to the charge state.
The columns are the subshells sorted as in {_SHELLORDER}."""
Element.e_bind.__doc__ = f"""Numpy array of binding energies associated with electron subshells.
The index of each row corresponds to the charge state.
The columns are the subshells sorted as in {_SHELLORDER}."""
Element.rr_z_eff.__doc__ = "Numpy array of effective nuclear charges for RR cross sections."
Element.rr_n_0_eff.__doc__ = "Numpy array of effective valence shell numbers for RR cross sections."
Element.dr_cs.__doc__ = "Numpy array of charge states for DR cross sections."
Element.dr_e_res.__doc__ = "Numpy array of resonance energies for DR cross sections."
Element.dr_strength.__doc__ = "Numpy array of transition strengths for DR cross sections."
Element.ei_lotz_a.__doc__ = "Numpy array of precomputed Lotz factor 'a' for each entry of 'e_cfg'."
Element.ei_lotz_b.__doc__ = "Numpy array of precomputed Lotz factor 'b' for each entry of 'e_cfg'."
Element.ei_lotz_c.__doc__ = "Numpy array of precomputed Lotz factor 'c' for each entry of 'e_cfg'."

logger.debug("Defining get_element")


def get_element(element_id, a=None):
    """
    [LEGACY]
    Factory function to create instances of the Element class.

    Parameters
    ----------
    element_id : str or int
        The full name, abbreviated symbol, or proton number of the element of interest.
    a : int or None, optional
        If provided sets the (isotopic) mass number of the Element object otherwise a reasonable
        value is chosen automatically, by default None.

    Returns
    -------
    ebisim.elements.Element
        An instance of Element with the physical data corresponding to the supplied element_id, and
        optionally mass number.

    Raises
    ------
    ValueError
        If the Element could not be identified or a meaningless mass number is provided.
    """
    return Element.get(element_id, a)
