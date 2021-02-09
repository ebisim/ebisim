"""
This module most notably implements the Element class, which serves as the main container for
physical data going into the ebisim computations.

Besides that, there are some small helper functions to translate certain element properties,
which may offer convenience to the user.
"""
from __future__ import annotations

import logging
from typing import NamedTuple, Tuple, Union, Optional
import numpy as np

from .utils import load_dr_data
from .xs import precompute_rr_quantities, lookup_lotz_factors
from .physconst import MINIMAL_KBT, MINIMAL_N_1D, Q_E, K_B, PI

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
_DR_DATA = load_dr_data()


# ----- Helper functions for translating chemical symbols
def element_z(element: str) -> int:
    """
    Returns the proton number of the given element.

    Parameters
    ----------
    element :
        The full name or the abbreviated symbol of the element.

    Returns
    -------
        Proton number
    """
    if len(element) < 3:
        idx = _ELEM_ES.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_Z[idx]


def element_symbol(element: Union[str, int]) -> str:
    """
    Returns the abbreviated symbol of the given element.

    Parameters
    ----------
    element :
        The full name or the proton number of the element.

    Returns
    -------
        Element symbol
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_NAME.index(element)
    return _ELEM_ES[idx]


def element_name(element: Union[str, int]) -> str:
    """
    Returns the name of the given element.

    Parameters
    ----------
    element :
        The abbreviated symbol or the proton number of the element.

    Returns
    -------
        Element name
    """
    if isinstance(element, int):
        idx = _ELEM_Z.index(element)
    else:
        idx = _ELEM_ES.index(element)
    return _ELEM_NAME[idx]


def element_identify(element_id: Union[str, int]) -> Tuple[int, str, str]:
    """
    Returns the proton number, name, and element symbol relating to the supplied element_id.

    Parameters
    ----------
    element_id :
        The proton number, name or symbol of a chemical element.

    Returns
    -------
        (proton number, name, symbol)

    Raises
    ------
    ValueError
        If the element_id could not be identified or found in the database.
    """
    try:
        if isinstance(element_id, int):
            z = element_id
        else:
            z = element_z(element_id)
        symbol = element_symbol(z)
        name = element_name(z)
    except ValueError as e:
        raise ValueError(f"Unable to interpret element_id = {element_id}, "
                         + "ebisim only supports elements up to Z = 105.") from e
    return z, name, symbol


class Element(NamedTuple):
    """
    The Element class is one of the main data structures in ebisim.
    Virtually any function relies on information provided in this data structure.
    The leading fields of the underlying tuple contain physcial properties,
    whereas the `n` `kT` and `cx` fields are optional and only required for
    advanced simulations.

    Instead of populating the fields manually, the user should choose one of the
    factory functions that meets their needs best.

    For basic simulations and cross sections calculations only the
    physical / chemical properties of and element are needed.
    In these cases use the generic `get()` method to create instances of this class.

    Advanced simulations require additional information about the initial
    particle densities, temperature and participation in charge exchange.
    The user will likely want to chose between the `get_ions()` and `get_gas()` methods,
    which offer a convenient interface for generating this data based on simple parameters.
    If these functions are not flexible enough, the `get()` method can be used to populate
    the required fields manually.

    This class is derived from collections.namedtuple which facilitates use with numba-compiled
    functions.

    See Also
    --------
    ebisim.elements.Element.get
    ebisim.elements.Element.get_ions
    ebisim.elements.Element.get_gas
    """

    z: int
    symbol: str
    name: str
    a: float
    ip: float
    e_cfg: np.ndarray
    e_bind: np.ndarray
    rr_z_eff: np.ndarray
    rr_n_0_eff: np.ndarray
    dr_cs: np.ndarray
    dr_e_res: np.ndarray
    dr_strength: np.ndarray
    ei_lotz_a: np.ndarray
    ei_lotz_b: np.ndarray
    ei_lotz_c: np.ndarray
    n: Optional[np.ndarray] = None
    kT: Optional[np.ndarray] = None
    cx: bool = True

    @classmethod
    def get(cls,
            element_id: Union[str, int],
            a: Optional[float] = None,
            n: Optional[np.ndarray] = None,
            kT: Optional[np.ndarray] = None,
            cx: bool = True) -> Element:
        """
        Factory method to create instances of the Element class.

        Parameters
        ----------
        element_id :
            The full name, abbreviated symbol, or proton number of the element of interest.
        a :
            If provided sets the mass number of the Element object otherwise a reasonable
            value is chosen automatically.
        n :
            <1/m>
            Only needed for advanced simulations!
            Array holding the initial ion line densities of each charge state.
            If provided, has to be an array of length Z+1, where Z is the nuclear charge.
        kT :
            <eV>
            Only needed for advanced simulations!
            Array holding the initial ion line densities of each charge state.
            If provided, has to be an array of length Z+1, where Z is the nuclear charge.
        cx :
            Only needed for advanced simulations!
            Boolean flag determining whether the neutral particles of this element contribute
            to charge exchange with ions.

        Returns
        -------
            An instance of Element with the user-supplied and generated data.

        Raises
        ------
        ValueError
            If the Element could not be identified or a meaningless mass number is provided.

        ValueError
            If the passed arrays for `n` or `kT` have the wrong shape.
        """
        # Basic element info
        z, name, symbol = element_identify(element_id)

        # Mass number and ionisation potential
        idx = _ELEM_Z.index(z)
        ip = float(_ELEM_IP[idx])
        if a is None:
            a = _ELEM_A[idx]
        if a <= 0:
            raise ValueError("Mass number 'a' cannot be smaller than or equal to 0.")

        # Electron configuration and shell binding energies
        e_cfg = _SHELL_CFG[z].copy()
        e_bind = _SHELL_EBIND[z].copy()

        # Precomputations for radiative recombination
        rr_z_eff, rr_n_0_eff = precompute_rr_quantities(e_cfg, _SHELL_N)

        # Data for computations of dielectronic recombination cross sections
        dr_cs = _DR_DATA[z]["dr_cs"].copy()
        dr_e_res = _DR_DATA[z]["dr_e_res"].copy()
        dr_strength = _DR_DATA[z]["dr_strength"].copy()

        # Precompute the factors for the Lotz formula for EI cross section
        ei_lotz_a, ei_lotz_b, ei_lotz_c = lookup_lotz_factors(e_cfg, _SHELLORDER)

        # Validate n and kT
        if n is not None:
            if len(n) != z+1:
                raise ValueError(f"n has the wrong shape {n.shape} for an element with z = {z}.")
            if np.any(n < MINIMAL_N_1D):
                n = np.maximum(n, MINIMAL_N_1D)
                logger.warning(
                    "One or more entries of 'n' were smaller than the minimal internal value.\n"
                    + f"Entries smaller than {MINIMAL_N_1D} were raised to that value."
                )

        if kT is not None:
            if len(kT) != z+1:
                raise ValueError(f"kT has the wrong shape {kT.shape} for an element with z = {z}.")
            if np.any(kT < MINIMAL_KBT):
                kT = np.maximum(kT, MINIMAL_KBT)
                logger.warning(
                    "One or more entries of 'kT' were smaller than the minimal internal value.\n"
                    + f"Entries smaller than {MINIMAL_KBT} were raised to that value."
                )

        # Make sure that all arrays are readonly - better safe than sorry
        arrays = [
            e_cfg,
            e_bind,
            rr_z_eff,
            rr_n_0_eff,
            dr_cs,
            dr_e_res,
            dr_strength,
            ei_lotz_a,
            ei_lotz_b,
            ei_lotz_c,
            n,
            kT,
        ]
        for arr in arrays:
            if arr is not None:
                arr.setflags(write=False)

        return cls(
            z=z,
            symbol=symbol,
            name=name,
            a=a,
            ip=ip,
            e_cfg=e_cfg,
            e_bind=e_bind,
            rr_z_eff=rr_z_eff,
            rr_n_0_eff=rr_n_0_eff,
            dr_cs=dr_cs,
            dr_e_res=dr_e_res,
            dr_strength=dr_strength,
            ei_lotz_a=ei_lotz_a,
            ei_lotz_b=ei_lotz_b,
            ei_lotz_c=ei_lotz_c,
            n=n,
            kT=kT,
            cx=cx,
        )

    @classmethod
    def get_gas(cls,
                element_id: Union[str, int],
                p: float,
                r_dt: float,
                T: float = 300.0,
                cx: bool = True,
                a: Optional[float] = None) -> Element:
        """
        Factory method for defining a neutral gas injection target.
        A gas target is a target with constant density in charge state 0.

        Parameters
        ----------
        element_id :
            The full name, abbreviated symbol, or proton number of the element of interest.
        p :
            <mbar>
            Gas pressure.
        r_dt :
            <m>
            Drift tube radius, required to compute linear density from volumetric density.
        T :
            <K>
            Gas temperature, by default 300 K (approx. room temperature)
        cx :
            Boolean flag determining whether the neutral particles of this element contribute
            to charge exchange with ions.
        a :
            If provided sets the mass number of the Element object otherwise a reasonable
            value is chosen automatically.

        Returns
        -------
            Element instance with automatically populated `n` and `kT` fields.

        Raises
        ------
        ValueError
            If the density resulting from the pressure and temperature is smaller than the
            internal minimal value.
        """
        z, _name, _symbol = element_identify(element_id)
        _n = np.full(z + 1, MINIMAL_N_1D, dtype=np.float64)
        _kT = np.full(z + 1, MINIMAL_KBT, dtype=np.float64)

        _n[0] = (p * 100) / (K_B * T) * PI * r_dt**2  # Convert from mbar to Pa and compute density
        if _n[0] < MINIMAL_N_1D:
            raise ValueError("The resulting density is smaller than the internal minimal value.")
        _kT[0] = K_B * T / Q_E

        return cls.get(element_id, a=a, n=_n, kT=_kT, cx=cx)

    @classmethod
    def get_ions(cls,
                 element_id: Union[str, int],
                 nl: float,
                 kT: float = 10.0,
                 q: int = 1,
                 cx: bool = True,
                 a: Optional[float] = None) -> Element:
        """
        Factory method for defining a pulsed ion injection target.
        An ion target has a given density in the charge state of choice q.

        Parameters
        ----------
        element_id :
            The full name, abbreviated symbol, or proton number of the element of interest.
        nl :
            <1/m>
            Linear density of the initial charge state (ions per unit length).
        kT :
            <eV>
            Temperature / kinetic energy of the injected ions.
        q :
            Initial charge state.
        cx :
            Boolean flag determining whether the neutral particles of this element contribute
            to charge exchange with ions.
        a :
            If provided sets the mass number of the Element object otherwise a reasonable
            value is chosen automatically.

        Returns
        -------
            Element instance with automatically populated `n` and `kT` fields.

        Raises
        ------
        ValueError
            If the requested density is smaller than the internal minimal value.
        """
        if nl < MINIMAL_N_1D:
            raise ValueError("The requested density is smaller than the internal minimal value.")

        z, _name, _symbol = element_identify(element_id)
        _n = np.full(z + 1, MINIMAL_N_1D, dtype=np.float64)
        _kT = np.full(z + 1, MINIMAL_KBT, dtype=np.float64)

        _n[q] = nl
        _kT[q] = kT

        return cls.get(element_id, a=a, n=_n, kT=_kT, cx=cx)

    @classmethod
    def as_element(cls, element: Union[Element, str, int]) -> Element:
        """
        If `element` is already an instance of `Element` it is returned.
        If `element` is a string or int identyfying an element an appropriate `Element` instance is
        returned.

        Parameters
        ----------
        element :
            An instance of the Element class, or an identifier for the element, i.e. either its
            name, symbol or proton number.

        Returns
        -------
            An instance of Element reflecting the input value.
        """
        if isinstance(element, cls):
            return element
        elif isinstance(element, (str, int)):
            return cls.get(element)
        else:
            raise TypeError("Could not convert {element} to ebisim.Element")

    def latex_isotope(self) -> str:
        """
        Returns the isotope as a LaTeX formatted string.

        Returns
        -------
        str
            LaTeX formatted string describing the isotope.
        """
        return fr"$\mathsf{{^{{{self.a}}}_{{{self.z}}}{self.symbol}}}$"

    def __str__(self) -> str:
        return f"Element: {self.name} ({self.symbol}, Z = {self.z}, A = {self.a})"


# Monkeypatching docstrings for all the fields of Element
Element.z.__doc__ = "Atomic number"
Element.symbol.__doc__ = "Element symbol e.g. H, He, Li"
Element.name.__doc__ = "Element name"
Element.a.__doc__ = "Mass number / approx. mass in proton masses"
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
Element.n.__doc__ = """<1/m> Array holding the initial linear density of each charge state."""
Element.kT.__doc__ = """<eV> Array holding the initial temperature of each charge state."""
Element.cx.__doc__ = """Boolean flag determining whether neutral particles of this target are
considered as charge exchange partners."""


def get_element(element_id: Union[str, int], a: Optional[float] = None) -> Element:
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
