"""
This module contains convenience and management functions not directly related to the
simulation code, e.g. loading resources. These functions are meant for internal use only, they
have no real use outside this scope.
"""
from importlib.resources import open_text
from typing import Dict, ForwardRef, List, TextIO, Any
from importlib import import_module
import numpy as np

from .resources import drdata as _drdata


def load_dr_data() -> Dict[int, Dict[str, np.ndarray]]:
    """
    Loads the avaliable DR transition data from the resource directory

    Returns
    -------
    dict of dicts
        A dictiomary with the proton number as dict-keys.
        Each value is another dictionary with the items
        "dr_e_res" (resonance energy),
        "dr_strength" (transitions strength),
        and "dr_cs" (charge state)
        The values are linear numpy arrays holding corresponding data on the same rows.

    """
    out = {}
    empt = np.array([])
    empt.setflags(write=False)
    for z in range(1, 106):
        try:
            with open_text(_drdata, f"DR_{z}.csv") as f:
                dat = _parse_dr_file(f)
        except FileNotFoundError:
            dat = dict(dr_e_res=empt.copy(), dr_strength=empt.copy(), dr_cs=empt.copy())
        dat["dr_cs"] = dat["dr_cs"].astype(int)  # Need to assure int for indexing purposes
        out[z] = dat
    return out


def _parse_dr_file(fobj: TextIO) -> Dict[str, np.ndarray]:
    """
    Parses the content of a single DR data file into a dict with three numpy arrays holding
    the data about resonance energies, transitions strengths and ion charge state.

    Parameters
    ----------
    fobj : file object
        File to parse

    Returns
    -------
    dict
        A dictionary object holding the following items:
        "dr_e_res" (resonance energy),
        "dr_strength" (transitions strength),
        and "dr_cs" (charge state)
        The values are linear numpy arrays holding corresponding data on the same rows.

    """
    fobj.seek(0)
    fobj.readline()
    e_res = []
    stren = []
    cs = []
    for line in fobj:
        data = line.strip().split(",")
        e_res.append(float(data[0]))
        stren.append(float(data[1]))
        cs.append(int(data[4]))
    e_res_arr = np.array(e_res)
    e_res_arr.setflags(write=False)
    stren_arr = np.array(stren)
    stren_arr.setflags(write=False)
    cs_arr = np.array(cs)
    cs_arr.setflags(write=False)
    return dict(dr_e_res=e_res_arr, dr_strength=stren_arr, dr_cs=cs_arr)


def _nparray_from_jagged_list(list_of_lists: List[List]) -> np.ndarray:
    """
    Takes a list of lists with varying length and turns them into a numpy array,
    treating each list as a left-aligned row and padding the right side with zeros

    Parameters
    ----------
    list_of_lists : list of lists
        Data to be transformed into an array

    Returns
    -------
    numpy.ndarray
        A ndarray of sufficient size to hold the data, left aligned, padded with zeros.
    """
    nrows = len(list_of_lists)
    ncols = max(map(len, list_of_lists))
    out = np.zeros((nrows, ncols))
    for irow, data in enumerate(list_of_lists):
        out[irow, :len(data)] = np.array(data)
    return out


def patch_namedtuple_docstrings(named_tuple: Any, docstrings: Dict[str, str]) -> None:
    """
    Add docstrings to the fields of a namedtuple/NamedTuple

    Parameters
    ----------
    named_tuple :
        The class definition inheriting from namedtupe or NamedTuple
    docstrings :
        Dictionary with field names as keys and docstrings as values
    """
    for _k, _v in docstrings.items():
        setattr(getattr(named_tuple, _k), "__doc__", _v)


def validate_namedtuple_field_types(instance: Any) -> bool:
    """
    Checks if the values of a typing.NamedTuple instance agree with the
    types that were annotated in the class definition.

    Values are permitted to be ints if type annotation is float.
    """
    for f in instance._fields:
        tp = instance.__annotations__[f]
        if isinstance(tp, ForwardRef):
            m = import_module(instance.__module__)
            tp = tp._evaluate(vars(m), {})
        if tp == Any:
            continue
        if "typing.Union" in str(getattr(tp, "__origin__", None)):
            tp = tp.__args__
        if tp == float:
            tp = (float, int)
        val = getattr(instance, f)
        if not isinstance(val, tp):
            return False
    return True
