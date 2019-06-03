"""
This module contains convenience and management functions not directly related to the
simulation code, e.g. loading resources.
"""
import json
from importlib.resources import open_text
import numpy as np

from . import resources as _resources
from .resources import drdata as _drdata


def load_element_info():
    """
    Loads the basic information about chemical elements from a json resource

    Returns
    -------
    z : tuple of ints
        Atomic numbers
    es : tuple of strings
        Element symbol, eg 'H'
    name : tuple of str
        Element name
    a : tuple of ints
        Naturally abundant / typical mass number

    """
    with open_text(_resources, "ElementInfo.json") as f:
        data = json.load(f)
    return tuple(map(tuple, [data["z"], data["es"], data["name"], data["a"]]))


def load_electron_info():
    """
    Loads the electron configurations and subshell binding energies for all elements into a
    convenient data structure

    Returns
    -------
    electron_info : dict of dicts
        A dictiomary with the proton number as dict-keys.
        Each value is another dictionary with the items "cfg" (electron configuration) and
        "ebind" (binding energy of the subshells).
        The values are 2D numpy arrays with the charge states as the rows and the subshells in the
        columns.
    shellorder : tuple of strings
        Subshell names in the order they appear in electron info.

    """
    with open_text(_resources, "BindingEnergies.json") as f:
        data = json.load(f)

    shellorder = tuple(data[0])
    data = data[1]

    electron_info = {}
    for key, data in data.items():
        new_key = int(key) #Cast String type key to int (this is Z of the element)
        new_cfg = _nparray_from_jagged_list(data["cfg"])
        new_cfg.setflags(write=False)
        new_ebind = _nparray_from_jagged_list(data["ebind"])
        new_ebind.setflags(write=False)

        electron_info[new_key] = dict(cfg=new_cfg, ebind=new_ebind)

    return electron_info, shellorder


def load_dr_data():
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
        dat["dr_cs"] = dat["dr_cs"].astype(int) # Need to assure int for indexing purposes
        out[z] = dat
    return out


def _parse_dr_file(fobj):
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
    e_res = np.array(e_res)
    e_res.setflags(write=False)
    stren = np.array(stren)
    stren.setflags(write=False)
    cs = np.array(cs)
    cs.setflags(write=False)
    return dict(dr_e_res=e_res, dr_strength=stren, dr_cs=cs)


def _nparray_from_jagged_list(list_of_lists):
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
