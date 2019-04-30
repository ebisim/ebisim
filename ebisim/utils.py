"""
This module contains convenience functions not directly related to the Simulation code
e.g. resource management
"""
import os
import json
import numpy as np

##### Logic for robust file imports
_MODULEDIR = os.path.dirname(os.path.abspath(__file__))
_RESOURCEDIR = os.path.join(_MODULEDIR, "resources/")

def _get_res_path(fn):
    """
    Generates the path to a filename in the resource folder

    fn - filename
    """
    return os.path.join(_RESOURCEDIR, fn)


def open_resource(fn):
    """
    Method for opening files in the resource folder in a robust way without worrying
    about the absolute path

    fn - filename
    """
    return open(_get_res_path(fn))


def load_element_info():
    """
    Loads the chemical element info from json resource

    Returns tuples of Z, Symbol, Name, A
    """
    with open_resource("ElementInfo.json") as f:
        data = json.load(f)
    return tuple(map(tuple, [data["z"], data["es"], data["name"], data["a"]]))


def load_electron_info():
    """
    Loads the electron configurations and subshell binding energies for all elements into a
    convenient data structure

    dict with keys Z (proton number)
      dicts with keys
        cfg: tuple of write protected numpy arrays with shell occupation numbers
        ebind: tuple of write protected numpy arrays with shell binding energies
        each tuple has one entry per charge state where the tuple index is equal to the charge state
    """
    with open_resource("BindingEnergies.json") as f:
        data = json.load(f)

    shellorder = data[0]
    data = data[1]

    new_data = {}
    for key, data in data.items():
        new_key = int(key) #Cast String type key to int (this is Z of the element)
        new_cfg = _nparray_from_jagged_list(data["cfg"])
        new_cfg.setflags(write=False)
        new_ebind = _nparray_from_jagged_list(data["ebind"])
        new_ebind.setflags(write=False)

        new_data[new_key] = dict(cfg=new_cfg, ebind=new_ebind)

    return new_data, shellorder

def _nparray_from_jagged_list(list_of_lists):
    """
    Takes a list of lists with varying length and turns them into a numpy array,
    treatin each list as a left-aligned row and padding the right side with zeros
    """
    nrows = len(list_of_lists)
    ncols = max(map(len, list_of_lists))
    out = np.zeros((nrows, ncols))
    for irow, data in enumerate(list_of_lists):
        out[irow, :len(data)] = np.array(data)
    return out
