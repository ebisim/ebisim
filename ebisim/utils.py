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

    new_data = {}
    for key, val in data.items():
        new_key = int(key) #Cast String type key to int (this is Z of the element)

        new_cfg = []
        for cfg in val["cfg"]:
            cfg = np.array(cfg)
            cfg.setflags(write=False) # This should be read only data
            new_cfg.append(cfg)
        new_cfg = tuple(new_cfg)

        new_ebind = []
        for ebind in val["ebind"]:
            ebind = np.array(ebind)
            ebind.setflags(write=False) # This should be read only data
            new_ebind.append(ebind)
        new_ebind = tuple(new_ebind)

        new_data[new_key] = dict(cfg=new_cfg, ebind=new_ebind)

    return new_data
