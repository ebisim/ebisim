"""
This module contains convenience functions not directly related to the Simulation code
e.g. resource management
"""
import os
import json

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
