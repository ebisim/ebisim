"""
This module contains convenience functions not directly related to the Simulation code
e.g. resource management
"""
import os

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
