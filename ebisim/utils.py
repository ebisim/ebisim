"""
This module contains convenience functions not directly related to the Simulation code
e.g. resource management
"""
import os

##### Logic for robust file imports
_MODULEDIR = os.path.dirname(os.path.abspath(__file__))
_RESOURCEDIR = os.path.join(_MODULEDIR, "resources/")
_GETRESDIR = lambda f: os.path.join(_RESOURCEDIR, f)

def open_resource(fn):
    """
    Method for opening files in the resource folder in a robust way without worrying
    about the absolute path
    """
    return open(_GETRESDIR(fn))