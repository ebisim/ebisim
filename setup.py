"""
setup for ebisim
"""

import os
import setuptools

NAME = "ebisim"
EMAIL = None
DESCRIPTION = "A package for simulating the charge state distribution evolution in an EBIS/EBIT."
URL = "https://github.com/ebisim/ebisim"
PYTHON_REQUIRES = ">=3.7.0"
INSTALL_REQUIRES = ["numpy>=1.17", "scipy>=1.3", "numba>=0.50", "matplotlib>=3"]
CLASSIFIERS = [
    'Programming Language :: Python :: 3',
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering :: Physics"
    ]
PACKAGE_DATA = {
    "ebisim": ["*"],
    "ebisim.resources": ["*"],
    "ebisim.resources.drdata": ["*"],
}
LICENSE = "MIT"

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()


def from_init(attr):
    """
    read a value from ebisim/__init__.py
    """
    # text = importlib.resources.read_text("ebisim", "__init__.py")
    loc = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(loc, "ebisim", "__init__.py")) as f:
        text = f.read()
    for line in text.split("\n"):
        if line.strip().startswith(attr):
            return line.split("=")[-1].strip(" \n\'\"")
    return "attr not found"


VERSION = from_init("__version__")
AUTHOR = from_init("__author__")

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    package_data=PACKAGE_DATA,
    classifiers=CLASSIFIERS,
    license=LICENSE,
)
