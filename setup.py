import importlib.resources
import setuptools

def from_init(attr):
    text = importlib.resources.read_text("ebisim", "__init__.py")
    for line in text.split("\n"):
        if line.strip().startswith(attr):
            return line.split("=")[-1].strip(" \n\'\"")
    return "attr not found"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ebisim",
    version=from_init("__version__"),
    author=from_init("__author__"),
    #author_email="n/a",
    description="A package for simulating the charge state distribution evolution in an EBIS/EBIT.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HPLegion/ebisim",
    python_requires=">=3.7.0",
    install_requires=["numpy", "scipy", "pandas", "numba", "matplotlib"],
    packages=setuptools.find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
)
