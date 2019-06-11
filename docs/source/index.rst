.. ebisim documentation master file, created by
   sphinx-quickstart on Fri May 31 09:32:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ebisim's documentation!
==================================

The ebisim package is being devleoped to provide a collection of tools for simulating the evolution
of the charge state distribution inside an Electron Beam Ion Source / Trap (EBIS/T)
using Python.

.. .. plot ::

..     import ebisim as eb
..     from matplotlib.pyplot import show

..     K = eb.get_element("Potassium")
..     eb.plot_ei_xs(K)
..     show()

This documentation contains a few :doc:`examples <examples/examples>` demonstrating the
general features of ebisim. For a detailed description of the included modules please refer
to the :doc:`API reference <reference/ebisim>`.


.. toctree::
   :maxdepth: 3
   :caption: Reference:

   examples/examples
   reference/ebisim


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
