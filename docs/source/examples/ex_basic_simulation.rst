Running a basic simulation
==========================

Ebisim offers a basic simulation scenario, which only takes three basic processes into account,
namely Electron Ionisation (EI), Radiative Recombination (RR) and Dielectronic Recombination
(DR). DR is only included on demand and depends on the availability of corresponding data
describing the transitions. Tables with data for KLL type transitions are included in the python
package.

Setting up such a simulation and looking at its results is straightforward.

.. plot :: examples/ex_basic_simulation_1.py
    :include-source:

The simulation above assumes that all ions start in charge state 1+. One can easily simulate
the continuous injection of a neutral gas by activating the flag for
Continuous Neutral Injection (CNI).

.. plot :: examples/ex_basic_simulation_2.py
    :include-source:
