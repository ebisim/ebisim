"""Example: Energy scan accross ionisation threshold"""

from matplotlib.pyplot import show
import numpy as np
import ebisim as eb

# For energy scans we have to create a python dictionary with the simulation parameters
# excluding the electron beam energy.
# The dictionary keys have to correspond to the argument and keyword argument names of the
# simulation function that one wants to use, these names can be found in the API Reference

sim_kwargs = dict(
    element=eb.get_element("Potassium"),  # element that is to be charge bred
    j=200,  # current density in A/cm^2
    t_max=100  # length of simulation in s
)
scan = eb.energy_scan(
    sim_func=eb.basic_simulation,  # the function handle of the simulation has to be provided
    sim_kwargs=sim_kwargs,  # Here the general parameters are injected
    energies=np.arange(4500, 4700),  # The sampling energies
    parallel=True  # Speed up by running simulations in parallel
)

# The scan result object holds the relevant data which could be inspected manually
# It also offers convenience methods for plotting

# One thing that can be done is to plot the abundance of certain charge states
# at a given time, dependent on the energies
# In order not to plot all the different charge states we can supply a filter list
scan.plot_abundance_at_time(t=20, cs=[17, 18])

# Alternatively, one can create a plot to see how a given charge states depends on the breeding time
# and the electron beam energy
scan.plot_abundance_of_cs(cs=18)

show()
