import ebisim as eb
from matplotlib.pyplot import show
import numpy as np

sim_kwargs= dict(
    element=eb.Element("Potassium"), # element that is to be charge bred
    j=200, # current density in A/cm^2
    t_max=1, # length of simulation in s
    dr_fwhm=15 # This time DR has to be activated by setting an effective line width
)
scan = eb.energy_scan(
    sim_func=eb.basic_simulation,
    sim_kwargs=sim_kwargs,
    energies=np.arange(2400, 2700), # The sampling energies cover the KLL band
    parallel=True
)

scan.plot_abundance_at_time(t=.1, cs=[14,15,16,17])
scan.plot_abundance_of_cs(cs=16)

show()
