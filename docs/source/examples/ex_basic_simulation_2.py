"""Example: Basic simulation with CNI"""

from matplotlib.pyplot import show
import ebisim as eb

element = eb.Element("Argon") # element that is to be charge bred
j = 200 # current density in A/cm^2
e_kin = 2470 # electron beam energy in eV
t_max = 1 # length of simulation in s

result = eb.basic_simulation(
    element=element,
    j=j,
    e_kin=e_kin,
    t_max=t_max,
    CNI=True # activate CNI
)

# Since the number of ions is constantly increasing with CNI,
# it is helpful to only plot the relative distribution at each time step
# This is easily achieved by setting the corresponding flag
result.plot_charge_states(relative=True)

show()
