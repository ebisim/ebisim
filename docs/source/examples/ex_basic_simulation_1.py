"""Example: Basic simulation with DR"""

from matplotlib.pyplot import show
import ebisim as eb

# For the basic simulation only a small number of parameters needs to be provided

element = eb.get_element("Potassium") # element that is to be charge bred
j = 200 # current density in A/cm^2
e_kin = 2470 # electron beam energy in eV
t_max = 1 # length of simulation in s
dr_fwhm = 15 # effective energy spread, widening the DR resonances in eV, optional

result = eb.basic_simulation(
    element=element,
    j=j,
    e_kin=e_kin,
    t_max=t_max,
    dr_fwhm=dr_fwhm,
)

# The result object holds the relevant data which could be inspected manually
# It also offers convenience methods to plot the charge state evolution

result.plot_charge_states()

show()
