"""
Central module for holding physical constants used in the simulation code
"""

import scipy.constants

##### Physical constants
Q_E = scipy.constants.elementary_charge #: Elementary charge <Unit: C>
M_E = scipy.constants.electron_mass #: Electron mass <Unit: kg>
PI = scipy.constants.pi #: Pi
EPS_0 = scipy.constants.epsilon_0 #: Vacuum permittivity <Unit: F/m>
K_B = scipy.constants.Boltzmann #: Boltzmann constant  <Unit: J/K>
C_L = scipy.constants.speed_of_light #: Speed of light in vacuum  <Unit: m/s>
ALPHA = scipy.constants.alpha #: Fine structure constant
HBAR = scipy.constants.hbar #: Reduced Planck constant  <Unit: J*s>
M_P = scipy.constants.proton_mass #: Proton mass  <Unit: kg>

##### Derived constants
M_E_EV = M_E * C_L**2 / Q_E #: Electron mass equivalent <Unit: eV>
RY_EV = scipy.constants.Rydberg * C_L * 2 * PI * HBAR / Q_E #: Rydberg energy  <Unit: eV>
COMPT_E_RED = HBAR / (M_E * C_L) #: Reduced electron compton wavelength  <Unit: m>


##### Numerical Thresholds
MINIMAL_DENSITY = 1e-10 #: Minimal particle number density <Unit: 1/m^3>
MINIMAL_KBT = 1e-3 #: Minimal temperature equivalent  <Unit: eV>
