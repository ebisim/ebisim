"""
Central module for holding physical constants used in the simulation code
"""

import scipy.constants

##### Physical constants
Q_E = scipy.constants.elementary_charge #: Elementary charge
M_E = scipy.constants.electron_mass #: Electron mass
PI = scipy.constants.pi #: Pi
EPS_0 = scipy.constants.epsilon_0 #: Vacuum permittivity
K_B = scipy.constants.Boltzmann #: Boltzmann constant
C_L = scipy.constants.speed_of_light #: Speed of light in vacuum
ALPHA = scipy.constants.alpha #: Fine structure constant
HBAR = scipy.constants.hbar #: Reduced Planck constant
M_P = scipy.constants.proton_mass #: Proton mass

##### Derived constants
M_E_EV = M_E * C_L**2 / Q_E #: Electron mass in eV
RY_EV = scipy.constants.Rydberg * C_L * 2 * PI * HBAR / Q_E #: Rydberg energy (approx. 13.6eV)
COMPT_E_RED = HBAR / (M_E * C_L) #: Reduced electron compton wavelength


##### Numerical Thresholds
MINIMAL_DENSITY = 1e-10 #m^-3 minimal particle number density
MINIMAL_KBT = 1e-3 #eV minimal temperature equivalent
