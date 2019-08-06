"""
Central module for holding physical constants used in the simulation code.
"""

import scipy.constants

##### Physical constants
Q_E = scipy.constants.elementary_charge #: <C> Elementary charge
M_E = scipy.constants.electron_mass #: <kg> Electron mass
PI = scipy.constants.pi #: Pi
EPS_0 = scipy.constants.epsilon_0 #: <F/m> Vacuum permittivity
K_B = scipy.constants.Boltzmann #: <J/K> Boltzmann constant
C_L = scipy.constants.speed_of_light #: <m/s> Speed of light in vacuum
ALPHA = scipy.constants.alpha #: Fine structure constant
HBAR = scipy.constants.hbar #: <J*s> Reduced Planck constant
M_P = scipy.constants.proton_mass #: <kg> Proton mass

##### Derived constants
M_E_EV = M_E * C_L**2 / Q_E #: <eV> Electron mass equivalent
RY_EV = scipy.constants.Rydberg * C_L * 2 * PI * HBAR / Q_E #: <eV> Rydberg energy
COMPT_E_RED = HBAR / (M_E * C_L) #: <m> Reduced electron compton wavelength


##### Numerical Thresholds
MINIMAL_DENSITY = 1e-10 #: <1/m^3> Minimal particle number density
MINIMAL_KBT = 1e-3 #: <eV> Minimal temperature equivalent
