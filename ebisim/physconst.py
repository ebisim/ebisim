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

##### Table with factors for LOTZ formula
_LOTZ_TABLE = {
    "1s1" : (4.00, 0.60, 0.56),
    "1s2" : (4.00, 0.75, 0.50),
    "2p1" : (3.80, 0.60, 0.40),
    "2p2" : (3.50, 0.70, 0.30),
    "2p3" : (3.20, 0.80, 0.25),
    "2p4" : (3.00, 0.85, 0.22),
    "2p5" : (2.80, 0.90, 0.20),
    "2p6" : (2.60, 0.92, 0.19),
    "3d1" : (3.70, 0.60, 0.40),
    "3d2" : (3.40, 0.70, 0.30),
    "3d3" : (3.10, 0.80, 0.25),
    "3d4" : (2.80, 0.85, 0.20),
    "3d5" : (2.50, 0.90, 0.18),
    "3d6" : (2.20, 0.92, 0.17),
    "3d7" : (2.00, 0.93, 0.16),
    "3d8" : (1.80, 0.94, 0.15),
    "3d9" : (1.60, 0.95, 0.14),
    "3d10": (1.40, 0.96, 0.13),
    "2s1" : (4.00, 0.30, 0.60),
    "2s2" : (4.00, 0.50, 0.60),
    "3p1" : (4.00, 0.35, 0.60),
    "3p2" : (4.00, 0.40, 0.60),
    "3p3" : (4.00, 0.45, 0.60),
    "3p4" : (4.00, 0.50, 0.50),
    "3p5" : (4.00, 0.55, 0.45),
    "3p6" : (4.00, 0.60, 0.40),
    "4d1" : (4.00, 0.30, 0.60),
    "4d2" : (3.80, 0.45, 0.50),
    "4d3" : (3.50, 0.60, 0.40),
    "4d4" : (3.20, 0.70, 0.30),
    "4d5" : (3.00, 0.80, 0.25),
    "4d6" : (2.80, 0.85, 0.20),
    "4d7" : (2.60, 0.90, 0.18),
    "4d8" : (2.40, 0.92, 0.17),
    "4d9" : (2.20, 0.93, 0.16),
    "4d10": (2.00, 0.94, 0.15),
    "3s1" : (4.00, 0.00, 0.00),
    "3s2" : (4.00, 0.30, 0.60),
    "np1" : (4.00, 0.00, 0.00),
    "np2" : (4.00, 0.00, 0.00),
    "np3" : (4.00, 0.20, 0.60),
    "np4" : (4.00, 0.30, 0.60),
    "np5" : (4.00, 0.40, 0.60),
    "np6" : (4.00, 0.50, 0.50),
    "nd1" : (4.00, 0.00, 0.00),
    "nd2" : (4.00, 0.20, 0.60),
    "nd3" : (3.80, 0.30, 0.60),
    "nd4" : (3.60, 0.45, 0.50),
    "nd5" : (3.40, 0.60, 0.40),
    "nd6" : (3.20, 0.70, 0.30),
    "nd7" : (3.00, 0.80, 0.25),
    "nd8" : (2.80, 0.85, 0.20),
    "nd9" : (2.60, 0.90, 0.18),
    "nd10": (2.40, 0.92, 0.17),
    "ns1" : (4.00, 0.00, 0.00),
    "ns2" : (4.00, 0.00, 0.00),
    "nf1" : (3.70, 0.60, 0.40),
    "nf2" : (3.40, 0.70, 0.30),
    "nf3" : (3.10, 0.80, 0.25),
    "nf4" : (2.80, 0.85, 0.20),
    "nf5" : (2.50, 0.90, 0.18),
    "nf6" : (2.20, 0.92, 0.17),
    "nf7" : (2.00, 0.93, 0.16),
    "nf8" : (1.80, 0.94, 0.15),
    "nf9" : (1.60, 0.95, 0.14),
    "nf10": (1.40, 0.96, 0.13),
    "nf11": (1.30, 0.96, 0.12),
    "nf12": (1.20, 0.97, 0.12),
    "nf13": (1.10, 0.97, 0.11),
    "nf14": (1.00, 0.97, 0.11)
} #: Dictionary with the Lotz formula factors for different shells, a in units of <1.0e-14cm**2/eV>
