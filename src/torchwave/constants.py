"""Physical and astrophysical constants for torchwave.

All constants are defined using standard SI units and geometric units.
Scientific Provenance: CODATA 2018, IAU 2015 Resolution B3.
"""

import math

# Speed of light in vacuum (m/s) - exact SI definition
C_SI: float = 299792458.0

# Newtonian constant of gravitation (m^3 kg^-1 s^-2) - CODATA 2018
G_SI: float = 6.67430e-11

# Nominal solar mass parameter (kg) - IAU 2015 Resolution B3: (GM)_nom / G
MSUN_SI: float = 1.9884098706980507e30

# Solar mass in time units: G * M_sun / c^3 (seconds)
MSUN_SEC: float = G_SI * MSUN_SI / (C_SI**3)  # ~ 4.925490947641267e-06 s

# One Megaparsec in meters (m)
MPC_SI: float = 3.085677581491367e22

# Euler-Mascheroni constant
EULER_GAMMA: float = 0.57721566490153286060651209008240243104215933593992

# Mathematical constants
PI: float = math.pi
