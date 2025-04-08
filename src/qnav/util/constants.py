"""
============
constants.py
============

:summary:
    A list of calculation constants that are across the toolbox.
    This shared module contains a collection of geological constants that are
    repeatedly used by multiple functions within the toolbox

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

from math import sqrt

#: :math:`g` - The average gravitation acceleration.
G = 9.81

#: :math:`f` - Flattening measure of the WGS84 spheroid.
F = 1 / 298.257223563

#: :math:`a` - Semi-major axis equator radius (m).
POLAR_AXIS_A = 6378137.0

# : :math:`b` - Semi-minor axis pole radius (m).
POLAR_AXIS_B = POLAR_AXIS_A * (1 - F)

#: :math:`R_e` - The average radius of the Earth (m).
R_E = (POLAR_AXIS_A + POLAR_AXIS_B) / 2

#: #: :math:`e^2` - The squared first eccentricity value for the Earth.
ECC_SQ = (POLAR_AXIS_A ** 2 - POLAR_AXIS_B ** 2) / POLAR_AXIS_A ** 2

#: :math:`e` - The first eccentricity value for the Earth.
ECC = sqrt(ECC_SQ)

#: :math:`e'` - The second eccentricity value for the Earth.
ECC_PRIME = sqrt((POLAR_AXIS_A ** 2 - POLAR_AXIS_B ** 2) / POLAR_AXIS_B ** 2)

#: :math:`{\gamma}_e` - Normal gravity at Equator (m/s**2).
G_E = 9.780325335903891718546

#: :math:`{\gamma}_b` - Normal gravity at Poles (m/s**2).
G_P = 9.83218493786340046183

#: :math:`{\omega}_e` - Approximate rotation rate of the Earth (m/s**2).
OMEGA_E = 7.2921159e-5

#: :math:`GM` - The Geocentric gravitational constant (m**3/s**2)
GM = 3986004.4188e8

#: :math:`M` - Somigliana constant, used for calculation normal gravity.
M = (OMEGA_E**2 * POLAR_AXIS_A**2 * POLAR_AXIS_B) / GM
