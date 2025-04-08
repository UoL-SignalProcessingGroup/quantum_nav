"""
===========
coriolis.py
===========

:summary:
    Responsible for the calculation of coriolis effects in gravity modelling.
    This module provides the necessary function for including the coriolis
    effects in gravity calculations. As these are commonly used, they
    have been placed in a module of their own.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

from qnav.util.transformations import cross_prod_xy

import qnav.util.constants as const
import numpy as np
import math


def apply_coriolis_terms(lat: float,
                         g_z: float,
                         alt_offset: float = 0.0,
                         geoid_offset: float = 0.0) -> np.ndarray:
    """
    Calculates the coriolis effects for given position and gravity values.
    A function commonly used in gravity models for including the coriolis
    effect in gravitational acceleration calculations.

    :param lat: The latitude of the position of interest (in decimal degrees).
    :type lat: float

    :param g_z: The calculated vertical gravity acceleration component (in m/s^2).
    :type g_z: float

    :param alt_offset: The elevation offset from the WGS-84 ellipsoid (in metres).
    :type alt_offset: float

    :param geoid_offset: The offset height from the reference geoid (in metres).
    :type geoid_offset: float

    :return: The gravitation acceleration including coriolis effects.
    :rtype: np.ndarray
    """

    # State the Earth's Equator and Polar radii (metres).
    a = const.POLAR_AXIS_A
    b = const.POLAR_AXIS_B

    # Convert latitude from degrees to radians.
    lat_rad = math.radians(lat)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    # The approximate altitude from the earth.
    h_correction = alt_offset - geoid_offset

    # Define Angular velocity for Earth's rotation (in local NED axes).
    omega_e = const.OMEGA_E * np.array([cos_lat, 0, -sin_lat])

    # Obtain the radius of the earth for the given latitude.
    r_e = math.sqrt(((a ** 2 * cos_lat) ** 2 + (b ** 2 * sin_lat) ** 2) /
                    ((a * cos_lat) ** 2 + (b * sin_lat) ** 2))

    # Calculate the gravity vector modified by Earth's rotation (metres/sec^2).
    rotation_vector = -np.array([0, 0, r_e + alt_offset])
    g = np.array([0, 0, g_z])

    # Finally, return the calculated gravity vector
    tmp = cross_prod_xy(omega_e, cross_prod_xy(omega_e, rotation_vector))
    g_xyz = g * ((r_e / (r_e + h_correction)) ** 2) - tmp
    return g_xyz


def apply_coriolis_terms_vec(lat: np.ndarray,
                             g_z: np.ndarray,
                             alt_offset: np.ndarray = None,
                             geoid_offset: np.ndarray = None):
    """
    Vectorised calculation of coriolis effects for gravity modelling.
    Calculates the coriolis effects for given position and gravity values.
    A function commonly used in gravity models for including the coriolis
    effect in gravitational acceleration calculations.

    :param lat: The latitude of the position of interest (in decimal degrees).
    :type lat: np.ndarray

    :param g_z: The calculated vertical gravity acceleration component (in m/s^2).
    :type g_z: p.ndarray

    :param alt_offset: The elevation offset from the WGS-84 ellipsoid (in metres).
    :type alt_offset: np.ndarray | None

    :param geoid_offset: The offset height from the reference geoid (in metres).
    :type geoid_offset: np.ndarray | None

    :return: The gravitation acceleration including coriolis effects.
    :rtype: np.ndarray
    """

    # Validate input
    assert np.shape(lat) == np.shape(g_z), \
        "Input latitude and acceleration arrays must be same size!"

    assert alt_offset is None or np.shape(lat) == np.shape(alt_offset), \
        "Altitude offset size must be same size as other arrays!"

    assert geoid_offset is None or np.shape(lat) == np.shape(geoid_offset), \
        "Geoid offset size must be same size as other arrays!"

    # State the Earth's Equator and Polar radii (metres).
    a = const.POLAR_AXIS_A
    b = const.POLAR_AXIS_B

    # Convert latitude from degrees to radians.
    num_of_rows = np.size(lat)
    output_size = (num_of_rows, 3)

    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    # Use zero as default value for alt:
    if alt_offset is None:
        alt_offset = 0

    # The approximate altitude from the earth:
    if geoid_offset is not None:
        h_correction = alt_offset - geoid_offset
    else:
        h_correction = alt_offset

    # Define Angular velocity for Earth's rotation (in local NED axes).
    omega_e = np.zeros(output_size)
    omega_e[:, 0] = const.OMEGA_E * cos_lat
    omega_e[:, 2] = const.OMEGA_E * -sin_lat

    # Obtain the radius of the earth for the given latitude.
    r_e = np.sqrt(((a ** 2 * cos_lat) ** 2 + (b ** 2 * sin_lat) ** 2) /
                  ((a * cos_lat) ** 2 + (b * sin_lat) ** 2))

    # Calculate the gravity vector modified by Earth's rotation (metres/sec^2).
    rotation_vector = np.zeros(output_size)
    rotation_vector[:, 2] = -(r_e + alt_offset)
    g = np.zeros(output_size)
    g[:, 2] = g_z

    # Finally, return the calculated gravity vector.
    g_xyz = g * ((r_e / (r_e + h_correction)) ** 2).reshape(
        [-1, 1]) - np.cross(omega_e, np.cross(omega_e, rotation_vector))
    return g_xyz


# if __name__ == '__main__':
#     print(apply_coriolis_terms(52, 9.81, 1000, 0))
#     print(apply_coriolis_terms(53, 9.82, 1000, 0))
#     print(apply_coriolis_terms(54, 9.83, 1000, 0))
#
#     lat_points = np.array([52, 53, 54])
#     gz_points = np.array([9.81, 9.82, 9.83])
#     alt_points = np.zeros(3) + 1000
#     print(apply_coriolis_terms_vec(lat_points, gz_points, alt_points))
