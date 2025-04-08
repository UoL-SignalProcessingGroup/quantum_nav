"""
===================
racetrack_config.py
===================

:summary:
    Functions related to initialising objects from racetrack config section.
    A collection of functions used for reading content under the 'racetrack'
    section within the configuration and initialising the corresponding
    objects.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""


from qnav.input.config_handler import ConfigHandler
from qnav.waypoints.tracks import generate_racetrack
from functools import partial

import qnav.input.ini.waypoint_config as waypoint
import numpy as np

# The shared section name to use
__SECTION_ID: str = "Racetrack"


def get_racetrack(config: ConfigHandler) -> np.ndarray:
    """
    Returns racetrack points based on configuration values.
    Generates and returns the base points of a looping racetrack trajectory,
    based on the values given in the configuration file.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A column array holding the base point times, latitudes,
        longitudes and altitude respectively.
    :rtype: np.ndarray
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    freq = waypoint.get_frequency(config)
    circuit_time = get_float("circuitTime", 500.0)
    max_time = get_float("maxTime", circuit_time)

    radius_x = get_float("distanceX", 1000.0)
    radius_y = get_float("distanceY", 1000.0)
    radius_z = get_float("distanceZ", 5.0)

    theta_x = get_float("startX", 0.0)
    theta_y = get_float("startY", 0.0)
    theta_z = get_float("startZ", 0.5)

    lat_origin = get_float("latOrigin", 0)
    lon_origin = get_float("lonOrigin", 0)
    alt_origin = get_float("altOrigin", 0)

    num_var_y = get_int("numVariationsY", 2)
    num_var_z = get_int("numVariationsZ", 3)

    # Calculate the racetrack positions to return.
    return generate_racetrack(
        freq, max_time, circuit_time,
        radius_x, radius_y, radius_z,
        theta_x, theta_y, theta_z,
        lat_origin, lon_origin, alt_origin,
        num_var_y, num_var_z)
