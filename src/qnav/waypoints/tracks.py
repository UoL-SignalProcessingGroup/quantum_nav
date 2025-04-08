"""
=========
tracks.py
=========

:summary:
    Provides default waypoint generation for testing tracks.
    This module contains various functions for generating different default
    waypoint tracks. This is an alternative instead of using user supplied
    trajectories. This is namely used for testing, but can also be used for
    rapid experimentation when a custom trajectory is unavailable.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""


from qnav.util.transformations import ned2lla_vec

import numpy as np
import math


def generate_racetrack(frequency: float = 1000.0,
                       max_time: float = 500,
                       lap_time: float = 500,
                       radius_x: float = 1000.0,
                       radius_y: float = 1000.0,
                       radius_z: float = 5.0,
                       theta_x: float = 0.0,
                       theta_y: float = 0.0,
                       theta_z: float = 0.0,
                       ref_lat: float = 0.0,
                       ref_lon: float = 0.0,
                       ref_alt: float = 0.0,
                       num_var_y: float = 2,
                       num_var_z: float = 3) -> np.ndarray:
    """
    Generates waypoints for a racetrack of requested dimensions.
    This function produces waypoints at a requested frequency that represent
    a simple looping path. The produced waypoints can be then used with
    previously implemented waypoint functions, essentially (efficiently)
    pre-calculating the values at each position along the racetrack.

    :param frequency: The simulation's update frequency (in Hz). This is used
        for calculating the number of intermediate points and total number of
        waypoints to return.
    :type frequency: float

    :param max_time: The time at which the simulation is to end (in seconds).
        No waypoints will be generated for points after this time
    :type max_time: float

    :param lap_time: The time period (in seconds) for one circuit of the track.
        This is used to calculate how fast the vehicle is moving around the
        generated racetrack.
    :type lap_time: float

    :param radius_x: The size of the racetrack along the x-direction (in metres).
    :type radius_x: float

    :param radius_y: The size of the racetrack along the y-direction (in metres).
    :type radius_y: float

    :param radius_z: The height variations of the racetrack (in metres).
    :type radius_z: float

    :param theta_x: The starting X position on the circuit.
    :type theta_x: float

    :param theta_y: The starting Y position on the circuit.
    :type theta_y: float

    :param theta_z: The starting Z position on the circuit.
    :type theta_z: float

    :param ref_lat: The reference latitude of the centre position of the
        circuit (in degrees).
    :type ref_lat: float

    :param ref_lon: The reference longitude of the centre position of the
        circuit (in degrees).
    :type ref_lon: float

    :param ref_alt: The reference altitude of the centre position of the
        circuit (in degrees).
    :type ref_alt: float

    :param num_var_y: Number of periods of y variations for each circuit in x.
    :type num_var_y: float

    :param num_var_z: Number of periods of height variations for each circuit in x.
    :type num_var_z: float

    :return: The calculated latitude, longitude and altitude for each time
        step from 0 to the maximum time requested.
    :rtype: numpy.ndarray (n-by-4 elements)
    """

    # Calculate the number of waypoints to produce.
    num_of_waypoints = int(math.floor(frequency * max_time))

    # Obtain the timestamps for each waypoint.
    times = np.linspace(0.0, max_time, num_of_waypoints)

    # Obtain the angles along the racetrack.
    omega_x = 2.0 * math.pi / lap_time
    omega_y = num_var_y * omega_x
    omega_z = num_var_z * omega_x

    # Calculate the position along the racetrack
    x = radius_x * np.sin(omega_x * times + theta_x) - radius_x * np.sin(theta_x)
    y = radius_y * np.sin(omega_y * times + theta_y) - radius_y * np.sin(theta_y)
    z = radius_z * np.sin(omega_z * times + theta_z) - radius_z * np.sin(theta_z)
    positions_ned = np.column_stack([x, y, z])

    # Specify the centre reference position.
    # TODO: Check dimensions of ref_lla vector
    ref_lla = np.array([ref_lat, ref_lon, ref_alt])

    # Convert the NED values to LLA values.
    positions_lla = ned2lla_vec(positions_ned, ref_lla)

    # Finally, return the calculated waypoints
    return np.column_stack([times, positions_lla])
