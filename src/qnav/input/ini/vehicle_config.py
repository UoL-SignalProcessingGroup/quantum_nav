"""
=================
vehicle_config.py
=================

:summary:
    Functions related to initialising objects from vehicle config section.
    A collection of functions used for reading content under the 'vehicle'
    section within the configuration and initialising the corresponding
    objects.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""


import numpy as np

from functools import partial

from qnav.input.config_handler import ConfigHandler
from qnav.input.ini.rng_config import get_waypoint_seed
from qnav.measurement.platform import SensorAxis
from qnav.waypoints.vehicle import Vehicle

# The shared section name to use
__SECTION_ID: str = "Vehicle"


def get_sensor_axis(config: ConfigHandler) -> SensorAxis:
    """
    Generates the sensor axis for the vehicle, based on configuration settings.
    Initialises and returns the true sensor axis for the platform.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The sensor axis instance for the vehicle.
    :rtype: SensorAxis
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    # Get the requested angles of the sensor placement (in degrees).
    sensor_angle_p = get_float("sensorAngleX", 0)
    sensor_angle_q = get_float("sensorAngleY", 0)
    sensor_angle_r = get_float("sensorAngleZ", 0)

    # Get the requested angles of the lever arm (in degrees).
    lever_arm_x = get_float("leverArmAngleX", 0)
    lever_arm_y = get_float("leverArmAngleY", 0)
    lever_arm_z = get_float("leverArmAngleZ", 0)

    # Produce required orientation vectors.
    sensor_angles = np.array([sensor_angle_p, sensor_angle_q, sensor_angle_r])
    lever_arm = np.array([lever_arm_x, lever_arm_y, lever_arm_z])

    # Initialise and return the sensor axis
    return SensorAxis(sensor_angles, lever_arm)


def get_vehicle_speed(config: ConfigHandler) -> float:
    """
    Obtains the average vehicle speed from the configuration.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The vehicle's average ground speed (in m/s).
    :rtype: float
    """
    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    # return get_float("vehicleSpeed", 100)
    return get_float("averageSpeed", 100)


def get_vehicle_model(config: ConfigHandler) -> Vehicle:
    """
    Generates a vehicle model based on configuration settings.
    Initialises and returns a vehicle model with the parameters defined in
    the given configuration file.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A vehicle model with the parameters from configuration settings.
    :rtype: Vehicle
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_str = partial(config.get_str, __SECTION_ID)
    rand_seed = get_waypoint_seed(config)

    return Vehicle(
        get_str("vehicleName", "Unnamed"),
        get_float( "maxTurnRate", 5.0),
        get_float( "maxAcceleration", 2.0),
        get_float( "maxDeceleration", 1.0),
        get_float( "timeDelayAcceleration", 1.0),
        get_float( "timeDelayTurning", 5.0),
        get_float( "pitchConstantOffset", 0.0),
        get_float( "pitchOscPeriod", 0.0),
        get_float( "pitchOscMagnitude", 0.0),
        get_float( "pitchMax", 90.0),
        get_float( "rollConstantOffset", 0.0),
        get_float( "rollOscPeriod", 0.0),
        get_float( "rollOscMagnitude", 0.0),
        get_float( "rollMax", 0.0),
        get_float( "rollTurnRate", 0.0),
        get_float( "vibrationNoiseAcceleration", 0.0),
        get_float( "vibrationNoiseAngleRates", 0.0),
        get_float( "vibrationDampingPeriod", 0.0),
        get_float( "variationsPerSecond1", 0.005),
        get_float( "variationsPerSecond2", 0.01),
        rand_seed=rand_seed
    )
