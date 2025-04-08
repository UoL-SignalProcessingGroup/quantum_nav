"""
===================
waypoints_config.py
===================

:summary:
    Functions related to initialising objects from waypoints config section.
    A collection of functions used for reading content under the 'waypoints'
    section within the configuration and initialising the corresponding
    objects. This is used for generating or importing trajectory data.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""

from functools import partial

from qnav.gravity.base import GravityModel
from qnav.input.config_handler import ConfigHandler
from qnav.input.config_handler import NavConfigError
from qnav.input.ini.racetrack_config import get_racetrack
from qnav.input.ini.vehicle_config import get_vehicle_speed
from qnav.waypoints.generation import read_csv_waypoints
from qnav.waypoints.trajectory import Trajectory
from qnav.waypoints.trajectory import Waypoints
from qnav.waypoints.trajectory import cumulative_distance
from qnav.waypoints.vehicle import Vehicle


# The shared section name to use
__SECTION_ID: str = "Waypoints"

# The default mode for loading waypoints
__DEFAULT_WAYPOINT_MODE: str = "normal"

# The default frequency to generate waypoints at
__DEFAULT_FREQUENCY: float = 100

# The default interpolation method for waypoint look-up
__DEFAULT_INTERP: str = "pchip"


def get_frequency(config: ConfigHandler) -> float:
    """
    Returns the frequency that base waypoints are to be generated at waypoint.
    This is the frequency (in Hz) that base positions along the generated
    trajectory are to be produced at.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The base frequency for waypoint positions (in Hz).
    :rtype: float
    """
    get_float = partial(config.get_float, __SECTION_ID)
    return get_float("waypointFrequency", 100)


def get_interp_method(config: ConfigHandler) -> str:
    """
    Returns the selected interpolation method for waypoint look-up.
    This is the string describing the selected interpolation method to use
    for obtaining intermediate waypoint records.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The selected interpolation method for waypoint look-up.
    :rtype: str
    """
    get_str = partial(config.get_str, __SECTION_ID)
    return get_str("waypointInterpolation", __DEFAULT_INTERP)


def get_waypoint_mode(config: ConfigHandler) -> str:
    """
    Returns the selected waypoint generation mode.
    This is the ID string used to select what method will be used for
    generating waypoints (i.e. 'ini', 'csv', 'racetrack' etc..).

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The method ID for generating waypoints.
    :rtype: str
    """
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    return get_id("waypointMode", __DEFAULT_WAYPOINT_MODE)


def _generate_ini_trajectory(config: ConfigHandler, vehicle: Vehicle, gravity: GravityModel) -> Trajectory:
    """
    Generates and returns trajectory from given INI values.
    This initialises and returns a trajectory using the base point
    information provided in the (base or sub) configuration file.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param vehicle: The vehicle instance to use for parameter values.
    :type vehicle: Vehicle

    :param gravity: The gravity model to use for parameter values.
    :type gravity: GravityModel

    :return: Constructed trajectory using INI waypoint parameters.
    :rtype: Trajectory
    """

    get_csv = partial(config.get_csv_numeric, __SECTION_ID)
    average_speed = get_vehicle_speed(config)
    frequency = get_frequency(config)

    times = get_csv("waypointTimes", None)
    lat_points = get_csv("waypointLat")
    lon_points = get_csv("waypointLon")
    alt_points = get_csv("waypointAlt")

    interp_method = get_interp_method(config)

    if times is None:
        cum_dist = cumulative_distance(lat_points, lon_points, alt_points)
        times = cum_dist / average_speed
        # times = np.round(times, 4)

    waypoints = Waypoints(times, lat_points, lon_points, alt_points)
    return Trajectory(waypoints, vehicle, gravity, frequency, interp_method)


def _generate_csv_trajectory(config: ConfigHandler, vehicle: Vehicle, gravity: GravityModel) -> Trajectory:
    """
    Generates and returns trajectory from given CSV file.
    This initialises and returns a trajectory using the base point
    information provided in linked CSV file.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param vehicle: The vehicle instance to use for parameter values.
    :type vehicle: Vehicle

    :param gravity: The gravity model to use for parameter values.
    :type gravity: GravityModel

    :return: Constructed trajectory using CSV waypoint file.
    :rtype: Trajectory
    """

    get_path = partial(config.get_value_filepath, __SECTION_ID)
    get_bool = partial(config.get_bool, __SECTION_ID)

    average_speed = get_vehicle_speed(config)
    frequency = get_frequency(config)

    file_path = get_path("waypointCSV")
    invert_lat_lon = get_bool("invertCSVLatLon", False)
    ignore_times = get_bool("ignoreCSVTimes", False)

    try:

        read_data = read_csv_waypoints(file_path)
        times = read_data["times"]
        lat_points = read_data["latitude"]
        lon_points = read_data["longitude"]
        alt_points = read_data["altitude"]

        if invert_lat_lon:
            lat_points, lon_points = lon_points, lat_points

        if times is None or ignore_times:
            cum_dist = cumulative_distance(lat_points, lon_points, alt_points)
            times = cum_dist / average_speed

        waypoints = Waypoints(times, lat_points, lon_points, alt_points)
        return Trajectory(waypoints, vehicle, gravity, frequency)

    except Exception as e:
         raise NavConfigError(__SECTION_ID, "waypointCSV",
                              "Invalid Data", repr(e))


def _generate_racetrack(config: ConfigHandler, vehicle: Vehicle, gravity: GravityModel) -> Trajectory:
    """
    Generates and returns trajectory from racetrack model.
    This initialises and returns a trajectory using the racetrack model
    parameters given in the configuration file.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param vehicle: The vehicle instance to use for parameter values.
    :type vehicle: Vehicle

    :param gravity: The gravity model to use for parameter values.
    :type gravity: GravityModel

    :return: Constructed trajectory using racetrack model.
    :rtype: Trajectory
    """

    frequency = get_frequency(config)
    racetrack = get_racetrack(config)
    times = racetrack[:, 0]
    lat_points = racetrack[:, 1]
    lon_points = racetrack[:, 2]
    alt_points = racetrack[:, 3]

    waypoints = Waypoints(times, lat_points, lon_points, alt_points)
    return Trajectory(waypoints, vehicle, gravity, frequency)


def generate_trajectory(config: ConfigHandler, vehicle: Vehicle, gravity: GravityModel) -> Trajectory:
    """
    Obtains and returns initialised trajectory specified by configuration.
    Generates (or loads) the requested trajectory type.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param vehicle: The vehicle model to use for maneuvering constraints.
    :type vehicle: Vehicle

    :param gravity: The gravity model to use for gravitational calculations.
    :type gravity: GravityModel

    :return: The initialised trajectory with the requested type.
    :rtype: Trajectory
    """

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    mode: str = get_id("waypointMode", __DEFAULT_WAYPOINT_MODE)

    match (mode.strip().casefold()):

        case "config":
            return _generate_ini_trajectory(config, vehicle, gravity)

        case "csv":
            return _generate_csv_trajectory(config, vehicle, gravity)

        case "racetrack":
            return _generate_racetrack(config, vehicle, gravity)

        case _:
            raise NavConfigError(
                __SECTION_ID, "waypointMode", "Unknown mode",
                f"Unrecognized waypoint mode: {mode}")
