"""
=============
gps_config.py
=============

:summary:
    Functions related to initialising the GPS sensor and its fusion method.
    A collection of functions used for reading content under the 'GPS' section
    within the configuration and initialising the corresponding objects.
    Specifically, this involves handling the corresponding Rinex files,
    setting-up satellite instances, sensor instances and fusion methods.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""


import numpy as np

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

from qnav.gps.zones import FlaggedPolygonGroup
from qnav.gps.zones import FlaggedPolygon

from qnav.input.config_handler import ConfigHandler
from qnav.input.config_handler import NavConfigError
from qnav.input.ini.database_config import get_gps_dir
from qnav.input.ini.rng_config import get_error_seed
from qnav.estimation.state import EstimatedState
from qnav.estimation.state import KalmanEstimatedState
from qnav.gps import ephemeris
from qnav.gps.fusion import GpsFusion
from qnav.gps.fusion import GpsFixedGainFusion
from qnav.gps.fusion import GpsLooseFusion
from qnav.gps.fusion import GpsTightFusion
from qnav.gps.fusion import calculate_cov_mat
from qnav.gps.satellite import Satellite
from qnav.gps.sensor import GpsSensor

# The shared section name to use
__SECTION_ID: str = "GPS"

# The default option for if the GPS is enabled
__DEFAULT_IS_USED: bool = False

# The default option for allowing downloads.
__DEFAULT_ALLOW_DOWNLOADS: bool = True

# The default year to use for GPS data.
__DEFAULT_GPS_YEAR: int = 2020

# The default month to use for GPS data.
__DEFAULT_GPS_MONTH: int = 1

# The default day to use for GPS data.
__DEFAULT_GPS_DAY: int = 1

# The default rho noise to use.
__DEFAULT_RHO_NOISE: float = 0.0

# The default rho rate to apply.
__DEFAULT_RHO_RATE: float = 0.0

# The default frequency of GPS measurements
__DEFAULT_GPS_FREQ: float = 1.0

# The default fusion method to use
__DEFAULT_FUSION_METHOD: str = "fixedgain"

# The default gain amount for applying fixed gain fusion
__DEFAULT_FIXED_GAIN: float = 5e-2

# The default pseudo-range position error (in metres).
# Used when applying loosely coupled fusion.
__DEFAULT_SIGMA_POSITION: float = 50.0

# The default pseudo-range position error (in metres/second).
# Used when applying loosely coupled fusion.
__DEFAULT_SIGMA_VELOCITY: float = 0.3


def get_gps_date(config: ConfigHandler) -> datetime:
    """
    Obtains the correctly formatted GPS data from the configuration file.
    This acquires the data for which the RINEX data is to be loaded.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A datetime instance corresponding to the matching Rinex date.
    :rtype: datetime
    """

    # Shorthand function for reading integer values from configuration
    get_int = partial(config.get_int, __SECTION_ID)

    # Obtain the date for the GPS data to download.
    gps_year = get_int("gpsTimeYear", __DEFAULT_GPS_YEAR)
    gps_month = get_int("gpsTimeMonth", __DEFAULT_GPS_MONTH)
    gps_day = get_int("gpsTimeDay", __DEFAULT_GPS_DAY)

    try:
        return datetime(gps_year, gps_month, gps_day)

    except ValueError as ve:
        raise NavConfigError(
            __SECTION_ID, "gpsTime*",
            "Invalid Date", str(ve))


def get_rinex_file(config: ConfigHandler, gps_date: datetime) -> Path:
    """
    Obtains the path for the Rinex file from the configuration file.
    Returns the local file system path for where the required Rinex file is
    to be located, using the information described in the configuration file.
    Note a file may not be located at this path if it is not pre-downloaded.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param gps_date: The datetime instance for the corresponding Rinex file.
    :type gps_date: datetime

    :return: The path to where the Rinex file is to be located.
    :rtype: Path
    """

    # Shorthand function for reading boolean values from configuration
    get_bool = partial(config.get_bool, __SECTION_ID)
    allow_downloads = get_bool("gpsAutoDownload", True)
    gps_dir = get_gps_dir(config) / "rinex"

    # gps_date = get_gps_date(config)
    gps_file = ephemeris.get_rinex_path(gps_date, gps_dir)

    if not gps_file.exists():
        if not allow_downloads:
            raise NavConfigError(
                __SECTION_ID, "gpsAutoDownload", "File Missing",
                "The GPS RINEX file for the requested date needs to be downloaded")
        else:
            ephemeris.download_rinex_file(gps_date, gps_dir)

    return gps_file


def get_gps_satellites(config: ConfigHandler) -> list[Satellite]:
    """
    Obtains the GPS satellite instances from the configuration file.
    Extracts the satellite information from the corresponding Rinex file
    described in the configuration file. If the file is missing, an attempt
    to download it will be made (subject to settings). Once read, the
    satellite information will be extracted and releavnt noise profiles
    will be applied.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A list of satellite instances with applied noise profiles.
    :rtype: list[Satellite]
    """

    gps_time = get_gps_date(config)
    gps_file = get_rinex_file(config, gps_time)

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_bool = partial(config.get_bool, __SECTION_ID)

    rho_noise = get_float("gpsRhoNoise", __DEFAULT_RHO_NOISE)
    rho_noise_rate = get_float("gpsRhoRateNoise", __DEFAULT_RHO_RATE)
    allow_downloads = get_bool("gpsAutoDownload", __DEFAULT_ALLOW_DOWNLOADS)

    if not gps_file.exists():
        if allow_downloads:
            ephemeris.download_rinex_file(gps_time, gps_file)
        else:
            raise NavConfigError(
                __SECTION_ID, "gpsTime*", "Missing file",
                f"Cannot acquire Rinex file due to downloading disabled: {gps_file}")


    sat_type = ephemeris.SatelliteType.GPS
    sat_data = ephemeris.read_rinex_file(gps_file)
    sat_params = ephemeris.rinex_to_satellite_params(sat_data, sat_type)

    satellites = []
    for params in sat_params:
        satellites.append(
            Satellite(params, gps_time, rho_noise, rho_noise_rate))

    return satellites







def get_gps_cov_matrices(config: ConfigHandler,
                         satellites: list[Satellite],
                         estimated_state: EstimatedState) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtains the GPS covariance matrices from the configuration file.
    This is namely required for loosely-coupled GPS fusion.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param satellites: A collection of involved satellite instances.
    :type satellites: list[Satellite]

    :param estimated_state: The initial estimated state.
    :type estimated_state: EstimatedState

    :return: A tuple containing the constructed position and velocity
        covariance matrices, respectively.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    est_pos_lla = estimated_state.position
    sigma_position = get_float("gpsSigmaPosition", __DEFAULT_SIGMA_POSITION)
    sigma_velocity = get_float("gpsSigmaVelocity", __DEFAULT_SIGMA_VELOCITY)

    return calculate_cov_mat(satellites, est_pos_lla,
                             sigma_position, sigma_velocity)


def get_gps_state_matrices(config: ConfigHandler) -> tuple[np.ndarray, np.ndarray]:
    """
    Obtains the GPS state and error matrices from the configuration file.
    This is namely required for tightly-coupled GPS fusion.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A tuple containing the tight state vector and error matrix.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    # Shorthand function for reading float values from configuration
    get_ext_float = partial(config.get_float, "Estimation")
    est_clock_isbm = get_ext_float("clockInitialStaticBiasMean", 0.0)
    est_clock_bdr = get_ext_float("clockBiasDriftRate", 0.0)

    tight_state_vector = np.zeros(17)
    tight_state_vector[15] = est_clock_isbm
    tight_state_vector[16] = est_clock_bdr

    tight_state_errors = np.zeros([17, 17])
    tight_state_errors[15, 15] = 1.0e-09
    tight_state_errors[16, 16] = 1.0e-09

    return tight_state_vector, tight_state_errors


def get_gps_sensor(config: ConfigHandler, satellites: list[Satellite]) -> GpsSensor:
    """
    Initialises GPS sensor based on configuration values and given satellites.
    Reads the parameters from the configuration file and uses them to
    initialise and return a GPS sensor instance, connected to a list of given
    satellites required for measurements.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param satellites: A collection of the satellite instances involved
        in GPS measurements.
    :type satellites: list[Satellite]

    :return: An initialised GPS sensor instance:
    :rtype: GpsSensor
    """

    # Shorthand function for reading boolean values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    frequency = get_float("gpsMeasurementFreq", __DEFAULT_GPS_FREQ)

    start_time: float = 0
    rand_seed = get_error_seed(config)
    gps_zones = get_gps_zones(config)

    return GpsSensor(frequency, satellites, gps_zones, start_time, rand_seed)


def get_gps_zones(config: ConfigHandler) -> FlaggedPolygonGroup | None:

    get_bool = partial(config.get_bool, __SECTION_ID)
    get_str = partial(config.get_str, __SECTION_ID)
    get_csv = partial(config.get_csv_numeric, __SECTION_ID)

    if get_bool('gpsUseZones', False):

        # Extract list of coordinates
        zone_coord_str = get_str('gpsZoneCoordinates', '')
        zone_coord_lines = zone_coord_str.splitlines()
        zone_coords = [np.fromstring(l, sep=',') for l in zone_coord_lines]
        zone_coords = [z for z in zone_coords if len(z) > 0]

        if not all([len(z) % 2 == 0 for z in zone_coords]):
            raise NavConfigError(
                __SECTION_ID, "gpsZoneCoordinates", "Invalid List",
                f"Latitude-longitude list must have 2-pairs of coordinates for each")

        zone_flags = get_csv('gpsZoneActive', [])

        if len(zone_flags) != len(zone_coords):
            raise NavConfigError(
                __SECTION_ID, "gpsZoneActive", "Invalid Length",
                f"Number of value for \'gpsZoneActive\' must "
                f"match number of lines for \'gpsZoneCoordinates\'")

        zone_coords = [np.reshape(z, (2, -1), order='F') for z in zone_coords]
        polygons = []

        for i, (lat, lon) in enumerate(zone_coords):
            is_active = bool(zone_flags[i])
            polygons.append(FlaggedPolygon(lat, lon, is_active, i))

        active_by_default = get_bool('gpsActiveByDefault', True)
        return FlaggedPolygonGroup(polygons, active_by_default)

    return None


def get_gps_fusion(config: ConfigHandler,
                   gps_sensor: GpsSensor,
                   gps_satellites: list[Satellite],
                   estimated_state: EstimatedState | KalmanEstimatedState) -> GpsFusion:
    """
    Initialises the GPS fusion method based on configuration values.
    This initialises and returns the relevant GPS fusion method, based on
    given configuration values. For correct initialisation, this requires
    the sensor, satellites and estimated state instances.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param gps_sensor: The GPS sensor instance used to get measurements.
    :type config: ConfigHandler

    :param gps_satellites: The collection of GPS satellite instances
    :type gps_satellites: list[Satellite]

    :param estimated_state: The initial estimated state.
    :type estimated_state: EstimatedState

    :return: The initialised GPS fusion method.
    :rtype: GpsFusion
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)

    # Obtain the given altimeter settings from the configuration
    method_name: str = get_id("gpsFusionMethod", __DEFAULT_FUSION_METHOD)

    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "fixedgain":
            gain_amount: float = get_float("gpsFixedGain", __DEFAULT_FIXED_GAIN)
            return GpsFixedGainFusion(gps_sensor, estimated_state, gain_amount)

        case "loose":
            pos_cov, vel_cov = get_gps_cov_matrices(config, gps_satellites, estimated_state)
            return GpsLooseFusion(gps_sensor, estimated_state, pos_cov, vel_cov)

        case "tight":
            state_vector, state_errors = get_gps_state_matrices(config)
            return GpsTightFusion(gps_sensor, estimated_state, state_vector, state_errors)

        case _:
            raise NavConfigError(__SECTION_ID, "gpsFusionMethod", "Unknown method",
                                 f"Unrecognized GPS fusion method: {method_name}")


def get_gps_sf(config: ConfigHandler, estimated_state: EstimatedState) -> Optional[GpsFusion]:
    """
    Initialises and returns the complete GPS sensor fusion instance.
    A compact function for initialising and returning the complete GPS sensor
    fusion instance, corresponding to the given configuration values.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param estimated_state: The initial estimated state.
    :type estimated_state: EstimatedState

    :return: The initialised GPS sensor fusion instance.
    :rtype: Optional[GpsFusion]
    """

    # Return None if sensor is not to be used
    if not config.get_bool(
            __SECTION_ID, "gpsEnabled", __DEFAULT_IS_USED):
        return None

    gps_satellites = get_gps_satellites(config)
    gps_sensor = get_gps_sensor(config, gps_satellites)
    return get_gps_fusion(config, gps_sensor, gps_satellites, estimated_state)
