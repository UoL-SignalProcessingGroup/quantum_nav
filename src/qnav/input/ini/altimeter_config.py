"""
===================
altimeter_config.py
===================

:summary:
    Functions related to initialising the altimeter and its fusion method.
    A collection of functions used for reading content under the 'altimeter'
    section within the configuration and initialising the corresponding
    objects.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""


from qnav.fusion.altimeter import FixedGainAltimeter
from qnav.fusion.altimeter import AlphaBetaAltimeter
from qnav.measurement.altimeter import Altimeter
from qnav.measurement.sensor import SensorFusion
from qnav.input.config_handler import ConfigHandler
from qnav.input.config_handler import NavConfigError
from qnav.input.ini.rng_config import get_error_seed
from qnav.input.ini.rng_config import get_init_rng
from functools import partial
from typing import Optional


# The shared section name to use
__SECTION_ID: str = "Altimeter"

# The default value for if the altimeter is used
__DEFAULT_IS_USED: bool = False

# The default fusion method to use for the altimeter
__DEFAULT_FUSION_METHOD: str = "fixed_gain"

# The default fixed gain amount to use for fixed-gain filtering
__DEFAULT_FIXED_GAIN: float = 0.1

# The default alpha value to use for alpha-beta filtering
__DEFAULT_ALPHA: float = 0.9

# The default beta value to use for alpha-beta filtering
__DEFAULT_BETA: float = 0.1


def get_sensor(config: ConfigHandler) -> Altimeter:
    """
    Initialises altimeter sensor based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return an altimeter sensor instance.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: An initialised altimeter sensor instance.
    :rtype: Altimeter
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    # Obtain the altimeter profile
    freq = get_float("altimeterMeasurementFreq")
    init_bias_mean = get_float("altimeterInitialStaticBiasMean", 0)
    bias_drift_error = get_float("altimeterBiasDriftRate", 0)

    # Obtain other parameters
    start_time: float = 0
    rand_seed: int = get_error_seed(config)

    # Randomise error properties
    rng = get_init_rng(config)
    bias_error = init_bias_mean * rng.normal()

    # Construct and return the altimeter instance
    return Altimeter(freq, bias_error, bias_drift_error,
                     start_time, rand_seed)


def get_fusion(config: ConfigHandler, altimeter: Altimeter) -> SensorFusion:
    """
    Initialises altimeter fusion method based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return an altimeter fusion instance with given
    altimeter sensor.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param altimeter: The altimeter sensor instance.
    :type altimeter: Altimeter

    :return: An initialised altimeter sensor instance.
    :rtype: Altimeter
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_bool = partial(config.get_bool, __SECTION_ID)

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)

    # Obtain the given altimeter settings from the configuration
    method_name: str = get_id("altimeterFusion", __DEFAULT_FUSION_METHOD)
    gain_amount: float = get_float("altimeterGainAmount", __DEFAULT_FIXED_GAIN)
    alpha: float = get_float("altimeterAlphaAmount", __DEFAULT_ALPHA)
    beta: float = get_float("altimeterBetaAmount", __DEFAULT_BETA)

    # Check that the beta value is not to be auto selected
    beta = None if get_bool("altimeterAutoSetBeta", False) else beta

    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "fixedgain":
            return FixedGainAltimeter(altimeter, gain_amount)

        case "alphabeta":
            return AlphaBetaAltimeter(altimeter, alpha, beta)

        case _:
            raise NavConfigError(__SECTION_ID, "altimeterFusion", "Unknown method",
                                 f"Unrecognized altimeter fusion method: {method_name}")


def get_altimeter_sf(config: ConfigHandler) -> Optional[SensorFusion]:
    """
    Initialises altimeter sensor fusion based on configuration values.
    Firstly, checks that an altimeter has been request in the configuration.
    If so, a fusion method include for the requested sensor and fusion type
    are initialised and return. Otherwise, None is returned.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: An initialised altimeter sensor fusion instance or None.
    :rtype: Optional[SensorFusion]
    """

    # Return None if sensor is not to be used
    if not config.get_bool(
            __SECTION_ID, "altimeterEnabled", __DEFAULT_IS_USED):
        return None

    altimeter = get_sensor(config)
    return get_fusion(config, altimeter)
