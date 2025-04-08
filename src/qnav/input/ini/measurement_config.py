"""
=====================
measurement_config.py
=====================

:summary:
    Functions related to initialising objects from measurement config section.
    A collection of functions used for reading content under the 'measurement'
    section within the configuration and initialising the corresponding
    objects. This mostly includes sensors used within simulations.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""

from qnav.input.config_handler import ConfigHandler
from qnav.input.ini.vehicle_config import get_sensor_axis
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.clock import Clock
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.guassian_markov import AccelerometerGM
from qnav.measurement.guassian_markov import GyroscopeGM
from qnav.measurement.gyroscope import Gyroscope
from qnav.input.ini.rng_config import get_error_seed
from qnav.input.ini.rng_config import get_init_rng
from functools import partial

# The shared section name to use
__SECTION_ID: str = "Measurement"


def get_clock(config: ConfigHandler) -> Clock:
    """
    Initialises clock based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return a clock instance.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: An initialised clock instance.
    :rtype: Clock
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    # Obtain the clock profile
    init_bias = get_float("clockInitialBias", 0)
    bias_drift_rate = get_float("clockBiasDriftRate", 0)

    # Obtain other parameters
    start_time = 0
    rand_seed = get_error_seed(config)

    # Construct and return the clock instance
    return Clock(init_bias, bias_drift_rate, start_time, rand_seed)


def get_imu_frequency(config: ConfigHandler) -> float:
    """
    Helper function to get IMU frequency from configuration values.
    This is the frequency (in Hz) to be used by the primary accelerometer
    and gyroscope sensors.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The IMU frequency (in Hz).
    :rtype: float
    """
    return config.get_float(__SECTION_ID, "imuMeasurementFreq")


def get_accelerometer(config: ConfigHandler) -> Accelerometer:
    """
    Initialises accelerometer based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return an accelerometer sensor instance.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: An initialised accelerometer sensor instance.
    :rtype: Accelerometer
    """

    # Shorthand function for reading float values from configuration
    get_bool = partial(config.get_bool, __SECTION_ID)
    get_float = partial(config.get_float, __SECTION_ID)

    # Obtain the accelerometer profile
    freq = get_imu_frequency(config)
    init_bias_mean = get_float("accelerometerInitialStaticBiasMean", 0)
    bias_drift_error = get_float("accelerometerBiasDriftRate", 0)
    scale_error = get_float("accelerometerScaleErrorMean", 0)
    measurement_error = get_float("accelerometerMeasurementError", 0)
    non_orth_error = get_float("accelerometerNonOrthogonalityMean", 0)

    # Obtain other parameters
    start_time = 0  # TODO: Set correctly
    sensor_axis = get_sensor_axis(config)
    rand_seed = get_error_seed(config)

    # Obtain init random number generator:
    rng = get_init_rng(config)

    # Initialise the error profile
    error_profile = ErrorProperties(
        init_bias_mean * rng.normal(size=3),
        bias_drift_error,
        scale_error * rng.normal(size=3),
        non_orth_error * rng.normal(size=6),
        measurement_error
    )

    # Switch to Gaussian Markov noise model if enabled:
    if get_bool('useGaussianMarkovNoise', False):

        # Construct and return the accelerometer instance
        corr_time = get_float("accelerometerTimeCorrelation", 10.0)  # TODO: Set suitable default
        psd_noise = get_float("accelerometerPowSpecDensity", 0.01)   # TODO: Set suitable default
        return AccelerometerGM(freq, error_profile, corr_time, psd_noise,
                               sensor_axis, start_time, rand_seed)

    else:

        # Construct and return the accelerometer instance
        return Accelerometer(freq, error_profile, sensor_axis,
                             start_time, rand_seed)


def get_gyroscope(config: ConfigHandler) -> Gyroscope:
    """
    Initialises gyroscope based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return a gyroscope sensor instance.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: An initialised gyroscope sensor instance.
    :rtype: Gyroscope
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_bool = partial(config.get_bool, __SECTION_ID)

    # Obtain the gyroscope profile
    freq = get_imu_frequency(config)
    init_bias_mean = get_float("gyroscopeInitialStaticBiasMean", 0)
    bias_drift_error = get_float("gyroscopeBiasDriftRate", 0)
    scale_error = get_float("gyroscopeScaleErrorMean", 0)
    measurement_error = get_float("gyroscopeMeasurementError", 0)
    non_orth_error = get_float("gyroscopeNonOrthogonalityMean", 0)

    # Obtain other parameters
    start_time = 0
    sensor_axis = get_sensor_axis(config)
    rand_seed = get_error_seed(config)

    # Obtain init random number generator:
    rng = get_init_rng(config)

    # Initialise the error profile
    error_profile = ErrorProperties(
        init_bias_mean * rng.normal(size=3),
        bias_drift_error,
        scale_error * rng.normal(size=3),
        non_orth_error * rng.normal(size=6),
        measurement_error
    )

    # Switch to Gaussian Markov noise model if enabled:
    if get_bool('useGaussianMarkovNoise', False):

        # Construct and return the accelerometer instance
        corr_time = get_float("gyroscopeTimeCorrelation", 10.0)  # TODO: Set suitable default
        psd_noise = get_float("gyroscopePowSpecDensity", 0.01)   # TODO: Set suitable default
        return GyroscopeGM(freq, error_profile, corr_time, psd_noise,
                               sensor_axis, start_time, rand_seed)

    else:

        # Construct and return the gyroscope instance
        return Gyroscope(freq, error_profile, sensor_axis,
                             start_time, rand_seed)
