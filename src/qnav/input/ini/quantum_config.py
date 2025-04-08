"""
=================
quantum_config.py
=================

:summary:
    Functions related to initialising objects from quantum config section.
    A collection of functions used for reading content under the 'quantum'
    section within the configuration and initialising the corresponding
    objects. This is used for initialising all quantum sensors and their
    fusion methods.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""

from typing import Optional
from functools import partial

from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini.measurement_config import get_imu_frequency
from qnav.input.ini.rng_config import get_error_seed, get_init_rng, get_init_seed
from qnav.input.ini.vehicle_config import get_sensor_axis
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.platform import SensorAxis
from qnav.quantum.base import ConceptQuantumImu, ConceptQuantumFusion
from qnav.quantum.misalignment import QuantumPFMFusion
from qnav.quantum.particle_filter import QuantumPFFusion, ParticleFilterParams
from qnav.quantum.realistic import ColdAtomInterferometer, QuantumIMU

# The shared section name to use.
__SECTION_ID = "Quantum"

# The default option for if the quantum IMU is enabled.
__DEFAULT_Q_IMU_IS_USED = False

# The default quantum sensor ID.
__DEFAULT_QUANTUM_SENSOR = "concept"

# The default quantum sensor measurement ID.
__DEFAULT_QUANTUM_IMU_FREQ: float = 1

# The default quantum sensor duty cycle ratio.
__DEFAULT_QUANTUM_IMU_DUTY_CYCLE: float = 0.5


def _get_accelerometer_errors(config: ConfigHandler) -> ErrorProperties:
    """
    Gets quantum accelerometer error profile based on configuration values.
    Reads the parameters from the configuration file and uses them to
    return the configured quantum accelerometer error profile.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured quantum accelerometer error profile.
    :rtype: ErrorProperties
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    init_bias_mean = get_float("quantumAccInitialStaticBiasMean", 0)
    bias_drift_error = get_float("quantumAccBiasDriftRate", 0)
    scale_error = get_float("quantumAccScaleErrorMean", 0)
    measurement_error = get_float("quantumAccMeasurementError", 0)
    non_orth_error = get_float("quantumAccNonOrthogonalityMean", 0)

    # Obtain init random number generator:
    rng = get_init_rng(config)

    # Initialise the error profile
    return ErrorProperties(
        init_bias_mean * rng.normal(size=3),
        bias_drift_error,
        scale_error * rng.normal(size=3),
        non_orth_error * rng.normal(size=6),
        measurement_error
    )


def _get_gyroscope_errors(config: ConfigHandler) -> ErrorProperties:
    """
    Gets quantum gyroscope error profile based on configuration values.
    Reads the parameters from the configuration file and uses them to
    return the configured quantum gyroscope error profile.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured quantum gyroscope error profile.
    :rtype: ErrorProperties
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    init_bias_mean = get_float("quantumGyroInitialStaticBiasMean", 0)
    bias_drift_error = get_float("quantumGyroBiasDriftRate", 0)
    scale_error = get_float("quantumGyroScaleErrorMean", 0)
    measurement_error = get_float("quantumGyroMeasurementError", 0)
    non_orth_error = get_float("quantumGyroNonOrthogonalityMean", 0)

    # Obtain init random number generator:
    rng = get_init_rng(config)

    # Initialise the error profile
    return ErrorProperties(
        init_bias_mean * rng.normal(size=3),
        bias_drift_error,
        scale_error * rng.normal(size=3),
        non_orth_error * rng.normal(size=6),
        measurement_error
    )


def _get_interferometer(config: ConfigHandler) -> ColdAtomInterferometer:
    """
    Initialises a quantum interferometer based on configuration values.
    Reads the parameters from the configuration file and uses them to
    return an initialised quantum interferometer, used in realistic
    quantum sensor models.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured quantum cold atom interferometer.
    :rtype: ColdAtomInterferometer
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    num_of_atoms = get_int("quantumImuNumAtoms", 1_000_000)
    atom_mass = get_float("quantumImuAtomMass", 2.20693925e-25)
    sensor_length = get_float("quantumImuBeamWidth", 0.5)
    beam_width = get_float("quantumImuLength", 0.01)
    eta = get_float("quantumImuETA", 0.3)
    time_pulse = get_float("quantumImuTimePulse", 0.16)
    recoil_velocity = get_float("quantumImuAtomRecoilVec", 7.0e-3)

    return ColdAtomInterferometer(
        num_of_atoms=num_of_atoms,
        atom_mass=atom_mass,
        sensor_length=sensor_length,
        beam_width=beam_width,
        eta=eta,
        time_pulse=time_pulse,
        recoil_velocity=recoil_velocity
    )


def _get_quantum_est_axis(config: ConfigHandler) -> Optional[SensorAxis]:
    """
    Obtains the estimated sensor axis to use for fusion.
    Namely used when there is purposeful axis misalignment between estimated
    and actual. If not applicable, None will be returned.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The estimated sensor axis to use for the quantum sensor.
    :rtype: SensorAxis | None
    """

    # TODO: Swap with real sensor axis?

    # Shorthand functions for reading values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_bool = partial(config.get_bool, __SECTION_ID)
    rng = get_init_rng(config)

    if not get_bool('quantumImuHasMisalignment', False):
        return None

    sigma_misalignment = get_float("quantumImuMisalignment", 0.1)
    misalignment = sigma_misalignment * rng.normal(size=3)
    true_axis = get_sensor_axis(config)

    return SensorAxis(
        true_axis.sensor_angles_deg + misalignment,
        true_axis.lever_arm
    )


def _get_particle_filter(config: ConfigHandler) -> ParticleFilterParams:
    """
    Obtains the particle filter parameters to use for fusion.
    Gets the particle filter parameters from the configuration that can be
    used by some fusion methods.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The particle filter parameters to use for fusion.
    :rtype: ParticleFilterParams
    """

    # Shorthand functions for reading values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    num_particles = get_int("quantumImuNumParticles", 200)
    sigma_misalignment = get_float("quantumImuSigmaMisalignment", 0.1)
    sigma_accelerometer = get_float("quantumImuSigmaAccelerometer", 0.1)
    sigma_gyroscope = get_float("quantumImuSigmaGyroscope", 0.1)
    seed = get_init_seed(config)

    # TODO: Add support for other parameters?
    return ParticleFilterParams(
        num_particles=num_particles,
        sigma_misalignment=sigma_misalignment,
        sigma_accelerometer=sigma_accelerometer,
        sigma_gyroscope=sigma_gyroscope,
        rng_seed=seed
    )


def get_quantum_imu_sensor(config: ConfigHandler) -> ConceptQuantumImu:
    """
    Initialises the quantum IMU sensor based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return a quantum IMU sensor instance. This is a quantum
    sensor that combines both a quantum accelerometer and gyroscope.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: An initialised quantum IMU sensor instance.
    :rtype: ConceptQuantumImu
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    method_name: str = get_id("quantumImuSensor", __DEFAULT_QUANTUM_SENSOR)

    sub_frequency = get_imu_frequency(config)
    full_frequency = get_float("quantumImuFrequency", __DEFAULT_QUANTUM_IMU_FREQ)
    duty_cycle = get_float("quantumImuDutyCycle", __DEFAULT_QUANTUM_IMU_DUTY_CYCLE)

    accelerometer_errors = _get_accelerometer_errors(config)
    gyroscope_errors = _get_gyroscope_errors(config)
    sensor_axis = get_sensor_axis(config)
    rand_seed = get_error_seed(config)
    start_time = 0

    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "concept":
            return ConceptQuantumImu(full_frequency, sub_frequency,
                                     accelerometer_errors, gyroscope_errors,
                                     sensor_axis, duty_cycle, start_time, rand_seed)

        case "realistic":
            interferometer = _get_interferometer(config)
            return QuantumIMU(full_frequency, sub_frequency,
                              accelerometer_errors, gyroscope_errors, interferometer,
                              sensor_axis, duty_cycle, start_time, rand_seed)

        case _:
            raise NavConfigError(__SECTION_ID, "quantumImuSensor", "Unknown type",
                                 f"Unrecognized quantum IMU sensor type: {method_name}")


def get_quantum_imu_fusion(config: ConfigHandler, sensor: ConceptQuantumImu,
                           accelerometer: Accelerometer, gyroscope: Gyroscope) -> ConceptQuantumFusion:
    """
    Initialises the quantum IMU fusion method based on configuration values.
    Reads the parameters from the configuration file and uses them to
    initialise and return a quantum IMU fusion instance.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param sensor: The quantum IMU instance to use for fusion.
    :type sensor: ConceptQuantumImu

    :param accelerometer: The shared INS accelerometer instance.
    :type accelerometer: Accelerometer

    :param gyroscope: The shared INS gyroscope instance.
    :type gyroscope: Gyroscope

    :return: An initialised quantum IMU fusion instance.
    :rtype: ConceptQuantumFusion
    """

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    method_name: str = get_id("quantumImuFusion", __DEFAULT_QUANTUM_SENSOR)

    # TODO: Swap real axis but provide estimate?
    # TODO: CHECK CORRECT AXIS
    est_axis = _get_quantum_est_axis(config)

    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "basic":
            return ConceptQuantumFusion(sensor, accelerometer, gyroscope, est_axis)

        case "particlefilter":
            filter_params = _get_particle_filter(config)
            return QuantumPFFusion(
                sensor, accelerometer, gyroscope, est_axis,
                filter_params=filter_params)

        case "misaligned":
            filter_params = _get_particle_filter(config)
            return QuantumPFMFusion(
                sensor, accelerometer, gyroscope, est_axis,
                filter_params=filter_params)

        case _:
            raise NavConfigError(
                __SECTION_ID, "quantumImuFusion", "Unknown type",
                f"Unrecognized quantum IMU fusion type: {method_name}")


def get_quantum_imu_sf(config: ConfigHandler, accelerometer: Accelerometer,
                       gyroscope: Gyroscope) -> Optional[ConceptQuantumFusion]:
    """
    Initialises and returns the complete quantum IMU sensor fusion instance.
    A compact function for initialising and returning the complete quantum
    IMU sensor-fusion instance, corresponding to the given configuration
    values.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param accelerometer: The shared INS accelerometer instance.
    :type accelerometer: Accelerometer

    :param gyroscope: The shared INS gyroscope instance.
    :type gyroscope: Gyroscope

    :return: An initialised quantum IMU fusion instance.
    :rtype: Optional[ConceptQuantumFusion]
    """

    # Return None if sensor is not to be used
    if not config.get_bool(
            __SECTION_ID, "quantumImuEnabled", __DEFAULT_Q_IMU_IS_USED):
        return None

    sensor = get_quantum_imu_sensor(config)
    return get_quantum_imu_fusion(config, sensor, accelerometer, gyroscope)



# ------------------------------------------------------------------------------------------------




