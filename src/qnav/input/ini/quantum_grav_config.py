from functools import partial
from typing import Optional

from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini.gravity_config import get_true_gravity_model
from qnav.input.ini.measurement_config import get_imu_frequency
from qnav.input.ini.rng_config import get_init_rng, get_error_seed
from qnav.input.ini.vehicle_config import get_sensor_axis
from qnav.measurement.error_properties import ErrorProperties
from qnav.quantum.gravity_gradient import GravityGradiometer, ParticleFilterParams
from qnav.quantum.gravity_gradient import GravityGradInterferometer
from qnav.quantum.gravity_gradient import GravityGradientPF

# The shared section name to use.
# __SECTION_ID = "MapMatching"
__SECTION_ID = "Quantum"

# The default option for if quantum gravity map-matching is enabled.
__DEFAULT_Q_GMM_IS_USED = False

# The default gravity gradient sensor to use
__DEFAULT_GRAVITY_SENSOR = "dual_interferometer"

# The default fusion method to use
__DEFAULT_GRAVITY_FUSION = "particle_filter"

# The default quantum sensor measurement ID.
__DEFAULT_QUANTUM_IMU_FREQ: float = 1.0

# The default quantum sensor duty cycle ratio.
__DEFAULT_QUANTUM_IMU_DUTY_CYCLE: float = 0.5


def _get_gravity_interferometer(config: ConfigHandler) -> GravityGradInterferometer:
    """
    Initialises a quantum interferometer based on configuration values.
    Reads the parameters from the configuration file and uses them to
    return an initialised quantum interferometer, used in realistic
    quantum sensor models.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured quantum cold atom interferometer.
    :rtype: GravityGradInterferometer
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    num_of_atoms = get_int("quantumGravNumAtoms", 200000)
    atom_mass = get_float("quantumGravAtomMass", 1.4431608262e-25)
    recoil_velocity = get_float("quantumGravRecoilVelocity", 5.8845e-3)
    stationary_velocity = get_float("quantumGravStationaryVelocity", 1.57)
    sensor_z_top = get_float("quantumGravSensorZTop", 0.0)
    sensor_z_bottom = get_float("quantumGravSensorZBottom", -1.0)
    eta = get_float("quantumGravEta", 0.25)
    time_pulse = get_float("quantumGravTimePulse", 0.2)
    sigma_phi = get_float("quantumGravSigmaPhi", 25e-3)
    sigma_n = get_float("quantumGravSigmaN", 1.0)
    beam_width = get_float("quantumGravBeamWidth", 0.01)
    sensor_length = get_float("quantumGravSensorLength", 1.5)
    rand_failure_prob = get_float("quantumGravRandFailureProb", 0.0)

    return GravityGradInterferometer(
        num_of_atoms=num_of_atoms,
        atom_mass=atom_mass,
        recoil_velocity=recoil_velocity,
        stationary_velocity=stationary_velocity,
        sensor_z_top=sensor_z_top,
        sensor_z_bottom=sensor_z_bottom,
        eta=eta,
        time_pulse=time_pulse,
        sigma_phi=sigma_phi,
        sigma_n=sigma_n,
        beam_width=beam_width,
        sensor_length=sensor_length,
        rand_failure_prob=rand_failure_prob
    )


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

    init_bias_mean = get_float("quantumGravInitialStaticBiasMean", 0)
    bias_drift_error = get_float("quantumGravBiasDriftRate", 0)
    scale_error = get_float("quantumGravScaleErrorMean", 0)
    measurement_error = get_float("quantumGravMeasurementError", 0)
    non_orth_error = get_float("quantumGravNonOrthogonalityMean", 0)

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

def get_quantum_grav_sensor(config: ConfigHandler) -> GravityGradiometer:

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    method_name: str = get_id("quantumGravSensor", __DEFAULT_GRAVITY_SENSOR)

    sub_frequency = get_imu_frequency(config)
    full_frequency = get_float("quantumGravFrequency", __DEFAULT_QUANTUM_IMU_FREQ)
    duty_cycle = get_float("quantumGravDutyCycle", __DEFAULT_QUANTUM_IMU_DUTY_CYCLE)

    interferometer = _get_gravity_interferometer(config)
    gravity_model = get_true_gravity_model(config)

    accelerometer_errors = _get_accelerometer_errors(config)
    sensor_axis = get_sensor_axis(config)
    rand_seed = get_error_seed(config)
    start_time = 0

    match (method_name.strip().casefold()):

        case 'dualinterferometer':
            return GravityGradiometer(
                full_frequency, sub_frequency,
                accelerometer_errors,
                interferometer,
                gravity_model,
                sensor_axis,
                duty_cycle,
                start_time,
                rand_seed)

        case _:
            raise NavConfigError(__SECTION_ID, "quantumGravSensor", "Unknown type",
                                 f"Unrecognized quantum gravity sensor type: {method_name}")


def _get_particle_filter_params(config: ConfigHandler) -> ParticleFilterParams:

    get_float = partial(config.get_float, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    num_particles = get_int("quantumGravFusionNumParticles", 1000)

    init_position_sigma = get_float("quantumGravFusionInitPositionSigma", 20.0)
    init_velocity_sigma = get_float("quantumGravFusionInitVelocitySigma", 1e-5)
    init_attitude_sigma = get_float("quantumGravFusionInitAttitudeSigma", 1e-5)

    weight_sigma = get_float("quantumGravFusionWeightSigma", 0.00126)
    alpha = get_float("quantumGravFusionAlphaSigma", 0.05)
    beta = get_float("quantumGravFusionBetaSigma", 0.05)
    gamma = get_float("quantumGravFusionGammaSigma", 0.05)

    position_sigma = get_float("quantumGravFusionPositionSigma", 2.0)
    velocity_sigma = get_float("quantumGravFusionVelocitySigma", 0.005)
    attitude_sigma = get_float("quantumGravFusionAttitudeSigma", 0.001)

    # TODO: NEW
    ellipse_intervals = get_int('quantumGravFusionEllipseIntervals', 361)
    ellipse_iterations = get_int('quantumGravFusionEllipseIterations', 5)


    return ParticleFilterParams(
        num_particles=num_particles,
        init_position_sigma=init_position_sigma,
        init_velocity_sigma=init_velocity_sigma,
        init_attitude_sigma=init_attitude_sigma,
        weight_sigma=weight_sigma,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        position_sigma=position_sigma,
        velocity_sigma=velocity_sigma,
        attitude_sigma=attitude_sigma,
        ellipse_intervals=ellipse_intervals,
        ellipse_iterations=ellipse_iterations
    )



def get_quantum_grav_fusion(config: ConfigHandler, sensor: GravityGradiometer) -> GravityGradientPF:

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    method_name: str = get_id("quantumGravFusion", __DEFAULT_GRAVITY_FUSION)
    rand_seed = get_error_seed(config)



    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "particlefilter":
            filter_params = _get_particle_filter_params(config)
            return GravityGradientPF(
                sensor,
                filter_params,
                est_interferometer=_get_gravity_interferometer(config),
                rng_seed=rand_seed,
            )

        case _:
            raise NavConfigError(
                __SECTION_ID, "quantumGravFusion", "Unknown type",
                f"Unrecognized quantum gravity fusion type: {method_name}")


def get_quantum_grav_sf(config: ConfigHandler) -> Optional[GravityGradientPF]:

    # Return None if sensor is not to be used
    if not config.get_bool(
            __SECTION_ID, "quantumGravityEnabled", __DEFAULT_Q_GMM_IS_USED):
        return None

    sensor = get_quantum_grav_sensor(config)
    return get_quantum_grav_fusion(config, sensor)
