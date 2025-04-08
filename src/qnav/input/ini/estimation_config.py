"""
===================
altimeter_config.py
===================

:summary:
    Functions related to initialising the estimation components.
    A collection of functions used for reading content under the 'estimation'
    section within the configuration and initialising the corresponding
    objects. This includes the INS estimation fusion method and any initial
    estimation errors.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from qnav.estimation.kalman import generate_matrices
from qnav.gravity.base import GravityModel
from qnav.input.ini.measurement_config import get_imu_frequency
from qnav.util.transformations import ned2lla
from qnav.estimation.state import EstimatedState, KalmanEstimatedState
from qnav.fusion.ins import NumericalINS, RungeKutta, AdamsBashforth
from qnav.fusion.kalman import KalmanINS
from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini.vehicle_config import get_sensor_axis
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.gyroscope import Gyroscope

from functools import partial

from qnav.waypoints.trajectory import GroundTruth

# The shared section name to use
__SECTION_ID: str = "Estimation"

# The default estimated state model
__DEFAULT_ES_MODEL: str = "simple"

# The default INS fusion method ID
__DEFAULT_INS_METHOD: str = "numerical"


def get_estimated_state(config: ConfigHandler, ground_truth: GroundTruth,
                        gravity_model: GravityModel) -> EstimatedState:

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    method_name: str = get_id("estimatedStateModel", __DEFAULT_ES_MODEL)

    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "simple":
            return EstimatedState(ground_truth, gravity_model)

        case "kalman":
            return _get_kalman_state(config, ground_truth, gravity_model)

        case _:
            raise NavConfigError(
                __SECTION_ID, "estimatedStateModel", "Unknown type",
                f"Unrecognized estimated state model: {method_name}")


def use_virtual_memory(config: ConfigHandler) -> bool:
    get_bool = partial(config.get_bool, __SECTION_ID)
    return get_bool("useVirtualMemory", False)


def _get_kalman_state(config: ConfigHandler, ground_truth: GroundTruth,
                      gravity_model: GravityModel) -> KalmanEstimatedState:

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    imu_frequency = get_imu_frequency(config)
    imu_rate = 1 / imu_frequency

    acc_meas_error = get_float('accelerometerMeasurementError', 1e-3)
    vel_meas_error = get_float( 'gyroscopeMeasurementError', 1e-3)
    acc_proc_noise = get_float('processNoiseAcceleration', 1e-3)
    vel_proc_noise = get_float( 'processNoiseVelocity', 1e-6)
    att_proc_noise = get_float('processNoiseAngular', 1e-5)
    ang_proc_noise = get_float('processNoiseAngleRate', 1e-5)

    h_matrix, r_matrix, f_matrix, q_matrix = generate_matrices(
        imu_rate, acc_meas_error, vel_meas_error, acc_proc_noise,
        vel_proc_noise, ang_proc_noise, att_proc_noise)

    state_errors = np.eye(15) * 1e-9

    return KalmanEstimatedState(
        ground_truth, h_matrix, r_matrix, f_matrix,
        q_matrix, state_errors, gravity_model)


def add_estimation_errors(config: ConfigHandler, estimated_state: EstimatedState):
    """
    Adds configured estimation errors to the estimated state.
    This applies any initial estimation errors to a given estimated state,
    causing selected misalignment between the ground truth and estimation
    and initialisation. By default, the estimated state perfectly matches
    the first record of the ground truth.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param estimated_state: The estimated state at beginning of simulation.
    :type estimated_state: EstimatedState
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    position_error = np.array([
        get_float('initialPositionErrorNorth', 0),
        get_float('initialPositionErrorEast', 0),
        get_float('initialPositionErrorDown', 0)
    ])

    velocity_error = np.array([
        get_float('initialVelocityErrorNorth', 0),
        get_float('initialVelocityErrorEast', 0),
        get_float('initialVelocityErrorDown', 0)
    ])

    acceleration_error = np.array([
        get_float('initialAccelerationErrorNorth', 0),
        get_float('initialAccelerationErrorEast', 0),
        get_float('initialAccelerationErrorDown', 0)
    ])

    attitude_error = np.array([
        get_float('initialAttitudeErrorHeading', 0),
        get_float('initialAttitudeErrorPitch', 0),
        get_float('initialAttitudeErrorRoll', 0)
    ])

    angle_rate_error = np.array([
        get_float('initialAngleRateErrorHeading', 0),
        get_float('initialAngleRateErrorPitch', 0),
        get_float('initialAngleRateErrorRoll', 0)
    ])

    # Obtain the current estimates
    position = estimated_state.position
    velocity = estimated_state.velocity
    acceleration = estimated_state.acceleration
    attitude = estimated_state.attitude
    angle_rates = estimated_state.angle_rates

    # Add errors to get updated estimates
    updated_position = ned2lla(position_error, position)
    updated_velocity = velocity + velocity_error
    updated_acceleration = acceleration + acceleration_error
    updated_attitude = attitude + attitude_error
    updated_angle_rates = angle_rates + angle_rate_error

    # Replace estimates with update values
    estimated_state.update_estimates(
        position=updated_position,
        velocity=updated_velocity,
        acceleration=updated_acceleration,
        attitude=updated_attitude,
        angle_rates=updated_angle_rates
    )

def has_estimation_errors(config: ConfigHandler) -> bool:
    """
    A helper function to test if initial estimation errors are set.
    Simply used for debugging and displaying if values are used.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: True if there are initial estimation errors, False otherwise.
    :rtype: bool
    """

    # Shorthand function for reading float values from configuration
    get_float = partial(config.get_float, __SECTION_ID)

    error_values = np.array([
        get_float('initialPositionErrorNorth', 0),
        get_float('initialPositionErrorEast', 0),
        get_float('initialPositionErrorDown', 0),
        get_float('initialVelocityErrorNorth', 0),
        get_float('initialVelocityErrorEast', 0),
        get_float('initialVelocityErrorDown', 0),
        get_float('initialAccelerationErrorNorth', 0),
        get_float('initialAccelerationErrorEast', 0),
        get_float('initialAccelerationErrorDown', 0),
        get_float('initialAttitudeErrorHeading', 0),
        get_float('initialAttitudeErrorPitch', 0),
        get_float('initialAttitudeErrorRoll', 0),
        get_float('initialAngleRateErrorHeading', 0),
        get_float('initialAngleRateErrorPitch', 0),
        get_float('initialAngleRateErrorRoll', 0)
    ])

    return np.any(error_values != 0)

def get_ins_str(config: ConfigHandler) -> str:
    """
    Returns the str used to indentify the INS method to use.
    Expected values are 'numerical', 'kalman', 'runge_

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The ID for the INS method to use.
    :rtype: str
    """

    get_id = partial(config.get_str, __SECTION_ID)
    return get_id("integrationMethod", __DEFAULT_INS_METHOD)

def get_ins(config: ConfigHandler,
            accelerometer: Accelerometer,
            gyroscope: Gyroscope) -> NumericalINS:
    """
    Returns the initialised INS fusion method corresponding the configuration.
    Using the values presented in the configuration file, this function uses
    them, along with the primary accelerometer and gyroscope instances, to
    initialise the fusion method.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param accelerometer: The accelerometer instance to attach to the INS.
    :type accelerometer: Accelerometer

    :param gyroscope: The gyroscope instance to attach to the INS.
    :type gyroscope: Gyroscope

    :return: The initialised INS fusion method to use within the simulation.
    :rtype: NumericalINS
    """

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    method_name: str = get_id("integrationMethod", __DEFAULT_INS_METHOD)

    est_acc_axis = get_sensor_axis(config)
    est_gyro_axis = get_sensor_axis(config)

    # Return the requested fusion method:
    match (method_name.strip().casefold()):

        case "numerical":
            return NumericalINS(accelerometer, gyroscope, est_acc_axis, est_gyro_axis)

        case "rungekutta":
            return RungeKutta(accelerometer, gyroscope, est_acc_axis, est_gyro_axis)

        case "adamsbashforth":
            return AdamsBashforth(accelerometer, gyroscope, est_acc_axis, est_gyro_axis)

        case "kalman":
            return KalmanINS(accelerometer, gyroscope, est_acc_axis, est_gyro_axis)

        case _:
            raise NavConfigError(__SECTION_ID, "integrationMethod", "Unknown type",
                                 f"Unrecognized IMU integration method: {method_name}")

