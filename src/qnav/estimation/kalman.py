"""
=========
kalman.py
=========

:summary:
    Implementation of a Kalman filter based Inertial Navigation System (INS).
    This module contains a Kalman filter INS solution and corresponding
    Kalman based functions/equations. This solution inherits from the
    standard INS model.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import math
import numpy as np
import qnav.util.transformations as trans

from math import cos, sin, tan
from qnav.util.constants import OMEGA_E
from qnav.gravity.base import GravityModel
from qnav.gravity.simple import SimpleUniform
from qnav.util.transformations import cross_prod_xy


def state_vector2ned(state_vector: np.ndarray, ref_lla: np.ndarray,
                     gravity_model: GravityModel = SimpleUniform) -> np.ndarray:
    """
    This function converts the state vector array and measurements into local
    North-East-Down (NED) coordinates from a given reference location,
    suitable for kalman filtering.

    :param state_vector: The state vector array holding the position,
        velocity, acceleration, attitude and angle rate variables.
    :type state_vector: numpy.ndarray (15-elements)

    :param ref_lla: The reference latitude-longitude-altitude vector (given in
        degrees-degrees-metres) identifying the centre point.
    :type ref_lla: numpy.ndarray (3-elements)

    :param gravity_model: The gravity model to use for estimating acceleration.
    :type gravity_model: GravityModel

    :return: The given state vector array and measurements converted to
        local NED/Earth axes for Kalman filtering.
    :rtype: numpy.ndarray (15-elements)
    """

    # Calculate the estimated gravity vector
    g = gravity_model.calc_gravity_xyz(*ref_lla)

    # Define Angular velocity for Earth's rotation (in local NED axes)
    lat_rad = math.radians(float(ref_lla[0]))
    omega_e = OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

    # Convert Positions from LLA to Local NED
    # Fix the reference Lat-Long-Altitude location for conversation from/to LLA and NED coordinates
    position = trans.lla2ned(state_vector[[0, 3, 6]], ref_lla)

    # Convert Velocities and Accelerations from Body to Local NED
    # and convert Attitudes from degrees to radians
    attitude = np.radians(state_vector[[9, 11, 13]])

    # Calculate ESTIMATED body axes from Earth-oriented axes
    rot_earth2body = trans.rotate_3d(*attitude)

    # Convert velocity from Body axes to Earth Axes
    velocity_b = state_vector[[1, 4, 7]]
    acceleration_b = state_vector[[2, 5, 8]]
    angle_rate_b = np.radians(state_vector[[14, 12, 10]])

    # Convert velocity from body axes to Earth Axes
    velocity_e = np.linalg.solve(rot_earth2body, velocity_b)

    # Convert measured acceleration from body axes to Earth axes and remove coriolis and gravity terms
    acceleration_e = np.linalg.solve(rot_earth2body, acceleration_b) + g

    # Remove effect of Coriolis term due to Earth's rotation and transport rate (in Local NED axes)
    acceleration_e -= 2.0 * cross_prod_xy(omega_e, velocity_e)
    # acceleration_e -= 2.0 * np.cross(omega_e, velocity_e)  # Can change?

    # Convert PQR Angle Rates from Body to Local NED/Earth Axes
    rot_pqr2euler = np.array([
        [1.0, sin(attitude[2]) * tan(attitude[1]), cos(attitude[2]) * tan(attitude[1])],
        [0.0, cos(attitude[2]), -sin(attitude[2])],
        [0.0, sin(attitude[2]) * (1 / cos(attitude[1])), cos(attitude[2]) * (1 / cos(attitude[1]))]
    ])
    euler_rates = rot_pqr2euler @ angle_rate_b

    # Check the attitude angles (periodicity)
    pi = np.pi
    attitude = np.remainder(attitude + pi, 2 * pi) - pi

    # Return the converted state vector
    return np.array([
        position[0], velocity_e[0], acceleration_e[0],
        position[1], velocity_e[1], acceleration_e[1],
        position[2], velocity_e[2], acceleration_e[2],
        attitude[0], euler_rates[2], attitude[1],
        euler_rates[1], attitude[2], euler_rates[0]
    ]).astype(np.float64)


def ned2state_vector(state_vector: np.ndarray, ref_lla: np.ndarray,
                     gravity_model: GravityModel = SimpleUniform) -> np.ndarray:
    """
    This function converts the state vector array back to Latitude-longitude-
    altitude (LLA) coordinates (in degrees-degrees-metres). Please note that
    the state error matrix remains in Local North-East-Down (NED) coordinates.

    :param state_vector: The state vector array holding the position,
        velocity, acceleration, attitude and angle rate variables.
    :type state_vector: numpy.ndarray (15-elements)

    :param ref_lla: The reference latitude-longitude-altitude vector (given in
        degrees-degrees-metres) identifying the centre point.
    :type ref_lla: numpy.ndarray (3-elements)

    :param gravity_model: The gravity model to use for estimating acceleration.
    :type gravity_model: GravityModel

    :return: The given state vector array converted to LLA.
    :rtype: numpy.ndarray (15-elements)
    """

    # Calculate the gravity vector for the current position
    g = gravity_model.calc_gravity_xyz(*ref_lla)

    # Define Angular velocity for Earth's rotation (in local NED axes)
    lat_rad = math.radians(float(ref_lla[0]))
    omega_e = OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

    # Convert Angle Rates from Local NED/Earth Axes to Body PQR Axes
    rot_euler2pqr = np.array([
        [1.0, 0.0, -sin(state_vector[11])],
        [0.0, cos(state_vector[13]), sin(state_vector[13]) * cos(state_vector[11])],
        [0.0, -sin(state_vector[13]), cos(state_vector[13]) * cos(state_vector[11])],
    ])
    angle_rate_b = rot_euler2pqr @ state_vector[[14, 12, 10]]

    # Calculate ESTIMATED body axes from Earth-oriented axes
    rot_earth2body = trans.rotate_3d(*state_vector[[9, 11, 13]])
    velocity_e = state_vector[[1, 4, 7]]
    acceleration_e = state_vector[[2, 5, 8]]

    acceleration_e += 2.0 * cross_prod_xy(omega_e, velocity_e)
    # acceleration_e += 2.0 * np.cross(omega_e, velocity_e)

    # Convert Velocities from Earth Axes to Body Axes
    velocity_b = rot_earth2body @ velocity_e

    # Add Coriolis and gravity terms and convert acceleration from Earth axes to Body Axes
    acceleration_b = rot_earth2body @ (acceleration_e - g)

    # Convert Positions from Local NED to LLA
    position = trans.ned2lla(state_vector[[0, 3, 6]], ref_lla)

    # Convert back to degrees
    angle_rate_b = np.degrees(angle_rate_b)

    # Update the state vector
    state_vector[0] = position[0]
    state_vector[1] = velocity_b[0]
    state_vector[2] = acceleration_b[0]
    state_vector[3] = position[1]
    state_vector[4] = velocity_b[1]
    state_vector[5] = acceleration_b[1]
    state_vector[6] = position[2]
    state_vector[7] = velocity_b[2]
    state_vector[8] = acceleration_b[2]
    state_vector[9] = math.degrees(state_vector[9])
    state_vector[10] = angle_rate_b[2]
    state_vector[11] = math.degrees(state_vector[11])
    state_vector[12] = angle_rate_b[1]
    state_vector[13] = math.degrees(state_vector[13])
    state_vector[14] = angle_rate_b[0]

    # Finally, return the state vector
    return state_vector


def generate_matrices(dt: float, acc_meas_error: float, gyro_meas_error: float,
                      acc_process_noise: float = 1e-3, vel_process_noise: float = 1e-6,
                      angr_process_noise: float = 1e-5, att_process_noise: float = 1e-5) -> tuple:
    """
    Generates the H, R, F and Q Kalman matrices from expected errors.
    This function generates and returns a tuple containing the kalman
    matrices. Specifically:
    * H: Measurement Matrix (measurements in Local NED/Earth axes)
    * R: Measurement Noise Matrix (measurements in Sensor axes)
    * F: Predict State Vector and Covariance Matrices for time step
    * Q: Process Noise Matrix (all states in Local NED/Earth axes)

    :param dt: The time difference between time steps.
    :type dt: float

    :param acc_meas_error: The expected accelerometer measurement error.
    :type acc_meas_error: float

    :param gyro_meas_error: The expected gyroscope measurement error.
    :type gyro_meas_error: float

    :param acc_process_noise: The expected acceleration noise.
    :type acc_process_noise: float

    :param vel_process_noise: The expected velocity noise.
    :type vel_process_noise: float

    :param angr_process_noise: The expected angle rate noise.
    :type angr_process_noise: float

    :param att_process_noise: The expected attitude noise.
    :type att_process_noise: float

    :return: A tuple holding the H, R, F and Q Kalman matrices.
    :rtype: tuple
    """
    # Ensure expected measurement errors are non-zeros to prevent mathematical errors.
    acc_meas_error = max(acc_meas_error, 1e-9)
    gyro_meas_error = max(gyro_meas_error, 1e-9)

    # Measurement error factor to allow for other non-random measurement error
    # sources (scaling errors, biases, non-orthogonality)
    measure_factor = 1.0
    acc_noise = (1e-6 * measure_factor * acc_meas_error) ** 2
    gyro_noise = (1e-6 * measure_factor * gyro_meas_error) ** 2

    # Define Measurement Matrix for Kalman Filter (measurements in Local NED/Earth axes)
    h = np.zeros([6, 15])
    h[[0, 1, 2, 3, 4, 5], [2, 5, 8, 10, 12, 14]] = 1

    # Define Measurement Noise Matrix for Kalman Filter (measurements in Sensor axes)
    r = np.zeros([6, 6])  # Modified by function (not constant)
    r[[0, 1, 2], [0, 1, 2]] = acc_noise
    r[[3, 4, 5], [3, 4, 5]] = gyro_noise

    # Define the Predict State Vector and Covariance Matrices for time step
    # TODO: Adapt to use different time steps?
    f = np.eye(15)
    f[[0, 1, 3, 4, 6, 7, 9, 11, 13], [1, 2, 4, 5, 7, 8, 10, 12, 14]] = dt
    f[[0, 3, 6], [2, 5, 8]] = 0.5 * dt ** 2

    # Define Process Noise Matrix for Kalman Filter (all states in Local NED/Earth axes)
    process_acceleration_noise = np.array([
        [0.25 * dt ** 4, 0.5 * dt ** 3, 0.5 * dt ** 2],
        [0.5 * dt ** 3, dt ** 2, dt],
        [0.5 * dt ** 2, dt, 1.0]
    ]) * (acc_process_noise ** 2)

    process_velocity_noise = np.array([
        [0.5 * dt ** 2, dt],
        [dt, 1]
    ]) * (vel_process_noise ** 2)

    process_angle_rate_noise = np.array([
        [0.25 * dt ** 4, 0.5 * dt ** 3],
        [0.5 * dt ** 3, dt ** 2]
    ]) * (angr_process_noise ** 2)

    process_angular_noise = dt ** 2 * (att_process_noise ** 2)

    #  Build Process noise matrix
    q = np.zeros([15, 15])
    q[0:3, 0:3] = process_acceleration_noise
    q[3:6, 3:6] = process_acceleration_noise
    q[6:9, 6:9] = process_acceleration_noise
    q[0:2, 0:2] += process_velocity_noise
    q[3:5, 3:5] += process_velocity_noise
    q[6:8, 6:8] += process_velocity_noise
    q[9:11, 9:11] = process_angle_rate_noise
    q[11:13, 11:13] = process_angle_rate_noise
    q[13:15, 13:15] = process_angle_rate_noise
    q[9, 9] += process_angular_noise
    q[11, 11] += process_angular_noise
    q[13, 13] += process_angular_noise

    # Finally, return the assembled matrices
    return h, r, f, q


def arrays_to_state_vector(position: np.ndarray,
                           velocity: np.ndarray,
                           acceleration: np.ndarray,
                           attitude: np.ndarray,
                           angle_rates: np.ndarray) -> np.ndarray:
    """
    Converts individual arrays to state vector format.

    :param position: The estimated position vector containing the
        latitude (in degrees), longitude (in degrees) and altitude (in
        metres) respectively.
    :type position: numpy.ndarray (3-elements)

    :param velocity: The estimated velocity vector for the
        North, East and Down components respectively (in m/s).
    :type velocity: numpy.ndarray (3-elements)

    :param acceleration: The estimated acceleration vector for
        the North, East and Down components respectively (in m/s^2).
    :type acceleration: numpy.ndarray (3-elements)

    :param attitude: The estimated attitude vector for the
        Heading, Pitch and Roll components respectively (in degrees).
    :type attitude: numpy.ndarray (3-elements)

    :param angle_rates: The estimated angle rate vector for
        the P, Q and R body rates respectively (in deg/s).
    :type angle_rates: numpy.ndarray (3-elements)

    :return: A state vector array containing the above.
    :rtype: numpy.ndarray (15-elements)
    """
    state_vector = np.zeros(15)
    state_vector[[0, 3, 6]] = position
    state_vector[[1, 4, 7]] = velocity
    state_vector[[2, 5, 8]] = acceleration
    state_vector[[9, 11, 13]] = attitude
    state_vector[[14, 12, 10]] = angle_rates
    return state_vector


def state_vector_to_arrays(state_vector: np.ndarray) -> tuple:
    """
    Converts from state vector format to individual arrays.

    :param state_vector: A state vector array to be converted.
    :type state_vector: numpy.array (15-elements)

    :return: A tuple containing the position, velocity, acceleration,
        attitude and angle rate vectors respectively.
    :rtype: tuple
    """
    position = state_vector[[0, 3, 6]]
    velocity = state_vector[[1, 4, 7]]
    acceleration = state_vector[[2, 5, 8]]
    attitude = state_vector[[9, 11, 13]]
    angle_rates = state_vector[[14, 12, 10]]
    return position, velocity, acceleration, attitude, angle_rates
