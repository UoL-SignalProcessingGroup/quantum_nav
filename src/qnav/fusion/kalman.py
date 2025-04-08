"""
=========
kalman.py
=========

:summary:
    Kalman basd fusion method for supplying Inertial Navigation.
    Provides Kalman Filter fusion for the Inertial Navigation system (INS).

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

from qnav.estimation.state import KalmanEstimatedState
from qnav.util.transformations import cross_prod_xy
from qnav.util.constants import OMEGA_E
from qnav.fusion.ins import NumericalINS
from qnav.gravity.base import GravityModel
from math import degrees, radians, cos, sin, tan

import qnav.util.transformations as trans
import numpy as np

class KalmanINS(NumericalINS):
    """
    Kalman Filter Inertial Navigation System (INS).
    This class represents a Kalman filter that uses linear quadratic
    estimations to update the estimated states and estimations of
    measurement errors. This solution is known to be more accurate
    than the other more basic INS methods provided in the project.
    """

    def perform_fusion(self, kalman_state: KalmanEstimatedState) -> None:
        """
        Performs fusion using latest accelerometer and gyroscope measurements.
        Updates the estimated states by performing Kalman Filtering.

        :param kalman_state: The current estimated state.
        :type kalman_state: EstimatedState
        """

        # Obtain the measurements from the accelerometer and gyroscope
        measured_acceleration = self._accelerometer.last_measurement
        measured_angle_rates = self._gyroscope.last_measurement

        # Calculate gravity vector modified by Earth's rotation (metres/sec^2)
        ref_lla = kalman_state.position

        # Calculate the estimated gravity vector
        gravity_model = kalman_state.gravity_model
        g = gravity_model.calc_gravity_xyz(*ref_lla)

        # Define Angular velocity for Earth's rotation (in local NED axes)
        lat_rad = radians(float(ref_lla[0]))
        omega_e = OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

        # Redefine notation for matrices to simplify the notation later on
        h_matrix = kalman_state.h_matrix
        r_matrix = kalman_state.r_matrix
        f_matrix = kalman_state.f_matrix
        q_matrix = kalman_state.q_matrix

        # Use current LLA position to define local reference axes (Local NED)
        state_vector = kalman_state.state_vector
        state_errors = kalman_state.state_errors

        # Convert Existing State Vector from LLA-body axes to Local NED axes
        state_vector = state_vector2ned(state_vector, ref_lla, gravity_model)

        # Get the estimated velocity in NED axis.
        velocity_e0 = state_vector[[1, 4, 7]]

        # Measured Body PQR Angle Rates to Euler Angle Rates
        psi, theta, phi = state_vector[[9, 11, 13]]

        # # Sensor to Body axes rotation matrix
        # sensor_axis = kalman_state.sensor_axis
        # sensor_angles = sensor_axis.sensor_angles_rad
        # lever_arm = sensor_axis.lever_arm

        # Unpack required sensor axis values:
        acc_lever_arm = self._acc_axis.lever_arm
        acc_rot_body2sensor = self._acc_axis.body2sensor_mat
        gyro_rot_body2sensor = self._gyro_axis.body2sensor_mat

        # Rotation matrix from body to sensor axes
        # rot_body2sensor = trans.rotate_3d(*sensor_angles)

        # Rotation matrix from Body axes to Local NED
        rot_earth2body = trans.rotate_3d(psi, theta, phi)

        # Rotation matrix from Body Angle Rates (PQR) to Euler Angle Rate
        rot_pqr2euler = np.array([
            [1.0, sin(phi) * tan(theta), cos(phi) * tan(theta)],
            [0.0, cos(phi), -sin(phi)],
            [0.0, sin(phi) * (1 / cos(theta)), cos(phi) * (1 / cos(theta))]
        ])

        measured_angle_rate_s1 = np.radians(measured_angle_rates)

        # Convert angle rates from sensor axes to body axes
        measured_angle_rate_b1 = np.linalg.solve(gyro_rot_body2sensor, measured_angle_rate_s1)

        #  Convert measured accelerations from sensor axes to body axes
        measured_angle_rate_b0 = np.radians(kalman_state.angle_rates)
        measured_acceleration_b1 = np.linalg.solve(acc_rot_body2sensor, measured_acceleration) - cross_prod_xy(
            measured_angle_rate_b0, cross_prod_xy(measured_angle_rate_b0, acc_lever_arm))

        # Convert measured acceleration from body axes to Earth axes and remove Coriolis and gravity terms
        measured_acceleration_e1 = np.linalg.solve(rot_earth2body, measured_acceleration_b1) + g
        measured_angle_rate_b1 -= rot_earth2body @ omega_e

        omega_tr = trans.get_transport_rate(ref_lla, np.linalg.solve(
            rot_earth2body, kalman_state.velocity))
        measured_angle_rate_b1 -= rot_earth2body @ omega_tr
        measured_euler_rates = rot_pqr2euler @ measured_angle_rate_b1

        # Remove effect of Coriolis term due to Earth's rotation and transport rate (in Local NED axes)
        measured_acceleration = measured_acceleration_e1 - 2.0 * cross_prod_xy(
            omega_e, velocity_e0) - cross_prod_xy(omega_tr, velocity_e0)

        r = np.copy(r_matrix)

        # Convert sensor errors to PQR errors
        r[3:6, 3:6] = np.linalg.solve(gyro_rot_body2sensor, r[3:6, 3:6]) * gyro_rot_body2sensor

        # Convert PQR errors to Euler angle errors
        r[3:6, 3:6] = rot_pqr2euler @ r[3:6, 3:6] @ rot_pqr2euler.T

        # Convert measured sensor acceleration errors to body axes
        r[0:3, 0:3] = np.linalg.solve(acc_rot_body2sensor, r[0:3, 0:3]) * acc_rot_body2sensor

        # Convert acceleration errors from body axes to Local NED measurement errors
        r[0:3, 0:3] = np.linalg.solve(rot_earth2body, r[0:3, 0:3]) * rot_earth2body

        """
        ======================================================================
        Kalman Filter Operations
        ======================================================================
        1. Measurement
            1a. Take New Measurement
            1b. Calculate Expected Measurement

        2. Update State
            2a. Calculate Innovation
            2b. Calculate Kalman Gain
            2c. Calculate Updated State Vector
            2d. Calculate Update State Error Matrix

        3. Predict State and Errors forward one time step
            3a. Calculate Predicted State Vector
            3b. Calculate Predicted Error Matrix

        4. Return to 1...
        """

        """ 
        1. Measurement
        """
        # 1a. Take New Measurement
        actual_measurement = np.concatenate(
            [measured_acceleration, measured_euler_rates[[2, 1, 0]]])

        # 1b. Calculate Expected Measurement
        expected_measurement = h_matrix @ state_vector

        """
        2. Update State
        """
        # 2a. Calculate Innovation
        innovation_vector = actual_measurement - expected_measurement

        # 2b. Calculate Kalman Gain
        s_matrix = r_matrix + h_matrix @ state_errors @ h_matrix.T
        k_matrix = state_errors @ h_matrix.T @ np.linalg.inv(s_matrix)

        # 2c. Calculate Updated State Vector
        state_vector += k_matrix @ innovation_vector

        # 2d. Calculate Update State Error Matrix
        state_errors -= k_matrix @ s_matrix @ k_matrix.T

        """
        3. Predict State and Errors forward one time step
        """
        # 3a. Calculate Predicted State Vector
        state_vector = f_matrix @ state_vector

        # 3b. Calculate Predicted Error Matrix
        state_errors = f_matrix @ state_errors @ f_matrix.T + q_matrix

        """
        ======================================================================
        Clean Up and Returning Values
        ======================================================================
        """

        # Check the attitude angles (periodicity)
        state_vector[[9, 11, 13]] = np.remainder(
            state_vector[[9, 11, 13]] + np.pi, 2 * np.pi) - np.pi

        state_vector = ned2state_vector(state_vector, ref_lla, gravity_model)

        kalman_state.update_estimates(
            position=state_vector[[0, 3, 6]],
            velocity=state_vector[[1, 4, 7]],
            acceleration=state_vector[[2, 5, 8]],
            attitude=state_vector[[9, 11, 13]],
            angle_rates = state_vector[[14, 12, 10]],
            state_errors=state_errors
        )


def state_vector2ned(state_vector: np.ndarray,
                     ref_lla: np.ndarray,
                     gravity_model: GravityModel) -> np.ndarray:
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

    :param gravity_model: The model for estimating downwards acceleration.
    :type gravity_model: GravityModel

    :return: The given state vector array and measurements converted to
        local NED/Earth axes for Kalman filtering.
    :rtype: numpy.ndarray (15-elements)
    """

    # Calculate the estimated gravity vector
    g = gravity_model.calc_gravity_xyz(*ref_lla)

    # Define Angular velocity for Earth's rotation (in local NED axes)
    lat_rad = radians(float(ref_lla[0]))
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


def ned2state_vector(state_vector: np.ndarray,
                      ref_lla: np.ndarray,
                      gravity_model: GravityModel) -> np.ndarray:
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

    :param gravity_model: The model for estimating downwards acceleration.
    :type gravity_model: GravityModel

    :return: The given state vector array converted to LLA.
    :rtype: numpy.ndarray (15-elements)
    """

    # Calculate the gravity vector for the current position
    g = gravity_model.calc_gravity_xyz(*ref_lla)

    # Define Angular velocity for Earth's rotation (in local NED axes)
    lat_rad = radians(float(ref_lla[0]))
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
    state_vector[9] = degrees(state_vector[9])
    state_vector[10] = angle_rate_b[2]
    state_vector[11] = degrees(state_vector[11])
    state_vector[12] = angle_rate_b[1]
    state_vector[13] = degrees(state_vector[13])
    state_vector[14] = angle_rate_b[0]

    # Finally, return the state vector
    return state_vector


# def generate_matrices(imu_frequency: float,
#                       accelerometer_error: ErrorProperties,
#                       gyroscope_errors: ErrorProperties) -> tuple:
#     """
#     Generates the H, R, F and Q Kalman matrices from expected errors.
#     This function generates and returns a tuple containing the kalman
#     matrices. Specifically:
#     * H: Measurement Matrix (measurements in Local NED/Earth axes)
#     * R: Measurement Noise Matrix (measurements in Sensor axes)
#     * F: Predict State Vector and Covariance Matrices for time step
#     * Q: Process Noise Matrix (all states in Local NED/Earth axes)
#
#     :param imu_frequency: The frequency of the IMU measurements.
#     :type imu_frequency: float
#
#     :param accelerometer_error: The expect error profile for of the
#         accelerometer measurements.
#     :type accelerometer_error: ErrorProperties
#
#     :param gyroscope_errors: The expect error profile for of the
#         gyroscope measurements.
#     :type gyroscope_errors: ErrorProperties
#
#     :return: A tuple holding the H, R, F and Q Kalman matrices.
#     :rtype: tuple
#     """
#
#     #
#     dt = 1 / imu_frequency
#
#     acc_meas_error = float(np.mean(accelerometer_error.avg_meas_noise))
#     gyro_meas_error = float(np.mean(gyroscope_errors.avg_meas_noise))
#
#     vel_process_noise = 0
#     acc_process_noise = 0
#     att_process_noise = 0
#     angr_process_noise = 0
#
#     # Ensure expected measurement errors are non-zeros to prevent mathematical errors.
#     acc_meas_error = max(acc_meas_error, 1e-9)
#     gyro_meas_error = max(gyro_meas_error, 1e-9)
#
#     # Measurement error factor to allow for other non-random measurement error
#     # sources (scaling errors, biases, non-orthogonality)
#     measure_factor = 1.0
#     acc_noise = (1e-6 * measure_factor * acc_meas_error) ** 2
#     gyro_noise = (1e-6 * measure_factor * gyro_meas_error) ** 2
#
#     # Define Measurement Matrix for Kalman Filter (measurements in Local NED/Earth axes)
#     h = np.zeros([6, 15])
#     h[[0, 1, 2, 3, 4, 5], [2, 5, 8, 10, 12, 14]] = 1
#
#     # Define Measurement Noise Matrix for Kalman Filter (measurements in Sensor axes)
#     r = np.zeros([6, 6])  # Modified by function (not constant)
#     r[[0, 1, 2], [0, 1, 2]] = acc_noise
#     r[[3, 4, 5], [3, 4, 5]] = gyro_noise
#
#     # Define the Predict State Vector and Covariance Matrices for time step
#     f = np.eye(15)
#     f[[0, 1, 3, 4, 6, 7, 9, 11, 13], [1, 2, 4, 5, 7, 8, 10, 12, 14]] = dt
#     f[[0, 3, 6], [2, 5, 8]] = 0.5 * dt ** 2
#
#     # Define Process Noise Matrix for Kalman Filter (all states in Local NED/Earth axes)
#     process_acceleration_noise = np.array([
#         [0.25 * dt ** 4, 0.5 * dt ** 3, 0.5 * dt ** 2],
#         [0.5 * dt ** 3, dt ** 2, dt],
#         [0.5 * dt ** 2, dt, 1.0]
#     ]) * (acc_process_noise ** 2)
#
#     process_velocity_noise = np.array([
#         [0.5 * dt ** 2, dt],
#         [dt, 1]
#     ]) * (vel_process_noise ** 2)
#
#     process_angle_rate_noise = np.array([
#         [0.25 * dt ** 4, 0.5 * dt ** 3],
#         [0.5 * dt ** 3, dt ** 2]
#     ]) * (angr_process_noise ** 2)
#
#     process_angular_noise = dt ** 2 * (att_process_noise ** 2)
#
#     #  Build Process noise matrix
#     q = np.zeros([15, 15])
#     q[0:3, 0:3] = process_acceleration_noise
#     q[3:6, 3:6] = process_acceleration_noise
#     q[6:9, 6:9] = process_acceleration_noise
#     q[0:2, 0:2] += process_velocity_noise
#     q[3:5, 3:5] += process_velocity_noise
#     q[6:8, 6:8] += process_velocity_noise
#     q[9:11, 9:11] = process_angle_rate_noise
#     q[11:13, 11:13] = process_angle_rate_noise
#     q[13:15, 13:15] = process_angle_rate_noise
#     q[9, 9] += process_angular_noise
#     q[11, 11] += process_angular_noise
#     q[13, 13] += process_angular_noise
#
#     # Finally, return the assembled matrices
#     return h, r, f, q