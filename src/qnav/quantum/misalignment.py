"""
===============
misalignment.py
===============

:summary:
    Extended implementations of Quantum Sensor Fusion for axis misalignment.
    This module contains fusion methods for quantum sensors that, in addition
    to position fixing, also correct axis misalignment, resulting in errors
    due to non-matching actual and estimated sensor axis.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""

from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.platform import SensorAxis
from qnav.quantum.particle_filter import QuantumPFFusion
from qnav.quantum.particle_filter import ParticleFilterParams
from qnav.quantum.base import ConceptQuantumImu

import numpy as np
import qnav.util.transformations as trans


class QuantumPFMFusion(QuantumPFFusion):
    """
    Quantum sensor fusion with particle filter estimation for axis misalignment.
    This provides position and axis misalignment correction by using a particle
    filter approach for estimating the bias errors within the class IMU sensors.
    This extends from the base class, overriding the relevant methods.
    """

    def __init__(self, qs_imu: ConceptQuantumImu,
                 shared_accelerometer: Accelerometer,
                 shared_gyroscope: Gyroscope,
                 est_qs_imu_axis: SensorAxis = None,
                 est_acc_axis: SensorAxis = None,
                 est_gyro_axis: SensorAxis = None,
                 filter_params: ParticleFilterParams = ParticleFilterParams()):
        """
        Generates a quantum sensor particle filter fusion instance.

        :param qs_imu: The quantum sensor to use for quantum IMU measurements.
        :type qs_imu: ConceptQuantumImu

        :param shared_accelerometer: The IMU's accelerometer instance.
        :type shared_accelerometer: Accelerometer

        :param shared_gyroscope: The IMU's gyroscope instance.
        :type shared_gyroscope: Gyroscope

        :param est_qs_imu_axis: (Optional) Estimated sensor axis for
            quantum IMU. If not provided, true axis will be used.
        :type est_qs_imu_axis: SensorAxis

        :param est_acc_axis: (Optional) Estimated sensor axis for
            accelerometer. If not provided, true axis will be used.
        :type est_acc_axis: SensorAxis

        :param est_gyro_axis: (Optional) Estimated sensor axis for
            gyroscope. If not provided, true axis will be used.
        :type est_gyro_axis: SensorAxis

        :param filter_params: (Optional) Parameter for particle filtering.
            This contains all parameter values to use during the filtering
             stage. Can be customised to improve accuracy.
        :type filter_params: ParticleFilterParams
        """

        super().__init__(qs_imu, shared_accelerometer, shared_gyroscope,
                         est_qs_imu_axis, est_acc_axis, est_gyro_axis,
                         filter_params)

        self._sigma_mis_align = filter_params.sigma_misalignment

        record_size = (self._num_particles, 3)
        self._x_particles_mis = 4 * self._sigma_mis_align * self._rng.normal(size=record_size)
        self._w_particles = np.ones(self._num_particles) / self._num_particles
        # self._w_particles_acc = np.copy(self._w_particles)
        # self._w_particles_gyro = np.copy(self._w_particles)


    def _calculate_bias_errors(self, acceleration_qs_imu: np.ndarray,
                               angle_rate_qs_imu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates and returns the accelerometer and gyroscope bias.
        Using the collected IMU measurements and calculated quantum IMU
        measurements, this method examines the difference and returns the
        estimated bias for each.

        :param acceleration_qs_imu: The quantum acceleration measurement
            (given in converted IMU axis).

        :param angle_rate_qs_imu: The quantum gyroscope measurement
            (given in converted IMU axis).

        :return: The estimated accelerometer and gyroscope biases, respectively.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Obtain the average acceleration value from the measurement record.
        mean_imu_acceleration = np.mean(self._imu_acc_data, 0)

        # Obtain the average angle rate value from the measurement record.
        mean_imu_angle_rate = np.mean(self._imu_gyro_data, 0)
        # mean_imu_angle_rate = np.radians(mean_imu_angle_rate)

        # Obtain rotation matrix for estimated IMU axis.
        sensor_angles_imu = self._acc_est_axis.sensor_angles_rad
        rot_body2sensor_imu = trans.rotate_3d(*sensor_angles_imu)

        # Use the last estimated attitude for correction
        attitude = np.deg2rad(self._estimated_state.attitude)
        rot_earth2body = trans.rotate_3d(*attitude)

        # Rotate gravity vector to sensor axes and form misalignment covariance matrix
        tmp = np.eye(3)
        tmp[2, 2] = 100  # TODO: WHY IS THIS HERE?

        co_var_mis = ((1.0 * np.radians(self._sigma_mis_align)) ** 2
                      * rot_body2sensor_imu @ rot_earth2body
                      @ tmp @ rot_earth2body.T @ rot_body2sensor_imu.T)

        for i in range(self._num_particles):

            mean_imu_acceleration_0 = mean_imu_acceleration - self._x_particles_acc[i, :]
            mean_imu_angle_rate_0 = mean_imu_angle_rate - self._x_particles_gyro[i, :]
            # mean_imu_angle_rate_0 = mean_imu_angle_rate - np.radians(self._x_particles_gyro[i, :])

            misalignment_0 = np.radians(self._x_particles_mis[i, 0:3])
            rot_sensor2measurement = trans.rotate_3d(*misalignment_0)

            acceleration_qs_0 = np.linalg.solve(rot_sensor2measurement, acceleration_qs_imu)
            angle_rate_qs_0 = np.linalg.solve(rot_sensor2measurement, angle_rate_qs_imu)

            deviation_1 = mean_imu_acceleration_0 - acceleration_qs_0
            co_var_acc = rot_sensor2measurement.T @ co_var_mis @ rot_sensor2measurement
            sigma_1 = co_var_acc + (1.0 + self._sigma_acc) ** 2 * np.eye(3)
            # self._w_particles[i] *= np.exp(-0.5 * deviation_1 / sigma_1 * deviation_1)
            self._w_particles[i] *= np.exp(-0.5 * np.linalg.solve(sigma_1.T, deviation_1.T).T @ deviation_1)


            deviation_2 = np.rad2deg(mean_imu_angle_rate_0 - angle_rate_qs_0)
            sigma_2 = (1.0 * self._sigma_gyro) ** 2 * np.eye(3)
            # self._w_particles[i] *= np.exp(-0.5 * deviation_2 / sigma_2 * deviation_2)
            self._w_particles[i] *= np.exp(-0.5 * np.linalg.solve(sigma_2.T, deviation_2.T).T @ deviation_2)
            self._w_particles[i] = max(float(self._w_particles[i]), 1e-16)

        # Re-weight particles
        self._w_particles /= np.sum(self._w_particles)

        # Group particles together so they share the same weighting
        all_x_particles = np.column_stack((
            self._x_particles_mis, self._x_particles_acc, self._x_particles_gyro))

        # Apply conditional re-sampling
        self._w_particles, all_x_particles = self._conditional_resample(
            self._w_particles, all_x_particles)

        # Ungroup particles back to original variables
        self._x_particles_mis = all_x_particles[:, 0:3]
        self._x_particles_acc = all_x_particles[:, 3:6]
        self._x_particles_gyro = all_x_particles[:, 6:9]

        # Add process noise
        record_size = (self._num_particles, 3)
        # all_x_particles[:, 0:3] += 0.2 * self._sigma_mis_align * self._rng.normal(size=record_size)
        # all_x_particles[:, 3:6] += 0.2 * self._sigma_acc * self._rng.normal(size=record_size)
        # all_x_particles[:, 6:9] += 0.1 * self._sigma_gyro * self._rng.normal(size=record_size)
        self._x_particles_mis += 0.2 * self._sigma_mis_align * self._rng.normal(size=record_size)
        self._x_particles_acc += 0.2 * self._sigma_acc * self._rng.normal(size=record_size)
        self._x_particles_gyro += 0.1 * self._sigma_gyro * self._rng.normal(size=record_size)

        # Re-weight particles
        self._w_particles /= np.sum(self._w_particles)

        # Calculate the misalignment and bias corrections
        # est_misalignment = self._w_particles @ self._x_particles_mis
        accelerometer_biases = self._w_particles @ self._x_particles_acc
        gyroscope_biases = self._w_particles @ self._x_particles_gyro

        # Return the calculated bias errors
        return accelerometer_biases, gyroscope_biases

    def _resample(self, weights: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs Importance Resampling (on top of Systematic Resampling):
        The function applies the core step for the particle filtering by updating
        the weights of approximated particles. This works by dividing the whole
        population of particles into subpopulations. It pre-partitions the (0, 1]
        interval into N disjoint sub-intervals. Random numbers are then drawn and
        then a bounding method based on the cumulative sum of normalized weights
        is used. The first random number is taken from the uniform distribution
        on (0 1, /N], and the rest of the numbers are obtained deterministically.
        Once completed, the updated particles and their new weights are returned.

        :param weights: An array containing the current particle weights.
        :type weights: np.ndarray

        :param values: An array containing the current particle values.
        :type values: np.ndarray

        :return: The particles, their updated weights and their indices.
        :rtype: tuple (3-elements)
        """

        _, new_values, p_i = super()._resample(weights, values)
        n = len(weights)

        qw = np.zeros(n)
        for i in range(n):
            qw[i] = np.sum(new_values == values[i]) / n

        new_weights = np.zeros(n)
        for i in range(n):
            new_weights[i] = weights[p_i[i]] / qw[p_i[i]]

        new_weights /= np.sum(new_weights)

        return new_weights, new_values
