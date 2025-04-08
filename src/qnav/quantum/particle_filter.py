"""
==================
particle_filter.py
==================

:summary:
    Particle filter implementations for quantum sensor fusion.
    This module contains fusion methods for quantum sensors that employ
    particle filter estimation.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""

from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.platform import SensorAxis
from qnav.quantum.base import ConceptQuantumFusion
from qnav.quantum.base import ConceptQuantumImu
from dataclasses import dataclass
from math import isinf

import numpy as np


@dataclass(frozen=True)
class ParticleFilterParams:
    """
    Parameters for a particle filter fusion for quantum sensors.
    This contains the arguments used for applying particle filtering for
    position fixing with IMU quantum sensors. This namely contains the
    number of particles to use and the expected errors.
    """

    # The number of particles to use for filtering
    num_particles: int = 200

    # The sigma value to use for average accelerometer bias
    sigma_accelerometer: float = 0.01

    # The sigma value to use for average gyroscope bias
    sigma_gyroscope: float = 0.01

    # The sigma value to use for axes misalignment
    sigma_misalignment: float = 0.01

    # The seed for random number generation
    rng_seed: int = None

    # The probability of applying low noise instead of high.
    low_noise_prob: float = 0.9

    # The low noise ratio to apply.
    low_noise: float = 0.1

    # The high noise ratio to apply.
    high_noise: float = 2.0


    def __post_init__(self):
        """
        Supplies post-generation validation for given filtering parameters.
        Used to ensure given parameters are valid and usable for performing
        particle filtering.
        """

        if self.num_particles < 1 or isinf(self.num_particles):
            raise ValueError("num_particles must be positive integer")

        if self.sigma_accelerometer < 0 or isinf(self.sigma_accelerometer):
            raise ValueError("sigma_accelerometer must be non-negative and finite")

        if self.sigma_gyroscope < 0 or isinf(self.sigma_gyroscope):
            raise ValueError("sigma_gyroscope must be non-negative and finite")

        if self.sigma_misalignment < 0 and isinf(self.sigma_misalignment):
            raise ValueError("sigma_misalignment must be non-negative and finite")


class QuantumPFFusion(ConceptQuantumFusion):
    """
    Quantum sensor fusion with particle filter estimation.
    This provides position correction by using a particle filter approach for
    estimating the bias errors within the class IMU sensors. This extends
    from the base class, overriding the relevant methods.
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

        # Call parent class's constructor
        super().__init__(qs_imu, shared_accelerometer, shared_gyroscope,
                         est_qs_imu_axis, est_acc_axis, est_gyro_axis)

        # Unpack filtering parameters and convert to required units:
        self._filter_params = filter_params
        self._num_particles = filter_params.num_particles

        # self._sigma_acc = filter_params.sigma_accelerometer * 1e-6 * G
        # self._sigma_gyro = filter_params.sigma_gyroscope * 1e-6 * G  # TODO: Radians?

        # TODO: CHECK?
        self._sigma_acc = max(filter_params.sigma_accelerometer, 1e-12)
        self._sigma_gyro = max(filter_params.sigma_gyroscope, 1e-12)

        # Generate a random number stream for
        self._rng = np.random.default_rng(filter_params.rng_seed)

        # Initialise the particle weights for accelerometer and gyroscope measurements
        self._w_particles_acc = np.ones(self._num_particles) / self._num_particles
        self._w_particles_gyro = np.ones(self._num_particles) / self._num_particles

        # Initialise particle values for accelerometer and gyroscope measurements
        record_size = (self._num_particles, 3)
        self._x_particles_acc = 4 * self._sigma_acc * self._rng.normal(size=record_size)
        self._x_particles_gyro = 4 * self._sigma_gyro * self._rng.normal(size=record_size)


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
        mean_imu_acceleration = np.mean(self._imu_acc_data, axis=0)

        # Obtain the average angle rate value from the measurement record.
        mean_imu_angle_rate = np.mean(self._imu_gyro_data, axis=0)

        # Obtain the acceleration particle weight adjustment
        cand_acc = mean_imu_acceleration - self._x_particles_acc
        deviation_acc = np.linalg.norm(cand_acc - acceleration_qs_imu, axis=1)
        adjustment_acc = np.exp(-0.5 * deviation_acc ** 2 / (self._sigma_acc ** 2))

        # Obtain the angle rate particle weight adjustment
        cand_gyro = mean_imu_angle_rate - self._x_particles_gyro
        deviation_gyro = np.linalg.norm(cand_gyro - angle_rate_qs_imu, axis=1)
        adjustment_gyro = np.exp(-0.5 * deviation_gyro ** 2 / (self._sigma_gyro ** 2))

        # Re-weight particles
        self._w_particles_acc *= adjustment_acc
        self._w_particles_gyro *= adjustment_gyro

        min_limit = 1e-50

        if np.any(self._w_particles_acc < min_limit):
            self._w_particles_acc += min_limit

        if np.any(self._w_particles_gyro < min_limit):
            self._w_particles_gyro += min_limit

        # Normalise the particle weights
        self._w_particles_acc /= np.sum(self._w_particles_acc)
        self._w_particles_gyro /= np.sum(self._w_particles_gyro)

        # Apply conditional re-sampling for acceleration particles
        self._w_particles_acc, self._x_particles_acc = self._conditional_resample(
            self._w_particles_acc, self._x_particles_acc)

        # Apply conditional re-sampling for angle rate particles
        self._w_particles_gyro, self._x_particles_gyro = self._conditional_resample(
            self._w_particles_gyro, self._x_particles_gyro)

        # Add process noise
        record_size = (self._num_particles, 3)
        self._x_particles_acc += 0.05 * self._sigma_acc * self._rng.normal(size=record_size)
        self._x_particles_gyro += 0.05 * self._sigma_gyro * self._rng.normal(size=record_size)

        # Re-weight particles
        self._w_particles_acc /= np.sum(self._w_particles_acc)
        self._w_particles_gyro /= np.sum(self._w_particles_gyro)

        # Recalculate the estimated values for the biases
        accelerometer_biases = self._w_particles_acc @ self._x_particles_acc
        gyroscope_biases = self._w_particles_gyro @ self._x_particles_gyro

        # Return the calculated bias errors
        return accelerometer_biases, gyroscope_biases


    def reset_pf(self):

        # Initialise the particle weights for accelerometer and gyroscope measurements
        self._w_particles_acc = np.ones(self._num_particles) / self._num_particles
        self._w_particles_gyro = np.ones(self._num_particles) / self._num_particles

        # Initialise particle values for accelerometer and gyroscope measurements
        record_size = (self._num_particles, 3)
        self._x_particles_acc = 4 * self._sigma_acc * self._rng.normal(size=record_size)
        self._x_particles_gyro = 4 * self._sigma_gyro * self._rng.normal(size=record_size)


    def _conditional_resample(self, weights: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs particle resampling when deemed necessary.
        Resamples the given weights and values if required, otherwise returns
        the given weights and values.

        :param weights: A selection of particle weights.
        :type weights: np.ndarray

        :param values: A selection of corresponding particle values.
        :type values: np.ndarray

        :return: The resampled (or original if not deemed necessary)
            weights and values, respectively.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        prob = self._filter_params.low_noise_prob
        low_noise = self._filter_params.low_noise
        high_noise = self._filter_params.high_noise

        n = 1
        # TODO: WHY NEEDED?
        if values.ndim > 1:
            n = values.shape[1]

        # Calculate the threshold and weight distribution
        n_pf_thr = 0.5 * self._num_particles
        n_eff = 1 / np.sum(weights ** 2)

        # If above the threshold, return:
        if n_eff >= n_pf_thr:
            return weights, values

        # Calculate the weight covariance matrix
        weights[weights <= 0] = 1e-300  # TODO: REMOVE/RESOLVE
        co_var_q = weighted_cov(values, weights)

        # Perform weight and value resampling
        new_weights, new_values, _ = self._resample(weights, values)
        l_q = np.linalg.cholesky(co_var_q)

        # Add random process noise particle
        for i in range(self._num_particles):
            if self._rng.random() < prob:
                l_q *= low_noise
            else:
                l_q *= high_noise
            new_values[i, :] += (l_q @ self._rng.normal(size=n))

        # Return resampled weights and values
        return new_weights, new_values

    def _resample(self, weights: np.ndarray, values: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Systematic Resampling:
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

        # Obtain the normalised weights
        probs = weights / np.sum(weights)
        n = len(weights)

        # Select Points for Resampling
        new_weights = np.ones(weights.shape) / n
        new_values = np.zeros(values.shape)
        p_i = np.zeros(weights.shape)
        indices = np.arange(n)

        for i in indices:
            j = self._rng.choice(indices, 1, True, probs)
            new_values[i, :] = values[j, :]
            p_i[i] = j

        return new_weights, new_values, p_i


def weighted_cov(y: np.ndarray, w: np.ndarray):
    """
    This function calculates a Weighted Covariance Matrix.
    Given a matrix of observations (rows) and variables (columns) along with
    an array of corresponding weights, this function returns a symmetric
    matrix of weighted covariances (a positive semi-definite matrix i.e. all
    its eigenvalues are non-negative).

    :param y: A T-by-N matrix of observations (rows) and variables (columns).
    :type y: NumPy Array (T-by-N elements)

    :param w: A T-element array of weights for the observations.
    :type w: NumPy Array (T elements)

    :return: A symmetric matrix of weighted covariances.
    :rtype: NumPy Array
    """

    w = np.real(w)

    ctrl = not np.isscalar(w) and np.all(np.isreal(w)) and \
           not np.any(np.isnan(w)) and not np.any(np.isinf(w)) \
           and np.all(w > 0)

    if ctrl:
        w = w.flatten() / np.sum(w)
    else:
        raise ValueError("None finite weight produced")

    ctrl = np.all(np.isreal(y)) and not np.any(np.isnan(y)) and \
           not np.any(np.isinf(y)) and len(y.shape) == 2

    if not ctrl:
        raise ValueError("None finite observation produced")

    ctrl = len(w) == y.shape[0]

    if not ctrl:
        raise ValueError("Non-uniform shape produced")

    w = w.flatten() / np.sum(w)
    t, n = y.shape

    c = y - np.tile(w @ y, [t, 1])
    c = c.T @ (c * np.tile(w.reshape(-1, 1), [1, n]))
    c = 0.5 * (c + c.T)

    return c
