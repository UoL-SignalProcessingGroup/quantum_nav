
import pytest
import numpy as np
from math import sqrt

from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.waypoints.trajectory import GroundTruth

from .conftest import assert_array_equal


def get_measurement(gyroscope: Gyroscope,
                    ground_truth: GroundTruth,
                    num_iter: int) -> np.ndarray:
    """
    Obtains a series of measurements from an gyroscope.
    Produces a collection of measurements captured from an gyroscope
    instance, given the same ground truth data with errors updated at
    each time step.

    :param gyroscope: An gyroscope instance to get measurements from.
    :type gyroscope: Gyroscope

    :param ground_truth: A ground truth instance to feed values with.
    :type ground_truth: GroundTruth

    :param num_iter: The number of interactions to get measurements for.
    :type num_iter: int

    :return: An array holding all obtained measurements.
    :rtype: np.ndarray
    """

    measurements = np.zeros((num_iter, 3))

    for i in range(num_iter):
        gyroscope.take_measurement(0, ground_truth)
        measurements[i, :] = gyroscope.last_measurement
        gyroscope.update()

    return measurements


@pytest.mark.parametrize("args",(
        {'freq': 1},
        {'freq': 100},
        {'freq': 200},
))
def test_timer(args):

    # The number of iterations to attempt.
    num_iter = 5

    freq = args['freq']
    errors = ErrorProperties()
    gyroscope = Gyroscope(freq, errors)

    dt = 1 / freq
    expected = dt

    for _ in range(num_iter):
        assert gyroscope.next_update == expected
        gyroscope.update()
        expected += dt


@pytest.mark.parametrize("args",(
        {'freq': 1, 'bias_scale': 0, 'seed': 0},
        {'freq': 1, 'bias_scale': 1, 'seed': 1},
        {'freq': 100, 'bias_scale': 500, 'seed': 100},
        {'freq': 200, 'bias_scale': 100, 'seed': 500},
))
def test_fixed_bias(random_truth: GroundTruth, args: dict):

    # The number of iterations to attempt.
    num_iter = 5

    # Initialise the error profile with parameters.
    bias_error = np.random.randn(3) * args['bias_scale']
    errors = ErrorProperties(bias_error=bias_error)

    # Initialise the gyroscope with random bias errors.
    gyroscope = Gyroscope(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_measurement(
        gyroscope, random_truth, num_iter)

    # Calculate the expected values.
    angle_rates = random_truth.angle_rates
    expected_bias = np.degrees(bias_error * 1e-6)
    expected = angle_rates + expected_bias

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)


@pytest.mark.parametrize("args",(
        {'freq': 1, 'bias_scale': 0, 'drift_scale': 0, 'seed': 0},
        {'freq': 1, 'bias_scale': 1, 'drift_scale': 0, 'seed': 1},
        {'freq': 1, 'bias_scale': 0, 'drift_scale': 1, 'seed': 2},
        {'freq': 100, 'bias_scale': 100, 'drift_scale': 10, 'seed': 100},
        {'freq': 200, 'bias_scale': 10, 'drift_scale': 1000, 'seed': 500},
))
def test_drift_rate(mock_rng, random_truth: GroundTruth, args: dict):

    # The number of iterations to attempt:
    num_iter = 5

    # Initialise the gyroscope with random bias errors
    bias_error = np.random.randn(3) * args['bias_scale']
    drift_rate = np.random.randn(3) * args['drift_scale']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(
        bias_error = bias_error,
        bias_drift_rate = drift_rate,
    )

    # Initialise the gyroscope with random bias errors.
    gyroscope = Gyroscope(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_measurement(
        gyroscope, random_truth, num_iter)

    # Calculate the expected values.
    angle_rates = random_truth.angle_rates
    expected_bias = np.degrees(bias_error * 1e-6)
    expected = angle_rates + expected_bias
    dt_sqrt = sqrt(1 / args['freq'])

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)
        expected += np.degrees((drift_rate * dt_sqrt * args['seed']) * 1e-6)


@pytest.mark.parametrize("args",(
        {'freq': 1, 'noise_scale': 0, 'seed': 0},
        {'freq': 1, 'noise_scale': 1, 'seed': 1},
        {'freq': 100, 'noise_scale': 100, 'seed': 100},
        {'freq': 200, 'noise_scale': 10000, 'seed': 500},
))
def test_measurement_noise(mock_rng, random_truth: GroundTruth, args: dict):

    # The number of iterations to attempt:
    num_iter = 5

    # Initialise the gyroscope with random bias errors
    noise = np.random.randn(3) * args['noise_scale']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(avg_meas_noise=noise)

    # Initialise the gyroscope with random bias errors.
    gyroscope = Gyroscope(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_measurement(
        gyroscope, random_truth, num_iter)

    # Calculate the expected values.
    angle_rates = random_truth.angle_rates
    expected_noise = np.degrees(noise * 1e-6 * args['seed'])
    expected = angle_rates + expected_noise

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)


@pytest.mark.parametrize("args",(
        {'freq': 1, 'scale_scale': 0, 'seed': 0},
        {'freq': 1, 'scale_scale': 1, 'seed': 1},
        {'freq': 100, 'scale_scale': 100, 'seed': 100},
        {'freq': 200, 'scale_scale': 10000, 'seed': 500},
))
def test_scaling_errors(mock_rng, random_truth: GroundTruth, args: dict):

    # The number of iterations to attempt:
    num_iter = 5

    # Initialise the gyroscope with random bias errors
    scale_error = np.random.rand() * args['scale_scale']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(scale_error=scale_error)

    # Initialise the gyroscope with random bias errors.
    gyroscope = Gyroscope(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_measurement(
        gyroscope, random_truth, num_iter)

    # Calculate the expected values.
    angle_rates = random_truth.angle_rates
    expected_error = scale_error * 1e-6 + 1
    expected = angle_rates * expected_error

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)


@pytest.mark.parametrize("args",(
        {'freq': 1, 'non_orth_errors': 0, 'seed': 0},
        {'freq': 1, 'non_orth_errors': 1, 'seed': 1},
        {'freq': 100, 'non_orth_errors': 100, 'seed': 100},
        {'freq': 200, 'non_orth_errors': 10000, 'seed': 500},
))
def test_non_orth_errors(mock_rng, random_truth: GroundTruth, args: dict):

    # The number of iterations to attempt:
    num_iter = 5

    # xy, xz, yx, yz, zx and zy
    # non_orth_errors = np.random.randn(6)
    non_orth_errors = np.random.randn(6) * args['non_orth_errors']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(non_orth_error=non_orth_errors)

    # Initialise the gyroscope with random bias errors.
    gyroscope = Gyroscope(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_measurement(
        gyroscope, random_truth, num_iter)

    # Calculate the expected values.
    angle_rates = random_truth.angle_rates
    tmp_index = ~np.eye(3, dtype=bool)
    tmp_matrix = np.ones(tmp_index.shape)
    tmp_matrix[tmp_index] = non_orth_errors * 1e-6
    expected = tmp_matrix.dot(angle_rates)

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)
