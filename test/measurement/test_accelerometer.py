
import pytest
import numpy as np
from math import sqrt

from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.waypoints.trajectory import GroundTruth
from qnav.util.constants import G

from .conftest import assert_array_equal


def get_accelerometer_measurement(accelerometer: Accelerometer,
                                  ground_truth: GroundTruth,
                                  num_iter: int) -> np.ndarray:
    """
    Obtains a series of measurements from an accelerometer.
    Produces a collection of measurements captured from an accelerometer
    instance, given the same ground truth data with errors updated at
    each time step.

    :param accelerometer: An accelerometer instance to get measurements from.
    :type accelerometer: Accelerometer

    :param ground_truth: A ground truth instance to feed values with.
    :type ground_truth: GroundTruth

    :param num_iter: The number of interactions to get measurements for.
    :type num_iter: int

    :return: An array holding all obtained measurements.
    :rtype: np.ndarray
    """

    measurements = np.zeros((num_iter, 3))

    for i in range(num_iter):
        accelerometer.take_measurement(0, ground_truth)
        measurements[i, :] = accelerometer.last_measurement
        accelerometer.update()

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
    accelerometer = Accelerometer(freq, errors)

    dt = 1 / freq
    expected = dt

    for _ in range(num_iter):
        assert accelerometer.next_update == expected
        accelerometer.update()
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

    # Initialise the accelerometer with random bias errors.
    accelerometer = Accelerometer(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_accelerometer_measurement(
        accelerometer, random_truth, num_iter)

    # Calculate the expected values.
    acceleration = random_truth.acceleration
    expected_bias = bias_error * 1e-6 * G
    expected = acceleration + expected_bias

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

    # Initialise the accelerometer with random bias errors
    bias_error = np.random.randn(3) * args['bias_scale']
    drift_rate = np.random.randn(3) * args['drift_scale']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(
        bias_error = bias_error,
        bias_drift_rate = drift_rate,
    )

    # Initialise the accelerometer with random bias errors.
    accelerometer = Accelerometer(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_accelerometer_measurement(
        accelerometer, random_truth, num_iter)

    # Calculate the expected values.
    acceleration = random_truth.acceleration
    expected = acceleration + bias_error * 1e-6 * G
    dt_sqrt = sqrt(1 / args['freq'])

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)
        expected += dt_sqrt * (drift_rate * 1e-6 * G) * args['seed']


@pytest.mark.parametrize("args",(
        {'freq': 1, 'noise_scale': 0, 'seed': 0},
        {'freq': 1, 'noise_scale': 1, 'seed': 1},
        {'freq': 100, 'noise_scale': 100, 'seed': 100},
        {'freq': 200, 'noise_scale': 10000, 'seed': 500},
))
def test_measurement_noise(mock_rng, random_truth: GroundTruth, args: dict):

    # The number of iterations to attempt:
    num_iter = 5

    # Initialise the accelerometer with random bias errors
    noise = np.random.randn(3) * args['noise_scale']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(avg_meas_noise=noise)

    # Initialise the accelerometer with random bias errors.
    accelerometer = Accelerometer(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_accelerometer_measurement(
        accelerometer, random_truth, num_iter)

    # Calculate the expected values.
    acceleration = random_truth.acceleration
    expected = acceleration + (noise * 1e-6 * G * args['seed'])

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

    # Initialise the accelerometer with random bias errors
    scale_error = np.random.rand() * args['scale_scale']

    # Initialise the error profile with parameters.
    errors = ErrorProperties(scale_error=scale_error)

    # Initialise the accelerometer with random bias errors.
    accelerometer = Accelerometer(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_accelerometer_measurement(
        accelerometer, random_truth, num_iter)

    # Calculate the expected values.
    acceleration = random_truth.acceleration
    expected = acceleration * (scale_error * 1e-6 + 1)

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

    # Initialise the accelerometer with random bias errors.
    accelerometer = Accelerometer(
        args['freq'], errors, rand_seed=args['seed'])

    # Obtain a series of measurements.
    measurements = get_accelerometer_measurement(
        accelerometer, random_truth, num_iter)

    # Calculate the expected values.
    acceleration = random_truth.acceleration
    tmp_index = ~np.eye(3, dtype=bool)
    tmp_matrix = np.ones(tmp_index.shape)
    tmp_matrix[tmp_index] = non_orth_errors * 1e-6
    expected = tmp_matrix.dot(acceleration)

    # Compare expected against each step:
    for i in range(num_iter):
        assert_array_equal(measurements[i], expected)
