

import pytest
import numpy as np

from qnav.estimation.kalman import generate_matrices
from qnav.estimation.state import KalmanEstimatedState
from qnav.waypoints.trajectory import GroundTruth

# The names of estimated state properties to check:
_PROP_NAMES = ['timestamp', 'position', 'velocity',
               'acceleration', 'attitude', 'angle_rates']


def make_clean_kalman_state(gt: GroundTruth, freq: float = 1.0,
                            state_errors: np.ndarray = np.eye(15) * 1e-9) -> KalmanEstimatedState:
    """
    Generates a clean kalman filter state with zero errors.
    """
    assumed_error = 0.0
    h_matrix, r_matrix, f_matrix, q_matrix = generate_matrices(
            freq, assumed_error, assumed_error, assumed_error,
            assumed_error, assumed_error, assumed_error)

    return KalmanEstimatedState(gt, h_matrix, r_matrix, f_matrix, q_matrix, state_errors)


def test_kalman_values(random_truth: GroundTruth):
    """
    Verifies that the correct properties are copied from the ground truth.
    """

    freq = 1.0
    errors = np.eye(15)
    est = make_clean_kalman_state(random_truth, freq, errors)
    np.testing.assert_array_equal(est.timestamp, random_truth.timestamp)
    np.testing.assert_array_equal(est.position, random_truth.position)
    np.testing.assert_array_equal(est.velocity, random_truth.velocity)
    np.testing.assert_array_equal(est.acceleration, random_truth.acceleration)
    np.testing.assert_array_equal(est.attitude, random_truth.attitude)
    np.testing.assert_array_equal(est.angle_rates, random_truth.angle_rates)
    np.testing.assert_array_equal(est.state_errors, errors)


def test_kalman_state_vector(random_truth: GroundTruth):
    """
    Verifies that the state vector representation is correct.
    """

    est = make_clean_kalman_state(random_truth)
    state_vector = est.state_vector

    np.testing.assert_array_equal(
        state_vector[[0, 3, 6]],
        random_truth.position)

    np.testing.assert_array_equal(
        state_vector[[1, 4, 7]],
        random_truth.velocity)

    np.testing.assert_array_equal(
        state_vector[[2, 5, 8]],
        random_truth.acceleration)

    np.testing.assert_array_equal(
        state_vector[[9, 11, 13]],
        random_truth.attitude)

    np.testing.assert_array_equal(
        state_vector[[14, 12, 10]],
        random_truth.angle_rates)


def test_kalman_state_update(random_truth: GroundTruth, another_truth: GroundTruth):
    """
    Verifies that the state vector representation is correct.
    """

    est = make_clean_kalman_state(random_truth)
    est.update_estimates(**another_truth.as_dict())
    state_vector = est.state_vector

    np.testing.assert_array_equal(
        state_vector[[0, 3, 6]],
        another_truth.position)

    np.testing.assert_array_equal(
        state_vector[[1, 4, 7]],
        another_truth.velocity)

    np.testing.assert_array_equal(
        state_vector[[2, 5, 8]],
        another_truth.acceleration)

    np.testing.assert_array_equal(
        state_vector[[9, 11, 13]],
        another_truth.attitude)

    np.testing.assert_array_equal(
        state_vector[[14, 12, 10]],
        another_truth.angle_rates)


@pytest.mark.parametrize('update_time', (False, True))
@pytest.mark.parametrize('update_pos', (False, True))
@pytest.mark.parametrize('update_vel', (False, True))
@pytest.mark.parametrize('update_acc', (False, True))
@pytest.mark.parametrize('update_att', (False, True))
@pytest.mark.parametrize('update_ang', (False, True))
@pytest.mark.parametrize('update_errors', (False, True))
def test_kalman_updating(random_truth: GroundTruth,
                        another_truth: GroundTruth,
                        update_time: bool, update_pos: bool, update_vel: bool, update_acc: bool,
                         update_att: bool, update_ang: bool, update_errors: bool):

    to_update = {
        'timestamp': update_time,
        'position': update_pos,
        'velocity': update_vel,
        'acceleration': update_acc,
        'attitude': update_ang,
        'angle_rates': update_ang,
    }

    expected = {}
    updated_state = {}

    for prop_name, has_update in to_update.items():
        if has_update:
            updated_state[prop_name] = getattr(another_truth, prop_name)
            expected[prop_name] = updated_state[prop_name]
        else:
            expected[prop_name] = getattr(random_truth, prop_name)

    freq = 1.0
    state_errors = np.eye(15) * 1e-9
    est = make_clean_kalman_state(random_truth, freq, state_errors)

    # Append test for state errors (not contained in ground truth)
    if update_errors:
        updated_state['state_errors'] = np.random.rand(15)
        expected['state_errors'] = updated_state['state_errors']
    else:
        expected['state_errors'] = state_errors

    est.update_estimates(**updated_state)
    for prop_name, prop_value in expected.items():
        np.testing.assert_array_equal(getattr(est, prop_name), prop_value)


def test_cloning(random_truth: GroundTruth, another_truth: GroundTruth):

    # Create an estimate and make a clone of it
    est_original = make_clean_kalman_state(random_truth)
    est_clone = est_original.clone()

    # Make changes to the clone
    update = another_truth.as_dict()
    est_clone.update_estimates(**update)

    # Check properties of original have not been affected:
    for prop_name in _PROP_NAMES:

        # Ensure properties are not the same
        assert np.all(getattr(est_original, prop_name) !=
                      getattr(est_clone, prop_name))


def test_to_dict(random_truth: GroundTruth):

    est = make_clean_kalman_state(random_truth)
    est_dict = est.as_dict()

    for prop_name in _PROP_NAMES:
        np.testing.assert_array_equal(
            getattr(est, prop_name), est_dict[prop_name])


def test_to_numpy(random_truth: GroundTruth):

    est = make_clean_kalman_state(random_truth)
    est_numpy = est.as_numpy()

    props = ([est.timestamp], est.position, est.velocity,
                est.acceleration, est.attitude, est.angle_rates)

    expected = np.concatenate(props, axis=0)
    np.testing.assert_array_equal(expected, est_numpy)


def test_prevent_overwrite(random_truth: GroundTruth, another_truth: GroundTruth):

    est = make_clean_kalman_state(random_truth)

    for prop_name in _PROP_NAMES:
        x = getattr(est, prop_name)
        x += getattr(another_truth, prop_name)

    for prop_name in _PROP_NAMES:
        prop = getattr(est, prop_name)
        expected = getattr(random_truth, prop_name)
        np.testing.assert_array_equal(prop, expected)
