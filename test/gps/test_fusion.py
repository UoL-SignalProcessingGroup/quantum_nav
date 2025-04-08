import math
from datetime import timedelta

import pytest

from qnav.estimation.kalman import generate_matrices
from qnav.util.transformations import ned2lla
from qnav.util.transformations import lla2ned
from qnav.waypoints.trajectory import GroundTruth
from qnav.estimation.state import EstimatedState
from qnav.estimation.state import KalmanEstimatedState
from qnav.gps.fusion import GpsFixedGainFusion
from qnav.gps.fusion import GpsLooseFusion
from qnav.gps.fusion import GpsTightFusion
from qnav.gps.fusion import GpsFusion
from qnav.gps.fusion import calculate_cov_mat
from qnav.gps.sensor import GpsSensor

import numpy as np


def assert_fusion_converges(sensor: GpsSensor, fusion: GpsFusion,
                  estimated_state: EstimatedState,
                  ground_truth: GroundTruth,
                  min_error: float = 1.0):

    true_position = ground_truth.position
    est_position = estimated_state.position

    est_error = lla2ned(est_position, true_position)
    error_dist =  np.linalg.norm(est_error)

    while error_dist > min_error:

        fusion.perform_fusion(estimated_state)
        est_position = estimated_state.position
        est_error = lla2ned(est_position, true_position)
        new_error_dist = np.linalg.norm(est_error)

        assert new_error_dist <= error_dist
        error_dist = new_error_dist
        sensor.update()


def apply_estimate_error(estimated_state: EstimatedState, position_error: float):

    position = estimated_state.position
    error_to_apply = np.ones(3) * position_error

    estimated_state.update_estimates(
        position=ned2lla(error_to_apply, position))

    if isinstance(estimated_state, KalmanEstimatedState):
        state_errors = np.eye(15) * position_error
        estimated_state.update_estimates(
            state_errors=state_errors)

@pytest.mark.parametrize('pos_error', (10, 1000, 100000))
class TestStatic:

    def test_fixed_gain(self, perfect_satellites, random_truth: GroundTruth, pos_error: float):

        # Specify the estimated state
        estimated_state = EstimatedState(random_truth)
        apply_estimate_error(estimated_state, pos_error)

        # Prepare the sensor
        sensor = GpsSensor(1.0, perfect_satellites)
        sensor.take_measurement(0, random_truth)

        # Prepare and test the fusion
        fusion = GpsFixedGainFusion(sensor, estimated_state, 0.1)
        assert_fusion_converges(sensor, fusion, estimated_state, random_truth)


    def test_loosely_coupled(self, perfect_satellites, random_truth: GroundTruth, pos_error: float):

        # Specify the estimated state
        kalman_matrices = generate_matrices(1.0, 1e-1, 1e-1)
        kalman_state = KalmanEstimatedState(random_truth, *kalman_matrices)
        apply_estimate_error(kalman_state, pos_error)

        # Prepare the sensor
        sensor = GpsSensor(1.0, perfect_satellites)
        sensor.take_measurement(0, random_truth)

        # Prepare the fusion
        sigma_pos = math.sqrt(pos_error)
        sigma_vel = 0.1
        cov_matrices = calculate_cov_mat(perfect_satellites, kalman_state.position, sigma_pos, sigma_vel)
        fusion = GpsLooseFusion(sensor, kalman_state, *cov_matrices)

        # Test the fusion
        assert_fusion_converges(sensor, fusion, kalman_state, random_truth)


    def test_tightly_coupled(self, perfect_satellites, random_truth: GroundTruth, pos_error: float):

        # Specify the estimated state
        kalman_matrices = generate_matrices(1.0, 1e-1, 1e-1)
        kalman_state = KalmanEstimatedState(random_truth, *kalman_matrices)
        apply_estimate_error(kalman_state, pos_error)

        # Prepare the sensor
        sensor = GpsSensor(1.0, perfect_satellites)
        sensor.take_measurement(0, random_truth)

        # Prepare the fusion
        state_vector = np.zeros(17)
        state_errors = np.eye(17) * 1e-9
        fusion = GpsTightFusion(sensor, kalman_state, state_vector, state_errors)

        # Test the fusion
        assert_fusion_converges(sensor, fusion, kalman_state, random_truth)


def check_fusion_over_time(sensor: GpsSensor,
                           fusion: GpsFusion, num_steps: int,
                           estimated_state: EstimatedState,
                           ground_truth: GroundTruth,
                           margin: float = 1.0):

    true_position = ground_truth.position
    threshold = float('inf')

    # For each time step:
    for i in range(num_steps):

        # Perform fusion and update estimate
        fusion.perform_fusion(estimated_state)
        est_position = estimated_state.position

        # Calculate the difference in error
        est_error = lla2ned(est_position, true_position)
        error_dist = np.linalg.norm(est_error)

        # Check distance threshold and update
        assert error_dist <= threshold
        threshold = error_dist + margin
        sensor.update()


# @pytest.fixture()
# def specific_truth() -> GroundTruth:
#
#     time_step = 0
#     position = [-34.33765981, 76.15014691, 757.29273136]
#     velocity = [83.86045531, 61.64601787, 9.44875977]
#     acceleration = [62.07250875, 13.77248255, 39.65469839]
#     attitude = [0., 0., 0.]
#     angle_rates = [0., 0., 0.]
#
#     return GroundTruth(
#         time_step, np.array(position), np.array(attitude),
#         np.array(attitude), np.array(attitude), np.array(angle_rates),
#     )


@pytest.mark.parametrize('freq', (1,))
@pytest.mark.parametrize('duration', (timedelta(days=1),))
class TestLong:

    def test_fixed_gain(self, perfect_satellites,
                        random_truth: GroundTruth,
                        duration: timedelta,
                        freq: float):

        # Specify the estimated state
        estimated_state = EstimatedState(random_truth)
        num_steps = int(duration.total_seconds() * freq)

        # Prepare the sensor
        sensor = GpsSensor(freq, perfect_satellites)
        sensor.take_measurement(0, random_truth)

        # Prepare and test the fusion
        fusion = GpsFixedGainFusion(sensor, estimated_state, 0.1)
        check_fusion_over_time(sensor, fusion, num_steps, estimated_state, random_truth)

    def test_loosely_coupled(self, perfect_satellites,
                             random_truth: GroundTruth,
                             duration: timedelta,
                             freq: float):

        # Specify the estimated state
        kalman_matrices = generate_matrices(1 / freq, 1e-1, 1e-1)
        kalman_state = KalmanEstimatedState(random_truth, *kalman_matrices)
        num_steps = int(duration.total_seconds() * freq)

        # Prepare the sensor
        sensor = GpsSensor(freq, perfect_satellites)
        sensor.take_measurement(0, random_truth)

        # Prepare the fusion
        sigma_pos = 0.1
        sigma_vel = 0.1
        cov_matrices = calculate_cov_mat(perfect_satellites, kalman_state.position, sigma_pos, sigma_vel)

        # Test the fusion
        fusion = GpsLooseFusion(sensor, kalman_state, *cov_matrices)
        check_fusion_over_time(sensor, fusion, num_steps, kalman_state, random_truth)

    def test_tightly_coupled(self, perfect_satellites,
                             random_truth: GroundTruth,
                             duration: timedelta,
                             freq: float):

        # Specify the estimated state
        kalman_matrices = generate_matrices(1 / freq, 1e-1, 1e-1)
        kalman_state = KalmanEstimatedState(random_truth, *kalman_matrices)
        num_steps = int(duration.total_seconds() * freq)

        # Prepare the sensor
        sensor = GpsSensor(freq, perfect_satellites)
        sensor.take_measurement(0, random_truth)

        # Prepare the fusion
        state_vector = np.zeros(17)
        state_errors = np.eye(17) * 1e-9

        fusion = GpsTightFusion(sensor, kalman_state, state_vector, state_errors)
        check_fusion_over_time(sensor, fusion, num_steps, kalman_state, random_truth)

