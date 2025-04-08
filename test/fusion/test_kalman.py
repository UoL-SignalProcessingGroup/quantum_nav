
import pytest

from fusion.generate import generate_circle, generate_straight_line, generate_static_position
from fusion.test_ins import TestSimpleNumerical, assert_estimation_follows
from qnav.estimation.kalman import generate_matrices
from qnav.estimation.state import KalmanEstimatedState
from qnav.fusion.kalman import KalmanINS
from qnav.gravity.base import GravityModel
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.waypoints.trajectory import GroundTruth


def get_kalman_state(gt: GroundTruth, frequency: float, gravity_model: GravityModel):

    assumed_error = 0.0

    h_matrix, r_matrix, f_matrix, q_matrix = generate_matrices(
        frequency, assumed_error, assumed_error, assumed_error,
        assumed_error, assumed_error, assumed_error)

    return KalmanEstimatedState(gt, h_matrix, r_matrix, f_matrix, q_matrix,
                                gravity_model=gravity_model)


class TestKalmanFilter(TestSimpleNumerical):

    @pytest.fixture
    def perfect_ins(self, frequency: float):
        """
        Returns a perfect clean INS instance.
        """
        error_profile = ErrorProperties()
        accelerometer = Accelerometer(frequency, error_profile)
        gyroscope = Gyroscope(frequency, error_profile)
        return KalmanINS(accelerometer, gyroscope)

    @staticmethod
    @pytest.mark.parametrize('start_pos', [(0.0, 0.0), ])
    @pytest.mark.parametrize('num_inters', [100, ])
    def test_static_position(perfect_ins, frequency, gravity_model,
                             start_pos: tuple[float, float],
                             num_inters: int):

        # Generate the ground truth records:
        ground_truths = generate_static_position(
            start_pos, frequency, num_inters, gravity_model)

        # Initialise the estimated state as the first ground truth
        estimated_state = get_kalman_state(ground_truths[0], frequency, gravity_model)

        # Verify the estimation continues to follow the ground truth until completion
        assert_estimation_follows(perfect_ins, estimated_state, ground_truths)

    @staticmethod
    @pytest.mark.parametrize('start_pos', [(0.0, 0.0), ])
    @pytest.mark.parametrize('angle', [180.0, ])
    @pytest.mark.parametrize('distance', [0.01, 0.1])
    @pytest.mark.parametrize('velocity', [100.0, ])
    def test_straight_line(perfect_ins, frequency, gravity_model,
                           start_pos: tuple[float, float], angle: float,
                           distance: float, velocity: float):

        # Generate the ground truth records:
        ground_truths = generate_straight_line(
            start_pos, frequency, angle, distance, velocity, gravity_model)

        # Initialise the estimated state as the first ground truth
        estimated_state = get_kalman_state(ground_truths[0], frequency, gravity_model)

        # Verify the estimation continues to follow the ground truth until completion
        assert_estimation_follows(perfect_ins, estimated_state, ground_truths)


    @staticmethod
    @pytest.mark.parametrize('centre_pos', [(0.0, 0.0), ])
    @pytest.mark.parametrize('angle', [180.0, ])
    @pytest.mark.parametrize('radius', [100, 1000])
    @pytest.mark.parametrize('velocity', [100.0, ])
    def test_circular_trajectory(perfect_ins, frequency, gravity_model,
                                 centre_pos: tuple[float, float], angle: float,
                                 radius: float, velocity: float):

        # Generate the ground truth records:
        ground_truths = generate_circle(
            centre_pos, frequency, radius, velocity, gravity_model)

        # Initialise the estimated state as the first ground truth
        estimated_state = get_kalman_state(ground_truths[0], frequency, gravity_model)

        # Verify the estimation continues to follow the ground truth until completion
        assert_estimation_follows(perfect_ins, estimated_state, ground_truths)
