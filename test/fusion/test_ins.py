

import pytest

from fusion.generate import generate_straight_line, generate_circle
from fusion.generate import  generate_static_position
from qnav.estimation.state import EstimatedState
from qnav.fusion.ins import NumericalINS
from qnav.gravity.base import GravityModel
from qnav.gravity.simple import FixedValue
from qnav.gravity.somigliana import Somigliana
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.waypoints.trajectory import GroundTruth


def assert_estimation_follows(
        perfect_ins: NumericalINS,
        estimated_state: EstimatedState,
        ground_truths: list[GroundTruth]):

    # For each of the remaining ground truths:
    for gt in ground_truths[1:]:

        # Update each of the sensor instances:
        for sensor in perfect_ins.sensors:
            sensor.take_measurement(0, gt)
            sensor.update()

        # Perform fusion using latest measurements
        perfect_ins.perform_fusion(estimated_state)
        estimated_state.update_estimates(
            timestamp=gt.timestamp)


class TestSimpleNumerical:
    """
    Tests to be conducted for the simple numerical INS solution.
    """

    @pytest.fixture(params=[1.0, 250.0,])
    def frequency(self, request) -> float:
        """
        The shared frequency to be used by sensors and fusion.
        """
        return request.param

    @pytest.fixture(params=(FixedValue, Somigliana))
    def gravity_model(self, request) -> GravityModel:
        """
        The shared gravity models to be used by fusion and ground truth.
        """
        return request.param()

    @pytest.fixture
    def perfect_ins(self, frequency: float):
        """
        Returns a perfect clean INS instance.
        """
        error_profile = ErrorProperties()
        accelerometer = Accelerometer(frequency, error_profile)
        gyroscope = Gyroscope(frequency, error_profile)
        return NumericalINS(accelerometer, gyroscope)

    @staticmethod
    def test_sensors(perfect_ins: NumericalINS):
        """
        Confirms that sensors contain only an accelerometer and gyroscope.
        """
        sensors = perfect_ins.sensors
        sensor_types = [type(s) for s in sensors]
        assert Accelerometer in sensor_types
        assert Gyroscope in sensor_types
        assert len(sensor_types) == 2

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
        estimated_state = EstimatedState(ground_truths[0], gravity_model)

        # Verify the estimation continues to follow the ground truth until completion
        assert_estimation_follows(perfect_ins, estimated_state, ground_truths)

    @staticmethod
    @pytest.mark.parametrize('start_pos', [(0.0, 0.0),])
    @pytest.mark.parametrize('angle', [180.0,])
    @pytest.mark.parametrize('distance', [0.01, 0.1])
    @pytest.mark.parametrize('velocity', [100.0, ])
    def test_straight_line(perfect_ins, frequency, gravity_model,
                           start_pos: tuple[float, float], angle: float,
                           distance: float, velocity: float):

        # Generate the ground truth records:
        ground_truths = generate_straight_line(
            start_pos, frequency, angle, distance, velocity, gravity_model)

        # Initialise the estimated state as the first ground truth
        estimated_state = EstimatedState(ground_truths[0], gravity_model)

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
        estimated_state = EstimatedState(ground_truths[0], gravity_model)

        # Verify the estimation continues to follow the ground truth until completion
        assert_estimation_follows(perfect_ins, estimated_state, ground_truths)






