from copy import deepcopy

import pytest
import numpy as np

from qnav.measurement.error_properties import ErrorProperties
from qnav.quantum.base import ConceptQuantumImu
from qnav.quantum.realistic import ColdAtomInterferometer
from qnav.quantum.realistic import QuantumIMU


def _assert_static_measurements(sensor, ground_truth):

    assert sensor.last_measurement is None, \
        'Last measurement should be initially None'

    total_steps = sensor.num_steps
    for i in range(total_steps):
        sensor.take_measurement(0, ground_truth)
        sensor.update()

        if i != total_steps - 1:
            assert sensor.last_measurement is None, \
                f'Last measurement should be None at step {i}'

    measurement = sensor.last_measurement
    assert measurement is not None, \
        'Measurement should not be none after final step'


class TestBasicSensor:

    @pytest.fixture(scope='class')
    def precision(self) -> int:
        return 8

    @pytest.fixture(scope='class', params=[1, 10, 20])
    def full_frequency(self, request) -> float:
        return request.param

    @pytest.fixture(scope='class', params=[20, 50, 100])
    def imu_frequency(self, request) -> float:
        return request.param

    @pytest.fixture(scope='class', params=[0.5, 0.75, 1.0])
    def duty_cycle(self, request) -> float:
        return request.param

    @pytest.fixture
    def sensor(self, full_frequency, imu_frequency, duty_cycle):
        return ConceptQuantumImu(
            full_frequency, imu_frequency,
            ErrorProperties(), ErrorProperties(),
            duty_cycle=duty_cycle)

    @staticmethod
    def test_num_steps(sensor, full_frequency, imu_frequency):
        ratio = imu_frequency / full_frequency
        assert sensor.num_steps == int(max(ratio, 1))

    @staticmethod
    def test_active_steps(sensor, full_frequency, imu_frequency, duty_cycle):
        ratio = (imu_frequency / full_frequency) * duty_cycle
        assert sensor.num_active_steps == int(max(ratio, 1))

    @staticmethod
    def test_take_measurement(sensor, random_truth, precision):

        _assert_static_measurements(sensor, random_truth)

        expected_acceleration = random_truth.acceleration
        expected_angle_rates = random_truth.angle_rates

        measured_acceleration, measured_angle_rates = sensor.last_measurement
        np.testing.assert_almost_equal(measured_acceleration, expected_acceleration, precision)
        np.testing.assert_almost_equal(measured_angle_rates, expected_angle_rates, precision)

    @staticmethod
    def test_reset_measurement(sensor, random_truth, empty_truth, precision):

        _assert_static_measurements(sensor, random_truth)
        sensor.reset()
        _assert_static_measurements(sensor, empty_truth)

        expected_acceleration = empty_truth.acceleration
        expected_angle_rates = empty_truth.angle_rates

        measured_acceleration, measured_angle_rates = sensor.last_measurement
        np.testing.assert_almost_equal(measured_acceleration, expected_acceleration, precision)
        np.testing.assert_almost_equal(measured_angle_rates, expected_angle_rates, precision)

    @staticmethod
    @pytest.mark.parametrize('num_steps', [1, 10, 10000])
    def test_update_sensor(sensor, random_truth, num_steps):

        qs_acc = sensor._qs_accelerometer
        qs_gyro = sensor._qs_gyroscope

        clone_acc = deepcopy(qs_acc)
        clone_gyro = deepcopy(qs_gyro)

        assert qs_acc != clone_acc, 'Failed to clone accelerometer'
        assert qs_gyro != clone_gyro, 'Failed to clone gyroscope'

        for _ in range(num_steps):

            sensor.take_measurement(0, random_truth)
            sensor.update()

            clone_acc.take_measurement(0, random_truth)
            clone_acc.update()

            clone_gyro.take_measurement(0, random_truth)
            clone_gyro.update()

            assert np.all(qs_acc.last_measurement == clone_acc.last_measurement), \
                'Accelerometer measurements do not match'

            assert qs_acc.last_update == clone_acc.last_update, \
                'Accelerometer last update times do not match'

            assert qs_acc.next_update == clone_acc.next_update, \
                'Accelerometer next update times do not match'

            assert np.all(qs_gyro.last_measurement == clone_gyro.last_measurement), \
                'Gyroscope measurements do not match'

            assert qs_gyro.last_update == clone_gyro.last_update, \
                'Gyroscope last update times do not match'

            assert qs_gyro.next_update == clone_gyro.next_update, \
                'Gyroscope next update times do not match'

            if sensor.last_update is not None:
                sensor.reset()



class TestRealisticSensor(TestBasicSensor):

    @pytest.fixture(scope='class')
    def precision(self) -> int:
        return 3

    @pytest.fixture
    def sensor(self, full_frequency, imu_frequency, duty_cycle):

        return QuantumIMU(
            full_frequency, imu_frequency,
            ErrorProperties(), ErrorProperties(),
            ColdAtomInterferometer(beam_width = 0.25),
            duty_cycle=duty_cycle)

    @staticmethod
    def test_take_measurement(sensor, random_truth, precision):
        _assert_static_measurements(sensor, random_truth)

        expected_acceleration = random_truth.acceleration
        expected_angle_rates = random_truth.angle_rates

        measured_acceleration, measured_angle_rates = sensor.last_measurement

        nan_acceleration = np.isnan(measured_acceleration)
        if any(nan_acceleration) and not all(nan_acceleration):
            pytest.skip('Acceleration too violent for valid reading')

        nan_angle_rate = np.isnan(measured_angle_rates)
        if any(nan_angle_rate) and not all(nan_angle_rate):
            pytest.skip('Angle rates too violent for valid reading')

        np.testing.assert_almost_equal(measured_acceleration, expected_acceleration, precision)
        np.testing.assert_almost_equal(measured_angle_rates, expected_angle_rates, precision)

    @staticmethod
    def test_reset_measurement(sensor, random_truth, empty_truth, precision):

        _assert_static_measurements(sensor, random_truth)
        sensor.reset()
        _assert_static_measurements(sensor, empty_truth)

        expected_acceleration = empty_truth.acceleration
        expected_angle_rates = empty_truth.angle_rates

        measured_acceleration, measured_angle_rates = sensor.last_measurement

        nan_acceleration = np.isnan(measured_acceleration)
        if any(nan_acceleration) and not all(nan_acceleration):
            pytest.skip('Acceleration too violent for valid reading')

        nan_angle_rate = np.isnan(measured_angle_rates)
        if any(nan_angle_rate) and not all(nan_angle_rate):
            pytest.skip('Angle rates too violent for valid reading')

        np.testing.assert_almost_equal(measured_acceleration, expected_acceleration, precision)
        np.testing.assert_almost_equal(measured_angle_rates, expected_angle_rates, precision)
