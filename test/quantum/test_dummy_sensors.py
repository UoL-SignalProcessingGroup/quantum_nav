
import pytest
import numpy as np

from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.quantum.dummy import DummyAccelerometer
from qnav.quantum.dummy import DummyGyroscope
from random import random


def _assert_properties_match(sensor, dummy):

    assert dummy.frequency == sensor.frequency, \
        'Sensor frequency should be the same'

    assert dummy.time_step == sensor.time_step, \
        'Sensor time step should be the same'

    assert np.all(dummy.last_measurement == sensor.last_measurement), \
        'Sensor last measurement should be the same'

    assert dummy.last_update == sensor.last_update, \
        'Sensor last update should be the same'

    assert dummy.next_update == sensor.next_update, \
        'Sensor next update should be the same'

    assert dummy.sensor_axis == sensor.sensor_axis, \
        'Sensor axes should be the same'


def _assert_measurement_changed(sensor, dummy):

    assert np.all(dummy.last_measurement != sensor.last_measurement), \
        'Sensor last measurement should not be the same'

    assert dummy.last_update != sensor.last_update, \
        'Sensor last update should not  be the same'

    assert dummy.next_update != sensor.next_update, \
        'Sensor next update should be the same'


class TestDummyAccelerometer:

    @pytest.fixture
    def frequency(self) -> float:
        """
        Returns the frequency to use for the sensor.
        """
        return 100

    @pytest.fixture
    def true_sensor(self, frequency):
        """
        Returns fresh true sensor instance.
        """
        return Accelerometer(frequency, ErrorProperties())

    @pytest.fixture
    def dummy_class(self, true_sensor):
        """
        Returns the class of the dummy sensor.
        """
        return DummyAccelerometer

    @staticmethod
    def test_conversion(true_sensor, dummy_class):
        dummy_sensor = dummy_class(true_sensor)
        _assert_properties_match(true_sensor, dummy_sensor)

    @staticmethod
    def test_separation(true_sensor, dummy_class, random_truth):
        dummy_sensor = dummy_class(true_sensor)
        true_sensor.take_measurement(0, random_truth)
        true_sensor.update()
        _assert_measurement_changed(true_sensor, dummy_sensor)

    @staticmethod
    def test_last_measurement(true_sensor, dummy_class, random_truth):

        true_sensor.take_measurement(0, random_truth)
        dummy_sensor = dummy_class(true_sensor)

        sensor_measurement = true_sensor.last_measurement
        assert np.all(dummy_sensor.last_measurement == sensor_measurement)
        dummy_measurement = -dummy_sensor.last_measurement
        dummy_sensor.last_measurement = dummy_measurement

        assert np.all(dummy_sensor.last_measurement != sensor_measurement)
        assert np.all(dummy_sensor.last_measurement == dummy_measurement)

    @staticmethod
    def test_last_update(true_sensor, dummy_class, random_truth):

        true_sensor.take_measurement(0, random_truth)
        dummy_sensor = dummy_class(true_sensor)

        sensor_last_update = true_sensor.last_update
        assert dummy_sensor.last_update == sensor_last_update
        dummy_last_update = 10 * random()
        dummy_sensor.last_update = dummy_last_update

        assert np.all(dummy_sensor.last_update != sensor_last_update)
        assert np.all(dummy_sensor.last_update == dummy_last_update)

    @staticmethod
    def test_next_update(true_sensor, dummy_class, random_truth):
        true_sensor.take_measurement(0, random_truth)
        dummy_sensor = dummy_class(true_sensor)

        sensor_next_update = true_sensor.next_update
        assert dummy_sensor.next_update == sensor_next_update
        dummy_next_update = 10 * random()
        dummy_sensor.next_update = dummy_next_update

        assert np.all(dummy_sensor.next_update != sensor_next_update)
        assert np.all(dummy_sensor.next_update == dummy_next_update)

    @staticmethod
    @pytest.mark.parametrize('change_measurement', [False, True])
    @pytest.mark.parametrize('change_last_update', [False, True])
    @pytest.mark.parametrize('change_next_update', [False, True])
    def test_property_change(true_sensor, dummy_class, random_truth,
                             change_measurement, change_last_update,
                             change_next_update):

        true_sensor.take_measurement(0, random_truth)
        true_sensor.update()

        expected_measurement = true_sensor.last_measurement
        expected_last_update = true_sensor.last_update
        expected_next_update = true_sensor.next_update

        dummy_sensor = dummy_class(true_sensor)

        if change_measurement:
            expected_measurement = true_sensor.last_measurement * random()
            dummy_sensor.last_measurement = expected_measurement

        if change_last_update:
            expected_last_update = random() * 10
            dummy_sensor.last_update = expected_last_update

        if change_next_update:
            expected_next_update = random() * 10
            dummy_sensor.next_update = expected_next_update

        assert np.all(dummy_sensor.last_measurement == expected_measurement), \
            'Mismatch in expected last measurement'

        assert dummy_sensor.last_update == expected_last_update, \
            'Mismatch in expected last update'

        assert dummy_sensor.next_update == expected_next_update, \
            'Mismatch in expected next update'




class TestDummyGyroscope(TestDummyAccelerometer):

    @pytest.fixture
    def true_sensor(self, frequency):
        """
        Returns fresh true sensor instance.
        """
        return Gyroscope(frequency, ErrorProperties())

    @pytest.fixture
    def dummy_class(self, true_sensor):
        """
        Returns the class of the dummy sensor.
        """
        return DummyGyroscope
