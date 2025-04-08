"""
================
dummy_sensors.py
================

:summary:
    Presents dummy sensors used for re-applying fusion algorithms.
    This module contains dummy sensor models used for re-applying previous
    fusion algorithms with corrected measurements. These dummy sensors allow
    internal states to be exposed and edited. These do not feature involvement
    of real ground truth data, errors states or "real" measurements.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""


from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.sensor import Sensor
from abc import ABC


class DummySensor(Sensor, ABC):
    """
    Abstract class for fake sensor to handle synthetic measurements.
    Provides an abstract class for dummy sensor, which allows its
    measurements to be fabricated. This is solely used for solutions
    that require the re-use of fusion methods with updates/corrections
    applied to previous measurements.
    """

    # The default last measurement
    _last_measurement = None

    @property
    def last_measurement(self) -> any:
        """
        Returns the last measurement produced by the sensor.

        :return: Last measurement value produced.
        :rtype: any
        """
        return self._last_measurement

    @last_measurement.setter
    def last_measurement(self, value: any):
        """
        Set the last measurement produced by the sensor.

        :param value: The values to use as the last measurement.
        :type value: any
        """
        self._last_measurement = value

    @property
    def last_update(self) -> float:
        """
        The time of the last measurement (in seconds).

        :return: Time of last measurement in seconds.
        :rtype: float
        """
        return self._time_elapsed

    @last_update.setter
    def last_update(self, value: float):
        """
        Set the time of the last measurement (in seconds).

        :param value: Time of the last measurement in seconds.
        :type value: float
        """
        self._time_elapsed = value

    @property
    def next_update(self) -> float:
        """
        The time of the next measurement (in seconds).

        :return: Time of next measurement in seconds.
        :rtype: float
        """
        return self._next_update

    @next_update.setter
    def next_update(self, value: float):
        """
        Set the time of the last measurement (in seconds).

        :param value: Time of the last measurement in seconds.
        :type value: float
        """
        self._next_update = value


class DummyAccelerometer(DummySensor, Accelerometer):
    """
    Dummy Accelerometer sensor used to mimic accelerometer measurements.
    This class is a dummy sensor that inherits functionalities for the
    Accelerometer class. Essentially supports synthetic accelerometer
    measurements to be used in fusion methods.
    """

    def __init__(self, accelerometer: Accelerometer):
        """
        Constructs dummy accelerometer for real accelerometer configuration.
        Given a real accelerometer, this constructs a virtual version,
        copying the given accelerometer's configuration (frequency,
        sensor axis, and last measurement values and time)

        :param accelerometer: The accelerometer to copy values from.
        :type accelerometer: Accelerometer
        """
        super().__init__(accelerometer.frequency, ErrorProperties(), accelerometer.sensor_axis)
        self._last_measurement = accelerometer.last_measurement
        self._time_elapsed = accelerometer.last_update
        self._next_update = accelerometer.next_update  # TODO: NEW!


class DummyGyroscope(DummySensor, Gyroscope):
    """
    Dummy Gyroscope sensor used to mimic gyroscope measurements.
    This class is a dummy sensor that inherits functionalities for the
    Gyroscope class. Essentially supports synthetic gyroscope measurements
    to be used in fusion methods.
    """

    def __init__(self, gyroscope: Gyroscope):
        """
        Constructs dummy gyroscope for real gyroscope configuration.
        Given a real gyroscope, this constructs a virtual version, copying
        the given gyroscope's configuration (frequency, sensor axis,
         and last measurement values and time)

        :param gyroscope: The gyroscope to copy values from.
        :type gyroscope: Gyroscope
        """
        super().__init__(gyroscope.frequency, ErrorProperties(), gyroscope.sensor_axis)
        self._last_measurement = gyroscope.last_measurement
        self._time_elapsed = gyroscope.last_update
        self._next_update = gyroscope.next_update  # TODO: NEW!
