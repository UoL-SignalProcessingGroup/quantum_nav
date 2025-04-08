"""
============
gyroscope.py
============

:summary:
    Contains the virtual gyroscope classes for producing measurements.
    This module contains virtual gyroscope classes that can be used for
    generating gyroscope measurements, from a simulated environment,
    with a configured level of noise/errors.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import math
import numpy as np

from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.platform import SensorAxis
from qnav.measurement.sensor import Sensor
from qnav.waypoints.trajectory import GroundTruth


class Gyroscope(Sensor):
    """
    Provides the functionality of a gyroscope in a simulated environment.
    This class essentially represents a virtual gyroscope converting
    waypoint angle rates to body rate axis and applying noise/errors.
    The output of this can then be passed to a selected INS solution.
    """

    # All properties of the class.
    __slots__ = (
        '_bias_error',
        '_bias_drift_rate',
        '_avg_meas_noise_md',
        '_meas_noise',
        '__sensor_axis',
        '__error_matrix',
        '__lever_arm',
        '__rot_body2sensor',
        '_time_step_sqrt'
    )

    def __init__(self,
                 frequency: float,
                 error_profile: ErrorProperties,
                 sensor_axis: SensorAxis = SensorAxis(),
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Initialises a gyroscope instance with configured errors/rates.
        The defined gyroscope is to operate at given frequency providing
        measurement of quality consistent with the defined error properties.
        Additional properties include a custom sensor axis, start time and
        random number generation seed.

         :param frequency: The frequency of gyroscope measurements (in Hz).
        :type frequency: float

        :param error_profile: Defines error properties for the gyroscope.
        :type error_profile: ErrorProperties

        :param sensor_axis: (Optional) The axis properties for the gyroscope.
        :type sensor_axis: SensorAxis

        :param start_time: (Optional) The starting time of the sensor in seconds.
        :type start_time: float

        :param rand_seed: (Optional) The random number generation seed.
        :type rand_seed: int
        """

        # Call base class constructor
        super().__init__(frequency, start_time, rand_seed)

        # Unpack required error properties:
        self._bias_error = error_profile.bias_error
        self._bias_drift_rate = error_profile.bias_drift_rate
        self._avg_meas_noise_md = np.degrees(error_profile.avg_meas_noise) * 1e-6
        self._meas_noise = self._avg_meas_noise_md * self._rng.normal(size=3)
        self.__error_matrix = error_profile.error_matrix

        # Unpack required sensor axis properties:
        self.__sensor_axis = sensor_axis
        self.__lever_arm = sensor_axis.lever_arm
        self.__rot_body2sensor = sensor_axis.body2sensor_mat

        # Define the last measurement
        self._last_measurement = None
        self._time_step_sqrt = math.sqrt(self._time_step)


    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        """
        Produces a gyroscope measurement from given truth data.
        This method converts the given waypoint data to body axes and
        applies the configured errors/noise. The output can then be
        fed to the selected INS solution.

        :param est_time: The estimated time the measurement of the sensor
            is expected to be captured at (in seconds).
        :type est_time: float

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth
        """

        # Unpack the ground truth
        true_angle_rates = ground_truth.angle_rates
        angle_rates_rad = np.radians(true_angle_rates)

        # Convert angle rates to sensor axis.
        angle_rate_s = self.__rot_body2sensor.dot(angle_rates_rad)

        # TODO: Confirm change
        # Convert bias errors to micro-rad/sec.root sec units.
        # gyro_bias_errors = np.degrees(self._bias_error) * 1e-6
        gyro_bias_errors = self._bias_error * 1e-6

        angle_rate_s = self.__error_matrix.dot(
            angle_rate_s) + gyro_bias_errors + self._meas_noise

        self._last_measurement = np.degrees(angle_rate_s)

    @property
    def last_measurement(self) -> np.ndarray:
        """
        Returns the latest recorded gyroscope produced by sensor.
        The measured angle rates of the gyroscope (P, Q and R
        angles in degrees/second).

        :return: Latest angle rates measurement produced by sensor.
        :rtype: np.ndarray (3-elements)
        """
        return self._last_measurement

    def update(self, time_sec: float = None):
        """
        Called every iteration to update simulated sensor errors.
        As sensor errors propagate with time, it is important that this
        method is called prior to every measurement update.

        :param time_sec: The current simulation time in seconds.
        :type time_sec: float
        """

        # Substitute None value for default time difference
        if time_sec is None:
            dt = self._time_step
            dt_sqrt = self._time_step_sqrt
        else:
            dt = (time_sec - self._time_elapsed)
            dt_sqrt = math.sqrt(dt)

        # Increase the time elapsed.
        self._time_elapsed += dt
        self._next_update = self._time_elapsed + self._time_step

        self._bias_error += \
            self._bias_drift_rate * dt_sqrt * self._rng.normal(size=3)

        self._meas_noise = \
            self._avg_meas_noise_md * self._rng.normal(size=3)


    @property
    def sensor_axis(self) -> SensorAxis:
        """
        Returns the true axis for the gyroscope.

        :return: The sensor axis of the gyroscope.
        :rtype: SensorAxis
        """
        return self.__sensor_axis
