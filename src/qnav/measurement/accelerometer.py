"""
================
accelerometer.py
================

:summary:
    Contains the virtual accelerometer classes for producing measurements.
    This module contains virtual accelerometer classes that can be used for
    generating accelerometer measurements, from a simulated environment,
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
from qnav.util.transformations import cross_prod_xy
from qnav.util.constants import G
from qnav.waypoints.trajectory import GroundTruth


class Accelerometer(Sensor):
    """
    Provides the functionality of an accelerometer in a simulated environment.
    This class essentially represents a virtual accelerometer converting
    waypoint accelerations to body rate axis and applying noise/errors.
    The output of this can then be passed to a selected INS fusion class.
    """

    # All properties of the class.
    __slots__ = (
        '_bias_error',
        '_bias_drift_rate',
        '_avg_meas_noise_mg',
        '_meas_noise',
        '__error_matrix',
        '__sensor_axis',
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
        Initialises an accelerometer instance with configured errors/rates.
        The defined accelerometer is to operate at given frequency providing
        measurement of quality consistent with the defined error properties.
        Additional properties include a custom sensor axis, start time and
        random number generation seed.

        :param frequency: The frequency of accelerometer measurements (in Hz).
        :type frequency: float

        :param error_profile: Defines error properties for the accelerometer.
        :type error_profile: ErrorProperties

        :param sensor_axis: (Optional) The axis properties for the accelerometer.
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
        self._avg_meas_noise_mg = error_profile.avg_meas_noise * G * 1e-6
        self._meas_noise = self._avg_meas_noise_mg * self._rng.normal(size=3)
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
        Produces an accelerometer measurement from given truth data.
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
        true_acceleration = ground_truth.acceleration

        angle_rate_acc = cross_prod_xy(true_angle_rates, cross_prod_xy(
            true_angle_rates, self.__lever_arm))

        # Convert acceleration to sensor axis
        acceleration_s = self.__rot_body2sensor.dot(
            true_acceleration + angle_rate_acc)

        # Obtain the bias and noise errors
        acc_bias_errors = self._bias_error * G * 1e-6

        # Apply measurement, bias and noise errors
        acceleration_s = self.__error_matrix.dot(
            acceleration_s) + acc_bias_errors + self._meas_noise

        # Record and return last measurement
        self._last_measurement = acceleration_s
        return self._last_measurement

    @property
    def last_measurement(self) -> np.ndarray:
        """
        Returns the latest recorded acceleration produced by sensor.
        The measured acceleration of the accelerometer (North, East
        and Down components in metres/second).

        :return: Latest acceleration measurement produced by sensor.
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

        # Adjust the bias error.
        self._bias_error += \
            self._bias_drift_rate * dt_sqrt * self._rng.normal(size=3)

        # Adjust the measurement noise.
        self._meas_noise = \
            self._avg_meas_noise_mg * self._rng.normal(size=3)

    @property
    def sensor_axis(self) -> SensorAxis:
        """
        Returns the true axis for the accelerometer.

        :return: The sensor axis of the accelerometer.
        :rtype: SensorAxis
        """
        return self.__sensor_axis


