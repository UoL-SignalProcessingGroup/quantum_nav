"""
============
altimeter.py
============

:summary:
    Contains the virtual altimeter classes for producing measurements.
    This module contains virtual altimeter classes that can be used for
    generating altimeter measurements, from a simulated environment,
    with a configured level of noise/errors.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

from math import sqrt

from qnav.measurement.sensor import Sensor
from qnav.waypoints.trajectory import GroundTruth


# class StandardAltimeter:
#     """
#     Provides the functionality of an altimeter in a simulated environment.
#     This class essentially represents a virtual altimeter that takes the
#     altitude from the waypoint data and adds a configured noise/error.
#     The output of this can then be passed to a selected INS solution.
#     """
#
#     # All properties of the class.
#     __slots__ = \
#         '__alt_bias_error', \
#         '__alt_bias_drift_rate', \
#         '__frequency', \
#         '__is_enabled', \
#         '__rng'
#
#     def __init__(self, freq: float,
#                  alt_bias_error: float = 0,
#                  alt_bias_drift_rate: float = 1,
#                  is_enabled: bool = True,
#                  rand_seed: int = None):
#         """
#         Initialises an altimeter instance with configured errors/rates.
#
#         :param freq: The frequency of altimeter measurements (in Hz).
#         :type freq: float
#
#         :param alt_bias_error: The initial altimeter bias error (in metres
#             above sea level).
#         :type alt_bias_error: float
#
#         :param alt_bias_drift_rate: The altimeter bias drift rate (in
#             metres/root sec).
#         :type alt_bias_drift_rate: float
#
#         :param rand_seed: The optional initialisation seed to use to
#             generate random numbers for noise and other errors.
#         :type rand_seed: int
#         """
#
#         assert freq > 0, \
#             'Altimeter measurement frequency must be a positive value!'
#
#         self.__frequency = freq
#         self.__is_enabled = is_enabled
#         self.__alt_bias_error = alt_bias_error
#         self.__alt_bias_drift_rate = alt_bias_drift_rate
#
#         # Initialise the random number generator
#         self.__rng = np.random.default_rng(rand_seed)
#
#     def update(self, dt: float):
#         """
#         Called every iteration to update simulated sensor errors.
#         As sensor errors propagate with time, it is important that this
#         method is called prior to every measurement update.
#
#         :param dt: Time difference since the last update.
#         :type dt: float
#         """
#         self.__alt_bias_error += \
#             self.__alt_bias_drift_rate * sqrt(dt) * self.__rng.normal()
#
#     def get_measurement(self, true_altitude: float) -> float:
#         """
#         Produces an altimeter measurement from given truth data.
#         This method takes the true altitude (above sea level) and
#         applies the configured errors/noise. The output can then be
#         fed to the selected INS solution.
#
#         :param true_altitude: The true altitude of the sensor (in metres
#             above sea level).
#         :type true_altitude: float
#
#         :return: The measured altitude of the altimeter (in metres above
#             sea level).
#         :rtype: float
#         """
#         return true_altitude + self.__alt_bias_error
#
#     def get_update_freq(self) -> float:
#         """
#         Returns the altimeter measurement frequency in Hz.
#
#         :return: The measurement frequency in Hz.
#         :rtype: float
#         """
#         return self.__frequency
#
#     def is_enabled(self) -> bool:
#         """
#         Returns if the altimeter is active for the simulation.
#         This method returns True if the altimeter instance is activated.
#
#         :return: True if the altimeter instance is active.
#         :rtype: bool
#         """
#         return self.__is_enabled


class Altimeter(Sensor):
    """
    Provides the functionality of an altimeter in a simulated environment.
    This class essentially represents a virtual altimeter that takes the
    altitude from the waypoint data and adds a configured noise/error.
    The output of this can then be passed to a selected INS solution.
    """

    # All properties of the class.
    __slots__ = '__bias_error', '__bias_drift_rate', '_time_step_sqrt'

    def __init__(self, frequency: float,
                 bias_error: float = 0.0,
                 bias_drift_rate: float = 0.0,
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Initialises an altimeter instance with configured errors/rates.
        The defined altimeter is to operate at given frequency providing
        measurement of quality consistent with the defined error properties.
        Additional properties include a custom sensor axis, start time and
        random number generation seed.

        :param frequency: The exact measurement frequency of the sensor in Hz.
        :type frequency: float

        :param bias_error: The bias error of the altimeter (in metres)
        :type bias_error: float

        :param bias_drift_rate: The bias drift rate of the altimeter
            (in metres/root sec)
        :type bias_drift_rate: float

        :param start_time: The starting time of the sensor in seconds.
        :type start_time: float

        :param rand_seed: The random number generation seed.
        :type rand_seed: int
        """
        super().__init__(frequency, start_time, rand_seed)
        self.__bias_error = bias_error
        self.__bias_drift_rate = bias_drift_rate
        self._time_step_sqrt = sqrt(self._time_step)
        self._last_measurement = None

    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        """
        Produces an altimeter measurement from given truth data.
        This method takes the true altitude (above sea level) and
        applies the configured errors/noise. The output can then be
        fed to the selected INS solution.

        :param est_time: The estimated time the measurement of the sensor
            is expected to be captured at (in seconds).
        :type est_time: float

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth
        """
        true_altitude = ground_truth.position[2]
        self._last_measurement = true_altitude + self.__bias_error

    @property
    def last_measurement(self) -> float:
        """
        Returns the latest recorded altitude produced by the altimeter.

        :return: Latest altitude measurement produced.
        :rtype: float
        """
        return self._last_measurement

    def update(self, time_sec: float = None):
        """
        Called on demand to update the altimeter's state.
        Typically called after measurements have been taken. When called
        this increases the next measurement time and propagates internal
        states, to the time difference.

        :param time_sec: The current simulation time in seconds.
        :type time_sec: float
        """

        # Substitute None value for default time difference
        if time_sec is None:
            dt = self._time_step
            dt_sqrt = self._time_step_sqrt
        else:
            dt = (time_sec - self._time_elapsed)
            dt_sqrt = sqrt(dt)

        # Increase the time elapsed.
        self._time_elapsed += dt
        self._next_update = self._time_elapsed + self._time_step

        self.__bias_error += \
            self.__bias_drift_rate * dt_sqrt * self._rng.normal()
