"""
========
clock.py
========

:summary:
    Contains the virtual clock/timer classes for estimating elapsed time.
    This module contains virtual clock classes that can be used for simulating
    recorded time (with errors and drift) within a simulated environment. This
    is namely to simulate clock errors within simulations.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import numpy as np
from math import sqrt


# class StandardClock:
#     """
#     Provides the functionality of a system clock in a simulated environment.
#     This class essentially represents a virtual clock that simulates the
#     recording of elapsed time, configured noise/error. The output of this
#     can then be passed to a selected INS solution.
#     """
#
#     # All properties of the class.
#     __slots__ = \
#         '__bias_value', \
#         '__bias_drift_rate', \
#         '__rng'
#
#     def __init__(self, clock_bias: float = 0, bias_drift_rate: float = 0, rand_seed: int = None):
#         """
#         Initialises a clock instance with configured errors/rates.
#
#         :param clock_bias: The initial clock bias (in nano-sec).
#         :type clock_bias: float
#
#         :param bias_drift_rate: The clock error drift rate (in
#             nano-sec/root sec)
#         :type bias_drift_rate: float
#         """
#         self.__bias_value = clock_bias
#         self.__bias_drift_rate = 1e-09 * bias_drift_rate
#
#         # Initialise the random number generator
#         self.__rng = np.random.default_rng(rand_seed)
#
#     def update(self, dt_sqrt: float):
#         """
#         Called every iteration to update clock measurement errors.
#         As clock errors propagate with time, it is important that this
#         method is called prior to every measurement update.
#
#         :param dt_sqrt: Square of the time difference since the last update.
#         :type dt_sqrt: float
#         """
#
#         drift_amount = dt_sqrt * self.__rng.normal()
#         self.__bias_value += self.__bias_drift_rate * drift_amount
#
#     def get_time(self, current_time: [float | tuple]) -> float:
#         """
#         Returns the believed elapsed time since the start of the simulation.
#         This is given in seconds (with considerations for rounding errors).
#
#         :param current_time: The true current time of the simulation.
#         :type current_time: float
#
#         :return: The current time with clock errors applied (in seconds).
#         :rtype: float
#         """
#         return round(current_time + self.__bias_value, 12)



class Clock:
    """
    Provides the functionality of a system clock in a simulated environment.
    Distinct from typical sensor hardware, the clock represents a virtual
    clock that simulated the recording of elapsed time with configured error drift.
    """

    # All properties of the class.
    __slots__ = '_bias_value', '_bias_drift_rate', '_time_elapsed', '_rng'

    def __init__(self, clock_bias: float = 0, bias_drift_rate: float = 0,
                 start_time: float = 0.0, rand_seed: int = None):
        """
        Initialises a clock instance with configured errors/rates.

        :param clock_bias: The initial clock bias (in nano-sec).
        :type clock_bias: float

        :param bias_drift_rate: The clock error drift rate (in
            nano-sec/root sec)
        :type bias_drift_rate: float

        :param start_time: The start time of the simulation.
        :type start_time: float

        :param rand_seed: The random seed used for random number generation.
        :type rand_seed: int
        """

        # Record error properties
        self._bias_value = clock_bias
        self._bias_drift_rate = 1e-09 * bias_drift_rate

        # Record the start time of clock
        self._time_elapsed = start_time

        # Initialise the random number generator
        self._rng = np.random.default_rng(rand_seed)

    def update(self, time_sec: float):
        """
        Called every iteration to update clock measurement errors.
        As clock errors propagate with time, it is important that this
        method is called prior to every measurement update.

        :param time_sec: The current simulation time in seconds.
        :type time_sec: float
        """

        # Calculate the amount of time elapsed:
        dt = time_sec - self._time_elapsed

        # Propagate errors on time increase:
        if dt > 0:
            self._time_elapsed = time_sec
            drift_amount = sqrt(dt) * self._rng.normal()
            self._bias_value += self._bias_drift_rate * drift_amount

    def get_time(self, current_time: float) -> float:
        """
        Returns the believed elapsed time since the start of the simulation.
        This is given in seconds (with considerations for rounding errors).

        :param current_time: The true current time of the simulation.
        :type current_time: float

        :return: The current time with clock errors applied (in seconds).
        :rtype: float
        """
        return round(current_time + self._bias_value, 12)


class DummyClock(Clock):

    def __init__(self, start_time: float = 0.0):
        super().__init__(0, 0, start_time=start_time)

    def update(self, time_sec: float):
        pass

    def get_time(self, current_time: float) -> float:
        return current_time