"""
=========
sensor.py
=========

:summary:
    Defines the abstract base classes for sensor usage in simulations.
    This module contains the base classes for both sensors (used for taking
    GroundTruth records and producing measurements) and sensor fusion (methods
    for applying measurement information to the current estimated states).

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""

import numpy as np

from qnav.estimation.state import EstimatedState
from qnav.waypoints.trajectory import GroundTruth

from collections.abc import Iterable
from math import isfinite
from abc import abstractmethod
from abc import ABC
from enum import Flag
from enum import auto


class Sensor(ABC):
    """
    Base class for sensors used in navigation.
    Defines the format of sensors used in the toolbox, specifically the
    methods, inputs and outputs expected for fusion and other classes.
    """

    # Defines all properties used:
    __slots__ = [
        '_last_measurement',
        '_time_elapsed',
        '_next_update',
        '_frequency',
        '_time_step',
        '_rng'
    ]

    def __init__(self,
                 frequency: float,
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Base constructor used to initialise the measurement frequency clock,
        elapsed time and random number generation.

        :param frequency: The exact measurement frequency of the sensor in Hz.
        :type frequency: float

        :param start_time: The starting time of the sensor in seconds.
        :type start_time: float

        :param rand_seed: The random number generation seed.
        :type rand_seed: int
        """

        # Capture the requested frequency.
        self._frequency = frequency
        self._time_step = 1 / frequency

        # Record the time elapsed.
        self._time_elapsed = start_time
        self._next_update = self._time_elapsed + self._time_step

        # Initialise a random number generator.
        self._rng = np.random.default_rng(rand_seed)

        # Validate the requested sensor frequency.
        assert self._frequency > 0 and isfinite(self._frequency), \
            "Sensor frequency must be a positive finite value!"

        # Validate the requested sensor frequency.
        assert self._time_elapsed >= 0 and isfinite(self._time_elapsed), \
            "Elapsed time must be a non-negative finite value!"

    @abstractmethod
    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        """
        Produces the measurement captured by the sensor.
        Given the current estimated time and the ground truth, this method
        shall return the measurement captured at that time. Alternatively, a
        None value can be returned to signify a measurement is not ready.

        :param est_time: The estimated time the measurement of the sensor
            is expected to be captured at (in seconds).
        :type est_time: float

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth
        """
        pass

    @property
    @abstractmethod
    def last_measurement(self) -> any:
        """
        Returns the latest recorded measurement produced by sensor.

        :return: Latest measurement produced by sensor.
        :rtype: any
        """
        pass

    def update(self, time_sec: float = None):
        """
        Called on demand to update the sensor's state.
        Typically called after measurements have been taken. When called
        this increases the next measurement time and propagates internal
        states, to the time difference.

        :param time_sec: The current simulation time in seconds.
        :type time_sec: float
        """

        # Substitute None value for default time difference
        time_diff = self._time_step if time_sec is None \
            else (time_sec - self._time_elapsed)

        # Increase the time elapsed.
        self._time_elapsed += time_diff
        self._next_update = self._time_elapsed + self._time_step


    @property
    def last_update(self) -> float:
        """
        Returns the time (in seconds) of the last measurement.

        :return: Time of last measurement (in seconds).
        :rtype: float
        """
        return self._time_elapsed

    @property
    def next_update(self) -> float:
        """
        Returns the due time (in seconds) of the next measurement.

        :return: Time of next measurement (in seconds).
        :rtype: float
        """
        return self._next_update

    @property
    def frequency(self) -> float:
        """
        Returns the sensor's measurement frequency in Hz.

        :return: The frequency in Hz.
        :rtype: float
        """
        return self._frequency

    @property
    def time_step(self) -> float:
        """
        Returns the sensor's measurement time step in seconds.

        :return: The measurement interval in seconds.
        :rtype: float
        """
        return self._time_step


class FusionTrigger(Flag):
    """
    Flags used to indicate the triggering mode for fusion.
    These essentially define when fusion is triggered, subject to the sensor
    measurements that ara available.
    """

    # Fusion is triggered when any sensor has a measurement.
    ANY = auto()

    # Fusion is triggered when all sensors have measurements.
    ALL = auto()

    # Switches from ALL to ANY mode after first reset.
    ALL_THEN_ANY = auto()


class SensorFusion(ABC):
    """
    Base class for a sensor fusion method used in navigation.
    Defines the format of data fusion used in the toolbox, specifically the
    methods, inputs and outputs expected.
    """

    def __init__(self, trigger: FusionTrigger, *sensors: Sensor):
        """
        Base constructor used to initialise the trigger mode and sensors.
        Here a trigger mode must be supplied along with one or more shared
        sensors associated with the fusion method.

        :param trigger: The trigger mode for deciding when to apply fusion.
        :type trigger: FusionTrigger

        :param sensors: All shared sensors used by the fusion method.
        :type sensors: Iterable[Sensor]
        """

        # Obtain input parameters:
        self._trigger = trigger
        self._sensors = sensors

        # Validate given sensors:
        if len(self._sensors) == 0:
            raise ValueError("No sensors supplied!")

    @property
    def trigger(self) -> FusionTrigger:
        """
        The selected trigger method for deciding when to apply fusion.
        This decides when fusion can be applied, subject to the available
        sensor measurements.

        :return: The selected fusion trigger method.
        :rtype: FusionTrigger
        """
        return self._trigger

    @property
    def sensors(self) -> Iterable[Sensor]:
        """
        The provided sensors associated with the fusion method.
        This is a collection of all shared sensors whose measurements will be
        used by the fusion method.

        :return: The shared sensors used by the fusion method.
        :rtype: Iterable[Sensor]
        """
        return self._sensors

    @abstractmethod
    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Performs the fusion with sensors and updates the estimated state.
        When called at scheduled time during a simulation loop, this method
        performs the data fusion by extracting last measurements from sensors
        and using them with the fusion algorithm to update the given current
        estimated state.

        :param estimated_state: The current estimated states of the platform.
                                This is what the fusion method will update.
        :type estimated_state: EstimatedState
        """
        pass
