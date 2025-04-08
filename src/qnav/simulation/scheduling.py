"""
=============
scheduling.py
=============

:summary:
    Coordinates the scheduling of sensors and fusion methods.
    This module provides classes that are used for organising the timing of
    sensor measurements and fusion methods during simulations. Essentially
    this is used to efficiently get the next non-inform time increment
    along with what is to occur at that time.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

from copy import copy
from typing import Iterable, Iterator
from qnav.measurement.sensor import Sensor
from qnav.measurement.sensor import SensorFusion
from qnav.measurement.sensor import FusionTrigger


class _SensorStatusMap:
    """
    A mapping between a fusion method and its sensors' status.
    This mapping is used to record which sensors have been used/updated
    recently. When a sensor has been used a flag for its status is set.
    When  a sufficient amount of sensor flags are set, the fusion can then
    be performed. After this, all flags can be reset.
    """

    # Slots used to reduce memory and improve look-up time.
    __slots__ = ('_fusion', '_group_trigger', '_sensors_ready')

    # The reference fusion method for the sensors.
    _fusion: SensorFusion

    # The triggering mode for the fusion method.
    _group_trigger: FusionTrigger

    # A ready status look-up for all sensors.
    _sensors_ready: dict[Sensor, bool]

    def __init__(self, sensor_fusion: SensorFusion):
        """
        Creates a status mapping between the fusion method and whether the
        sensors required by it have updated. This, in turn, is then used to
        flag that the fusion method is ready.

        :param sensor_fusion: The SensorFusion groups used in the simulation.
        :type sensor_fusion: SensorFusion
        """

        # Reference the fusion method:
        self._fusion = sensor_fusion

        # Copy the trigger used for the fusion method:
        self._group_trigger = copy(self._fusion.trigger)

        # Create a mapping for all sensors statuses:
        sensors = list(self._fusion.sensors)
        is_ready = [False] * len(sensors)
        self._sensors_ready = dict(
            zip(sensors, is_ready))

    def get_fusion(self) -> SensorFusion:
        """
        Returns the fusion method associated with the mapping.

        :return: The SensorFusion instance used in the mapping.
        :rtype: SensorFusion
        """
        return self._fusion

    def set_sensor_ready(self, sensor: Sensor) -> None:
        """
        Set the flag for a given sensor as ready.
        This is typically called directly prior or after a measurement from
        the sensor has been produced to notify its fusion method(s).

        :param sensor: The sensor instance which is ready to be used.
        :type sensor: Sensor
        """
        if sensor in self._sensors_ready:
            self._sensors_ready[sensor] = True

    def is_fusion_ready(self) -> bool:
        """
        States if the fusion method is ready to be performed.
        This is subject to ready flags of sensors and the trigger mode
        selected by the fusion method.

        :return: True if fusion is ready to be performed.
        :rtype: bool
        """
        status = [s for s in self._sensors_ready.values()]
        if self._group_trigger is FusionTrigger.ANY:
            return any(status)
        else:
            return all(status)

    def reset(self) -> None:
        """
        Resets all sensor flags, setting them to false.
        Simply resets the mapping for the fusion instance so that all ready
        flags are set to false. Commonly used once fusion has been performed.
        """
        self._sensors_ready.update((k, False) for k in self._sensors_ready)
        if self._group_trigger is FusionTrigger.ALL_THEN_ANY:
            self._group_trigger = FusionTrigger.ANY

    def print_mapping(self):
        """
        Displays the internal mapped sensor status in the output console.
        When called, prints the attached sensors and their ready statuses
        for the sensor fusion instance. Useful for debugging states.
        """

        print(f"{type(self._fusion).__name__}: {hex(id(self._fusion))} [{self._group_trigger}]")
        for sensor, status in self._sensors_ready.items():
            status_char = "✓" if status else "✗"
            print(f"\t{type(sensor).__name__}: {hex(id(sensor))} [{status_char}]")


class TimingHandler:
    """
    A handler for managing the timings for sensors and fusion methods.
    On creation this creates a mapping between shared sensors and fusion
    methods. Through this the sensor with the earliest due time can be
    obtained. Once this sensor is used (i.e a measurement has been produced)
    a flag can be set for the fusion methods associate with it. When
    sufficient sensor flags for a fusion method are set, the fusion method
    can then be also obtained and returned.
    """

    # Slots used to reduce memory and improve look-up time.
    __slots__ = ('_all_sensors', '_sensor_map')

    # A list of all unique sensor instances.
    _all_sensors: list[Sensor]

    # A mapping between sensors and their fusion methods' ready status.
    _sensor_map: dict[Sensor, list[_SensorStatusMap]]

    def __init__(self, hardware: Iterable[SensorFusion]):
        """
        Creates a timing scheduler for sensors and fusion methods.
        On creation returns a timing handler for the given
        class:`SensorFusion` instances.

        :param hardware: Collection of SensorFusion instances.
        :type hardware: Iterable[SensorFusion]
        """

        self._all_sensors = []
        self._sensor_map = {}

        # Get list of all unique sensor instances:
        for sensor_fusion in hardware:
            for sensor in sensor_fusion.sensors:
                if sensor not in self._all_sensors:
                    self._all_sensors.append(sensor)

        # Check unique sensors are usable:
        if len(self._all_sensors) == 0:
            raise ValueError('No usable sensors provided')

        # Add look-up dictionary for sensors-fusions:
        for sensor_fusion in hardware:
            mapping = _SensorStatusMap(sensor_fusion)

            # Ensure all associated sensors point to instance:
            for sensor in sensor_fusion.sensors:
                tmp = self._sensor_map.get(sensor, [])
                tmp.append(mapping)
                self._sensor_map[sensor] = tmp

    def update_sensors_past_time(self, delay_time: float) -> None:
        """
        Repeatedly updates sensors until all due time are past the given time.
        Applies updates and skips measurements of all sensors until their due
        times for next measurements are past the given time. This is used to
        correctly induce a delay for measurements from the start of the
        simulation. Note this does not collect measurements or change any
        of the connected statuses used for fusion.

        :param delay_time: The time to update all sensors past (in seconds).
            Essentially the delay before measurements are to be processed.
        :type delay_time: float
        """
        for sensor in self.sensor_iterator(delay_time):
            sensor.update()

    def set_sensor_used(self, sensor: Sensor) -> None:
        """
        Flags that the given sensor has been used.
        Marks the given sensor instance as used for all fusion methods that
        requires this. If conditions are met, this trigger fusion methods
        that use this sensor to be ready.

        :param sensor: The sensor instance that has been used recently.
        :type sensor: Sensor
        """
        [m.set_sensor_ready(sensor) for m
         in self._sensor_map[sensor]]

    def get_next_sensor(self) -> Sensor:
        """
        Returns the sensor with the earliest due time for measurement.
        Searches through shared sensors and returns the one with the
        earlier due time.

        :return: Sensor with the earliest due time.
        :rtype: Sensor
        """
        times = [s.next_update for s in self._all_sensors]
        index = times.index(min(times))
        return self._all_sensors[index]

    def sensor_iterator(self, max_time: float) -> Iterator[Sensor]:
        """
        Generates an iterator for continually getting the next sensor.
        The produced iterator will be return the sensor with the earliest
        measurement due time. The iteration stops when the next earliest
        due time exceeds the given max time.

        :param max_time: The maximum time due for measurements (in seconds).
        :type max_time: float

        :return: An iterator returning the sensor with the earliest due time.
        :rtype: Iterator[Sensor]
        """
        while ((sensor := self.get_next_sensor()) and
               (sensor.next_update <= max_time)):
            yield sensor
            # sensor.update()

    def get_next_fusion(self, sensor: Sensor = None) -> list[SensorFusion]:
        """
        Returns any fusion methods that are to be used.
        Searches through fusion methods and returns a list collection of those
        that are ready to be used. These are automatically marked as used on
        return. If no fusion methods are ready, then the list will be empty.
        For faster retrieval, the search of fusion methods can be filtered
        to only those assigned with a given sensor instance.

        :param sensor: Sensor instance to filter fusion methods by (Optional).
        :type sensor: Sensor

        :return: A list of fusion methods ready to be applied.
        :rtype: list[SensorFusion]
        """

        # If no sensor provided:
        if sensor is None:

            # Iterate across all unique sensors and return flattened list
            to_flatten = [self.get_next_fusion(s) for s in self._all_sensors]
            return [item for sublist in to_flatten for item in sublist]

        to_return = []
        for m in self._sensor_map[sensor]:
            if m.is_fusion_ready():
                to_return.append(m.get_fusion())
                m.reset()

        return to_return

    def print_mappings(self):
        """
        Displays the internal scheduling mappings in the output console.
        When called, prints the fusion methods attached to each sensor,
        along with their ready status. Useful for debugging states.
        """

        print(f"{"[ Sensor Mappings ]":-^60}")

        mapping_set = set()
        for mappings in self._sensor_map.values():
            for mapping in mappings:
                mapping_set.add(mapping)

        for mapping in mapping_set:
            mapping.print_mapping()

        print(f"{"":-^60}")



