"""
=============
gps/sensor.py
=============

:summary:
    GPS sensor activities used for collecting measurements from satellites.
    This module provides sensor classes that are responsible for obtaining
    and returning GPS measurements from selected satellites. These
    measurements can then be passed to the appropriate GPS fusion method.

:authors:
    | Kallum O'Hara - sgkohara@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

# .. TODO:
#       - Implement sensor axes properties

from typing import Collection
from qnav.gps.satellite import Satellite
from qnav.gps.zones import FlaggedPolygonGroup
from qnav.measurement.sensor import Sensor
from qnav.waypoints.trajectory import GroundTruth


class GpsSensor(Sensor):
    """
    A basic sensor for collecting measurements from GPS satellites.
    This sensor can be used to collect measurements on demand from
    presented GPS satellites at requested frequency.
    """

    def __init__(self, frequency: float,
                 satellites: Collection[Satellite],
                 gps_zones: FlaggedPolygonGroup = None,
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Generates a GPS sensor that produces GPS satellite measurements.

        :param frequency: The exact measurement frequency of the sensor in Hz.
        :type frequency: float

        :param satellites: The satellites to collect measurements from.
        :type satellites: Collection[Satellite]

        :param start_time: The starting time of the sensor in seconds.
        :type start_time: float

        :param rand_seed: The random number generation seed.
        :type rand_seed: int
        """
        super().__init__(frequency, start_time, rand_seed)
        self._satellites = list(satellites)
        self._last_measurement = None
        self._gps_zones = gps_zones

    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        """
        Obtains updated measurement from GPS satellites.
        Obtains measurements from all GPS satellites and updates the last
        measurement. Performed prior to GPS fusion.

        :param est_time: The estimated time the measurement of the sensor
            is expected to be captured at (in seconds).
        :type est_time: float

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth
        """

        # Unpack the ground truth data
        ant_pos_lla = ground_truth.position
        ant_vel_body = ground_truth.velocity
        ant_att_deg = ground_truth.attitude

        # Get a measurement from each of the satellites.
        sat_measurements = [sat.get_measurements(
            ant_pos_lla, ant_att_deg, ant_vel_body)
            for sat in self._satellites]

        # Apply zones if active:
        is_usable = True
        if self._gps_zones is not None:
            lat, lon = ground_truth.position[:2]
            is_usable = self._gps_zones.get_point_flag(lat, lon)

        # Record the last measurement group
        self._last_measurement = {
            'is_usable': is_usable,
            'sat_measurements': sat_measurements
        }

    @property
    def last_measurement(self) -> dict:
        """
        Returns the list of latest GPS satellite measurements.

        :return: Latest GPS satellite measurements.
        :rtype: dict
        """
        return self._last_measurement

    def update(self, time_sec: float = None):
        """
        Updates the state of the satellites for the given amount of seconds.
        This method is used to update each of the underlying satellites in
        the given collection.

        :param time_sec: The estimated current simulation time.
        :type time_sec: float
        """
        # Substitute None value for default time difference
        if time_sec is None:
            dt = self._time_step
        else:
            dt = (time_sec - self._time_elapsed)

        # Increase the time elapsed.
        self._time_elapsed += dt
        self._next_update = self._time_elapsed + self._time_step
        [s.update(dt) for s in self._satellites]