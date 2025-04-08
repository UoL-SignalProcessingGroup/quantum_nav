"""
=============
trajectory.py
=============

:summary:
    Used for the core representation of ground truth trajectory information.
    This module is responsible for handling ground truth waypoint
    trajectories, including the managing of input data, position and value
    calculations and look-up interpolation of ground truth records.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of this module.
"""


import numpy as np

from typing import Type
from dataclasses import dataclass

from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import CubicHermiteSpline

from qnav.gravity.base import GravityModel
from qnav.waypoints.vehicle import Vehicle
from qnav.waypoints.generation import get_positions
from qnav.waypoints.generation import get_trajectory
from qnav.util.transformations import quaternion_to_euler
from qnav.util.transformations import quaternion_to_euler_vec
from qnav.util.transformations import haversine_vec


@dataclass(frozen=True)
class GroundTruth:
    """
    A record class for trajectory ground truth a given time.
    This class contains the ground truth (the true trajectory values) at a
    given time. The purpose of this is to supply the "real-world" data to
    sensor models, where simulated noise and errors will then be applied to
    produce measurement data.
    """

    #: The time since initialisation (in seconds).
    timestamp: float

    #: The position as latitude-longitude-altitude (in deg-deg-metres).
    position: np.ndarray

    #: The velocity as along-across-down in body axis (in metres/second).
    velocity: np.ndarray

    #: The acceleration as along-across-down in body axis (in metres/second^2).
    acceleration: np.ndarray

    #: The attitude as north-east-down in NED axis (in degrees).
    attitude: np.ndarray

    #: The angle rates as north-east-down in NED axis (in degrees/second).
    angle_rates: np.ndarray

    def as_dict(self) -> dict:
        """
        Returns class properties in dictionary form.

        :return: A dictionary containing ground truth values.
        :rtype: dict
        """
        return {
            'timestamp': self.timestamp,
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'attitude': self.attitude,
            'angle_rates': self.angle_rates
        }

    def as_numpy(self) -> np.ndarray:
        """
        Returns call properties in 1D numpy array form.

        :return: A numpy array containing ground truth values.
        :rtype: np.ndarray
        """
        return np.concatenate((
            [self.timestamp],
            self.position,
            self.velocity,
            self.acceleration,
            self.attitude,
            self.angle_rates
        ))


@dataclass(frozen=True)
class Waypoints:
    """
    A container for trajectory waypoint data used to define a course.
    This data class will contain the arrival time and positioning records
    for core points along a trajectory. These can either represent sparse
    base points that a trajectory is to cover, or a series of dense
    positions, a fixed frequency, for an already generated trajectory.
    Additionally, attitude values can be provided for each position.
    """

    timestamps: np.ndarray
    lat_points: np.ndarray
    lon_points: np.ndarray
    alt_points: np.ndarray
    attitudes: np.ndarray = None

    def __post_init__(self) -> None:
        """
        Post-initialisation parameter validation.
        Performs checks after the waypoint are initialised to ensure all
        given data is valid and usable for trajectory generation.
        """

        to_test = [self.lat_points, self.lon_points, self.alt_points, self.timestamps]
        if any([x is None for x in to_test]):
            raise ValueError('Incomplete waypoint data provided')

        if self.attitudes is not None:
            to_test.append(self.attitudes)

        exp_size = to_test.pop(0).shape

        if len(exp_size) != 1 or exp_size[0] == 0:
            raise ValueError('Records must be one-dimensional')

        if any([np.array_equal(exp_size, x.size) for x in to_test]):
            raise ValueError('Records must be of same size')

        if not all([np.all(np.isfinite(x)) for x in to_test]):
            raise ValueError('Records must be of finite values')

    def as_array(self) -> np.ndarray:
        """
        Returns required records formatted as column arrays.
        Specifically returns an n-by-4 array holding the timestamps, latitude,
        longitude and altitude records as columns, respectively. Optional
        Attitude records are not included in this output.

        :return: Position data formatted as column array.
        :rtype: np.ndarray
        """
        return np.column_stack((
            self.timestamps,
            self.lat_points,
            self.lon_points,
            self.alt_points
        ))


class Trajectory:
    """
    A fully-generated trajectory with ground truth look-up.
    On initialization, this generates the full trajectory from given base
    waypoints, vehicle model and gravity model. The core trajectory is
    generated with base points calculated at fixed frequency. Intermediate
    values are then interpolated.
    """

    # Base data frequency (in Hz).
    __base_freq: float

    # The time of the first record (in seconds).
    __start_time: float

    # The time of the final record (in seconds).
    __end_time: float

    # The base position waypoints used. Maintained for verification.
    __waypoints: Waypoints

    # The vehicle model used for maneuvering constraints and noise factors.
    __vehicle_model: Vehicle

    # The gravity model used for true gravity calculations.
    __gravity_model: GravityModel

    # The trajectory interpolation use for value look-up.
    __interp: CubicHermiteSpline

    # The selected interpolation method
    __interp_method: str

    def __init__(self,
                 waypoints: Waypoints,
                 vehicle_model: Vehicle,
                 gravity_model: GravityModel,
                 base_freq: float,
                 inter_method: str = 'pchip'):
        """
        Generates the trajectory and look-up interpolation.
        Given the base waypoint, vehicle and gravity models and base
        frequency, this generates and returns the trajectory. On succession,
        this allows direct look-up of the trajectory ground truth values at
        any requested time.

        If the frequency of the waypoint data is sufficiently high-enough,
        interpolation of positions will be performed directly.

        :param waypoints: The waypoints defining positions on the trajectory.
        :rtype waypoints: Waypoints

        :param vehicle_model: The vehicle model used for applying maneuvering
            constraints, orientation behaviour and noise factors (the latter
            used in advanced settings).
        :rtype vehicle_model: Vehicle

        :param gravity_model: The gravity model used for calculating the
            gravitation acceleration at requested positions.
        :rtype gravity_model: GravityModel

        :param base_freq: The frequency of interpolation points (in Hz).
            A frequency of around 50-100 Hz is recommended. Higher
            frequencies have increased chances of resulting in numerical
            instabilities.
        :type base_freq: float

        :param inter_method: The interpolation method to use.
            Supported methods are 'pchip', 'cubic' and 'akima'.
        :type inter_method: str
        """

        if base_freq <= 0:
            raise ValueError('Frequency must be positive')

        self.__vehicle_model = vehicle_model
        self.__waypoints = waypoints
        self.__gravity_model = gravity_model
        self.__base_freq = base_freq
        self.__interp_method = inter_method.casefold()

        interp_class = self.__get_interp_class()
        waypoint_freq = 1 / np.max(np.diff(waypoints.timestamps))

        # TODO: Add fixed minimum thresholding!
        # if waypoint_freq < self.__base_freq:
        if round(waypoint_freq) < (self.__base_freq / 10):
            base_points = self.__calculate_positions()
        else:
            base_points = self.__interp_positions()

        # TODO: Support use of attitude values
        data_points = self.__calculate_values(base_points)
        self.__start_time = float(data_points[0, 0])
        self.__end_time = float(data_points[-1, 0])

        x_points = data_points[:, 0]
        y_points = data_points[:, 1:]
        self.__interp = interp_class(x_points, y_points)

    def get_record(self, time_sec: float) -> GroundTruth:
        """
        Returns values of the trajectory at given time (in seconds).
        Safely interpolates and returns ground truth at requested time.

        :param time_sec: A requested time in seconds.
        :type time_sec: float

        :return: The corresponding ground truth record.
        :rtype: GroundTruth
        """

        record = np.array(self.__interp(time_sec))
        record.setflags(write=False)

        position = record[0:3]
        velocity = record[3:6]
        acceleration = record[6:9]
        angle_rates = record[12:15]
        attitude = quaternion_to_euler(record[15:19])

        return GroundTruth(
            time_sec, position, velocity, acceleration,
            attitude, angle_rates
        )

    def get_records(self, time_steps: np.ndarray) -> dict[str, np.ndarray]:
        """
        Obtains full records of the trajectory for a given times (in seconds).
        Essentially performs vectorised interpolation for given time series.

        :param time_steps: The time steps (in seconds) to obtain records for.
        :type time_steps: np.ndarray

        :return: A dictionary holding the records for each time step.
        :rtype: dict[str, np.ndarray]
        """

        record = np.array(self.__interp(time_steps))

        return {
            'time_steps': time_steps,
            'position': record[:, 0:3],
            'velocity': record[:, 3:6],
            'acceleration': record[:, 6:9],
            'angle_rates': record[:, 12:15],
            'attitude': quaternion_to_euler_vec(record[:, 15:19])
        }

    @property
    def start_time(self) -> float:
        """
        Returns the time of the first trajectory record (in seconds).

        :return: The start time of the trajectory
        :rtype: float
        """
        return self.__start_time

    @property
    def end_time(self) -> float:
        """
        Returns the time of the final trajectory record (in seconds).

        :return: The end time of the trajectory
        :rtype: float
        """
        return self.__end_time

    @property
    def interp_method(self) -> str:
        """
        The interpolation method used for ground truth generation.

        :return: Interpolation name.
        :rtype: str
        """
        return self.__interp_method

    def __calculate_positions(self) -> np.ndarray:
        """
        Calculates intermediate positions using additional information.
        This calculates and returns all positions of the trajectory at set
        frequency, using all information including optional vehicle model and
        orientation positions.

        :return: An array holding time steps and LLA positions.
        :rtype: np.ndarray
        """
        return get_positions(self.__waypoints.lat_points,
                             self.__waypoints.lon_points,
                             self.__waypoints.alt_points,
                             self.__vehicle_model,
                             self.__base_freq, float('nan'),
                             self.__waypoints.timestamps)

    def __interp_positions(self) -> np.ndarray:
        """
        Simply interpolates intermediate positions.
        Used as an alternative to fully calculating them with a vehicle
        model. This function simply interpolates the data at set frequency.
        This is used when the provided data is either of sufficient
        resolution or too high of resolution.

        :return: An array holding time steps and LLA positions.
        :rtype: np.ndarray
        """

        time_step = 1 / self.__base_freq
        old_time_steps = self.__waypoints.timestamps

        start_time = time_step * (old_time_steps[0] // time_step)
        end_time = time_step * (old_time_steps[-1] // time_step)
        new_time_steps = np.arange(start_time, end_time, time_step)

        interp_class = self.__get_interp_class()
        interp = interp_class(old_time_steps, self.__waypoints.as_array())
        return interp(new_time_steps)

    def __calculate_values(self, base_points: np.ndarray) -> np.ndarray:
        """
        Calculates the trajectory values at all positions.
        Given an array of high-resolution and fix frequency base points,
        this method will calculate the trajectory values at each step. This
        essentially populates the look-up table for the trajectory.

        :param base_points: High-resolution base points to calculate
        trajectory values for. These must be given at uniform
        time steps.
        :type base_points: np.ndarray (n-by-4 elements)

        :return: An array holding the trajectory values at each step.
        :rtype: np.ndarray (n-by-20 elements)
        """
        return get_trajectory(base_points, self.__gravity_model)

    def __get_interp_class(self) -> Type[CubicSpline | PchipInterpolator | Akima1DInterpolator]:
        """
        For given interpolation name, returns the class type to use.
        A mapping function which returns the CubicHermiteSpline class type that
        corresponds to the given interpolation name. If fails to match, a Value
        error will be thrown.

        :return: The interpolation class to use.
        :rtype: Type[CubicHermiteSpline]
        """

        match self.__interp_method:

            case 'cubic':
                return CubicSpline

            case 'pchip':
                return PchipInterpolator

            case 'akima':
                return Akima1DInterpolator

            case _:
                ValueError(f"Unsupported interpolation method '{self.__interp_method}'")



def cumulative_distance(lat_points: np.ndarray,
                        lon_points: np.ndarray,
                        alt_points: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative 'great-circle' distance between coordinate points.
    For a given series of latitude-longitude-altitude points (given in
    deg-deg-m), this calculates the cumulative distance between them using
    the standard WGS-84 reference ellipsoid.

    :param lat_points: A series of latitude positions (in decimal degrees).
    :type lat_points: np.ndarray (n-elements)

    :param lon_points: A series of longitude positions (in decimal degrees).
    :type lat_points: np.ndarray (n-elements)

    :param alt_points: A series of altitude positions (in decimal degrees).
    :type lat_points: np.ndarray (n-elements)

    :return: The cumulative distance all points, starting from first.
    :rtype: np.ndarray
    """

    # Validate the input to ensure it is of the same one dimensional size
    if np.ndim(lat_points) != 1 or lat_points.shape != lon_points.shape != lon_points.shape:
        raise ValueError('input must be of same 1-dimensional size')

    # Pre-allocate the return distances
    distances = np.zeros(len(lat_points))

    # Calculate the distance between the waypoints.
    distances[1:] = haversine_vec(
        lat_points[:-1], lon_points[:-1], alt_points[:-1],
        lat_points[1:], lon_points[1:], alt_points[1:])

    # Calculate the cumulative sum of the times.
    return np.cumsum(distances)


# if __name__ == '__main__':
#
#
#     test_lats = np.array([0.0, 0.10])
#     test_lons = np.array([0.0, 0.0])
#     test_alts = np.array([0.0, 0.0])
#
#     vehicle = Vehicle()
#     avg_speed = 10.0
#     freq = 10.0
#
#     test_out = get_positions(test_lats, test_lons, test_alts,
#                              vehicle, freq, avg_speed)
#
#     data_out = get_trajectory(test_out, FixedValue())












