"""
========
state.py
========

:summary:
    Supported base estimation state classes used to hold estimated data.
    This module contains classes that handle the estimated state of the
    platform during simulations. These are shared between the employed
    fusion methods. Extensions of the base class are required for the
    handling of addition data (such as error states and kalman matrices).

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import numpy as np

from copy import copy, deepcopy
from qnav.gravity.base import GravityModel
from qnav.gravity.simple import SimpleUniform
from qnav.waypoints.trajectory import GroundTruth


class EstimatedState:
    """
    The base estimated state holding only general state information.
    Simply maintains the estimated time, position, velocity, acceleration,
    attitude and angle rates of the platform. Additionally, this also holds
    a shared gravity model for calculating the expected gravity dynamics.
    """

    # All properties of the class
    __slots__ = (
        '_est_timestamp',
        '_est_position',
        '_est_velocity',
        '_est_acceleration',
        '_est_attitude',
        '_est_angle_rates',
        '_gravity_model'
    )

    def __init__(self, ground_truth: GroundTruth,
                 gravity_model: GravityModel = SimpleUniform()):
        """
        Create an estimated state with initial values matching the ground truth.
        This generates an estimated state with properties initialised using
        the given ground truth record. Additionally, a shared gravity model
        can be given that will be used among fusion methods that update
        the estimated state.

        :param ground_truth: Ground truth record used to initialise estimates.
            This means that all initial estimates will be correct (unless
            modified prior to simulation).
        :type ground_truth: GroundTruth

        :param gravity_model: The shared gravity model to use for estimation.
            This is required by some fusion methods as a reference for
            expected gravity.
        :type gravity_model: GravityModel
        """

        # Unpack the ground truth (copy values):
        self._est_timestamp = copy(ground_truth.timestamp)
        self._est_position = copy(ground_truth.position)
        self._est_velocity = copy(ground_truth.velocity)
        self._est_acceleration = copy(ground_truth.acceleration)
        self._est_attitude = copy(ground_truth.attitude)
        self._est_angle_rates = copy(ground_truth.angle_rates)

        # Unpack the sensor axis and gravity model:
        self._gravity_model = gravity_model

    @property
    def timestamp(self) -> float:
        """
        Returns the current estimated duration of the simulation (in seconds).

        :return: The estimated time (in seconds).
        :rtype: float
        """
        return self._est_timestamp

    @property
    def position(self) -> np.ndarray:
        """
        Returns the estimated position (in degrees-degrees-metres).
        This is the estimated position represented by an array of latitude,
        longitude and altitude values respectively.

        :return: The estimated position (in degrees-degrees-metres).
        :rtype: np.ndarray
        """
        return copy(self._est_position)

    @property
    def velocity(self) -> np.ndarray:
        """
        Returns the estimated velocity (in meters/second).
        This is the estimated velocity in body axis represented as an array
        of along, across and down velocities respectively.

        :return: The estimated velocity (in meters/second).
        :rtype: np.ndarray
        """
        return copy(self._est_velocity)

    @property
    def acceleration(self) -> np.ndarray:
        """
        Returns the estimated acceleration (in meters/second^2).
        This is the estimated acceleration in body axis represented as an
        array of along, across and down accelerations respectively.

        :return: The estimated acceleration (in meters/second^2).
        :rtype: np.ndarray
        """
        return copy(self._est_acceleration)

    @property
    def attitude(self) -> np.ndarray:
        """
        Returns the estimated attitude (in degrees).
        This is the estimated attitude in NED axis represented as an array
        of heading, pitch and yaw angles respectively.

        :return: The estimated attitude (in degrees).
        :rtype: np.ndarray
        """
        return copy(self._est_attitude)

    @property
    def angle_rates(self) -> np.ndarray:
        """
        Returns the estimated angle rates (in degrees/second).
        This is the estimated angle rates in body axis represented as an
        array of P, Q and R euler angles respectively.

        :return: The estimated angle rates (in degrees/second).
        :rtype: np.ndarray
        """
        return copy(self._est_angle_rates)

    @property
    def gravity_model(self) -> GravityModel:
        """
        Returns the shared gravity model.
        This is the gravity model instance used for estimating the
        gravitational effect during estimation and fusion. For consistency,
        this is held in the estimated state so it is shared.

        :return: The estimation gravity model.
        :rtype: GravityModel
        """
        return self._gravity_model

    def update_estimates(self,
                         timestamp: float = None, position: np.ndarray = None,
                         velocity: np.ndarray = None, acceleration: np.ndarray = None,
                         attitude: np.ndarray = None, angle_rates: np.ndarray = None):
        """
        Overwrites selected current estimated states.
        This is namely used when estimation fusion is applied and the
        internal states must be updated. To prevent accidental overwriting of
        estimated states or the setting of invalid data, all changes must be
        submitted through this method.

        :param timestamp: The updated timestamp marking the estimated current
            duration of the simulation (in seconds).
        :type timestamp: float

        :param position: The updated position vector containing the
            latitude (in degrees), longitude (in degrees) and altitude
            (in metres) respectively.
        :type position: numpy.ndarray (3-elements)

        :param velocity: The updated velocity vector for the
            North, East and Down components respectively (in m/s).
        :type velocity: numpy.ndarray (3-elements)

        :param acceleration: The updated acceleration vector for
            the North, East and Down components respectively (in m/s^2).
        :type acceleration: numpy.ndarray (3-elements)

        :param attitude: The updated attitude vector for the Heading,
            Pitch and Roll components respectively (in degrees).
        :type attitude: numpy.ndarray (3-elements)

        :param angle_rates: The updated angle rate vector for the
            P, Q and R body rates respectively (in deg/s).
        :type angle_rates: numpy.ndarray (3-elements)
        """
        if timestamp is not None:
            self._est_timestamp = timestamp

        if position is not None:
            self._est_position[:] = position

        if velocity is not None:
            self._est_velocity[:] = velocity

        if acceleration is not None:
            self._est_acceleration[:] = acceleration

        if attitude is not None:
            self._est_attitude[:] = attitude

        if angle_rates is not None:
            self._est_angle_rates[:] = angle_rates

    def as_dict(self) -> dict[str, float | np.ndarray]:
        """
        Represents the state and its content as a dictionary.
        Namely used when a simple representation is needed.

        :return: States in a python dictionary.
        :rtype: dict[str, float | np.ndarray]
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
        Represents the state and its content as a numpy array.
        Namely used when a simple representation is needed.

        :return: States in concatenated numpy array.
        :rtype: numpy.ndarray
        """

        return np.concatenate((
            [self._est_timestamp],
            self._est_position,
            self._est_velocity,
            self._est_acceleration,
            self._est_attitude,
            self._est_angle_rates
        ), axis=0)

    def clone(self):
        """
        Generates a cloned instance of itself.

        :return: Deep clone of this instance.
        :rtype: State
        """
        return deepcopy(self)

    def __str__(self) -> str:
        """
        Represents the state and its content as a string.

        :return: A String holding the state's variables.
        :rtype: str
        """

        pos = self.position
        vel = self.velocity
        acc = self.acceleration
        att = self.attitude
        ang = self.angle_rates

        return "\n".join((
            f"Timestep:\t\t {self._est_timestamp}",
            f"Position:\t\t ({pos[0]}, {pos[1]}, {pos[2]})",
            f"Velocity:\t\t ({vel[0]}, {vel[1]}, {vel[2]})",
            f"Acceleration:\t ({acc[0]}, {acc[1]}, {acc[2]})",
            f"Attitude:\t\t ({att[0]}, {att[1]}, {att[2]})",
            f"Angle Rates:\t ({ang[0]}, {ang[1]}, {ang[2]})",
        ))


class KalmanEstimatedState(EstimatedState):
    """
    Extended estimation state featuring Kalman Filter properties.
    This is an extension for the standard estimated state so that Kalman
    states can also be represented. These additions include shared Kalman
    Filter matrices and state errors.
    """

    def __init__(self, ground_truth: GroundTruth,
                 h: np.ndarray, r: np.ndarray,
                 f: np.ndarray, q: np.ndarray,
                 state_errors: np.ndarray = np.eye(15) * 1e-9,
                 gravity_model: GravityModel = SimpleUniform()):
        """
        Create an estimated Kalman State with ground truth values.
        This generates a Kalman estimated state with properties initialised
        using the given ground truth record and kalman matrices. Additionally,
        a shared gravity model can be given that will be used among fusion
        methods that update the estimated state.

        :param ground_truth: Ground truth record used to initialise estimates.
            This means that all initial estimates will be correct (unless
            modified prior to simulation).
        :type ground_truth: GroundTruth

        :param h: Measurement matrix H (measurements in Local NED/Earth axes).
        :type h: np.ndarray (6-by-15 elements)

        :param r: Measurement Noise Matrix R (measurements in Sensor axes).
        :type r: np.ndarray (6-by-6 elements)

        :param f: Predict State Vector/Covariance Matrix F for time step.
        :type f: np.ndarray (15-by-15 elements)

        :param q: Process Noise Matrix Q (all states in Local NED/Earth axes).
        :type q: np.ndarray (15-by-15 elements)

        :param state_errors: The Kalman state error matrix.
        :type state_errors: np.ndarray (15-by-15 elements)

         :param gravity_model: The shared gravity model to use for estimation.
            This is required by some fusion methods as a reference for
            expected gravity.
        :type gravity_model: GravityModel
        """

        # Use parent class constructor:
        super().__init__(ground_truth, gravity_model)

        # Copy given kalman matrices
        self._h_matrix = np.copy(h)
        self._r_matrix = np.copy(r)
        self._f_matrix = np.copy(f)
        self._q_matrix = np.copy(q)
        self._state_errors = np.copy(state_errors)

        self._h_matrix.setflags(write=False)
        self._r_matrix.setflags(write=False)
        self._f_matrix.setflags(write=False)
        self._q_matrix.setflags(write=False)

    @property
    def h_matrix(self) -> np.ndarray:
        return self._h_matrix

    @property
    def r_matrix(self) -> np.ndarray:
        return self._r_matrix

    @property
    def f_matrix(self) -> np.ndarray:
        return self._f_matrix

    @property
    def q_matrix(self) -> np.ndarray:
        return self._q_matrix

    @property
    def state_errors(self) -> np.ndarray:
        return np.copy(self._state_errors)

    @property
    def state_vector(self):
        """
        Returns estimated states in form of state vector.
        A commonly used representation of estimated states in Kalman Filters.

        :return: Kalman State Vector
        :rtype: numpy.ndarray (15-elements)
        """
        state_vector = np.zeros(15)
        state_vector[[0, 3, 6]] = self.position
        state_vector[[1, 4, 7]] = self.velocity
        state_vector[[2, 5, 8]] = self.acceleration
        state_vector[[9, 11, 13]] = self.attitude
        state_vector[[14, 12, 10]] = self.angle_rates
        return state_vector

    def update_estimates(self,
            timestamp: float = None, position: np.ndarray = None,
            velocity: np.ndarray = None, acceleration: np.ndarray = None,
            attitude: np.ndarray = None, angle_rates: np.ndarray = None,
            state_errors: np.ndarray = None):
        """
        Overwrites selected current estimated states.
        This is namely used when estimation fusion is applied and the
        internal states must be updated. To prevent accidental overwriting of
        estimated states or the setting of invalid data, all changes must be
        submitted through this method.

        :param timestamp: The updated timestamp marking the estimated current
            duration of the simulation (in seconds).
        :type timestamp: float

        :param position: The updated position vector containing the
            latitude (in degrees), longitude (in degrees) and altitude
            (in metres) respectively.
        :type position: numpy.ndarray (3-elements)

        :param velocity: The updated velocity vector for the
            North, East and Down components respectively (in m/s).
        :type velocity: numpy.ndarray (3-elements)

        :param acceleration: The updated acceleration vector for
            the North, East and Down components respectively (in m/s^2).
        :type acceleration: numpy.ndarray (3-elements)

        :param attitude: The updated attitude vector for the Heading,
            Pitch and Roll components respectively (in degrees).
        :type attitude: numpy.ndarray (3-elements)

        :param angle_rates: The updated angle rate vector for the
            P, Q and R body rates respectively (in deg/s).
        :type angle_rates: numpy.ndarray (3-elements)

        :param state_errors: The update Kalman state errors.
        :type state_errors: numpy.ndarray (15-by-15 elements)
        """

        # Call parent class's method for other parameters
        super().update_estimates(
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            attitude=attitude,
            angle_rates=angle_rates)

        # Update the error state if given:
        if state_errors is not None:
            self._state_errors = state_errors
