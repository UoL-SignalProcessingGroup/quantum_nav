"""
======
ins.py
======

:summary:
    Implementation of a basic numerical Inertial Navigation System (INS).
    This module contains the standard INS implementation which is based
    of simple numerical integration. Other compatible INS solutions will
    inherit from this model.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""


import math
import numpy as np
import qnav.util.constants as const
import qnav.util.transformations as trans

from qnav.gravity.base import GravityModel
from qnav.gravity.simple import SimpleUniform
from qnav.util.transformations import cross_prod_xy


class StandardINS:
    """
    The standard Inertial Navigation System (INS) featured in the project.
    This INS solution is based on simple numerical integration. While its
    estimation calculations are relatively primitive, it serves as a base
    model that other solutions can extend from. The simple implementation
    also provides straight-forward means for testing calculations.
    """

    # All properties of the class
    __slots__ = \
        '_est_position', \
        '_est_velocity', \
        '_est_acceleration', \
        '_est_attitude', \
        '_est_angle_rates', \
        '_altimeter_gain', \
        '_sensor_angles_rad', \
        '_lever_arm', \
        '_state_errors', \
        '_gravity_model'

    def __init__(self, position: np.ndarray,
                 velocity: np.ndarray,
                 acceleration: np.ndarray,
                 attitude: np.ndarray,
                 angle_rates: np.ndarray,
                 gravity_model: GravityModel = SimpleUniform(),
                 altimeter_gain: float = 0.1):
        """
        Initialises the INS based on initial estimates.

        :param position: The initial estimated position vector containing the
            latitude (in degrees), longitude (in degrees) and altitude (in
            metres) respectively.
        :type position: numpy.ndarray (3-elements)

        :param velocity: The initial estimated velocity vector for the
            North, East and Down components respectively (in m/s).
        :type velocity: numpy.ndarray (3-elements)

        :param acceleration: The initial estimated acceleration vector for
            the North, East and Down components respectively (in m/s^2).
        :type acceleration: numpy.ndarray (3-elements)

        :param attitude: The initial estimated attitude vector for the
            Heading, Pitch and Roll components respectively (in degrees).
        :type attitude: numpy.ndarray (3-elements)

        :param angle_rates: The initial estimated angle rate vector for
            the P, Q and R body rates respectively (in deg/s).
        :type angle_rates: numpy.ndarray (3-elements)

        :param altimeter_gain: The gain amount to use for fusing altimeter
            measurements with the current altitude estimate (default=0.1).
        :type altimeter_gain: float
        """

        # Initialise the start position estimate.
        self._est_position = position
        self._est_velocity = velocity
        self._est_acceleration = acceleration
        self._est_attitude = attitude
        self._est_angle_rates = angle_rates

        # Get the given gravity model instance.
        self._gravity_model = gravity_model

        # Initialise the sensor and lever angles.
        # TODO: Set these in constructor?
        self._sensor_angles_rad = np.radians(np.array([0, 0, 0]))
        self._lever_arm = np.array([0, 0, 0])

        # Set the altimeter gain amounts.
        self._altimeter_gain = altimeter_gain

        # Introduced for compatability (not used).
        self._state_errors = np.eye(15) * 1e-9

    def take_inertia_measurement(self, time_step: float,
                                 measured_acceleration: np.ndarray,
                                 measured_angle_rates: np.ndarray):
        """
        Processes given acceleration and angle rate measurements and uses
        them to update the internal estimation states.

        :param time_step: The timestep since the last measurement was
            provided (in sqrt seconds)
        :type time_step: float

        :param measured_acceleration: The measured acceleration vector for
            the North, East and Down components respectively (in m/s^2).
        :type measured_acceleration: numpy.ndarray (3-elements)

        :param measured_angle_rates: The measured angle rate vector for
            the P, Q and R body rates respectively (in deg/s).
        :type measured_angle_rates: numpy.ndarray (3-elements)
        """

        # Calculate the estimated gravity vector
        g = self._gravity_model.calc_gravity_xyz(*self._est_position)

        # Define Angular velocity for Earth's rotation (in local NED axes)
        lat_rad = math.radians(float(self._est_position[0]))
        omega_e = const.OMEGA_E * np.array([math.cos(lat_rad), 0.0, -math.sin(lat_rad)])

        # Set up acceleration and angle rate measurements (Sensor axes)
        acceleration_s1 = measured_acceleration
        angle_rate_s1 = np.radians(measured_angle_rates)

        # Convert measured acceleration and angle rates to Body axes
        angle_rate_b0 = np.radians(self._est_angle_rates)

        rot_body2sensor = trans.rotate_3d(*self._sensor_angles_rad)
        angle_rate_b1 = np.linalg.solve(rot_body2sensor, angle_rate_s1)
        acceleration_b1 = np.linalg.solve(rot_body2sensor, acceleration_s1) - cross_prod_xy(
            angle_rate_b0, cross_prod_xy(angle_rate_b0, self._lever_arm))

        # Calculate estimated body axes from Earth-oriented axes.
        attitude_0 = np.radians(self._est_attitude)
        psi_0, theta_0, phi_0 = attitude_0
        rot_earth2body_0 = trans.rotate_3d(*attitude_0)

        # ------------------------------------------------------------------------

        # Convert position to local NED co-ordinates
        ref_lla = self._est_position
        position_e0 = trans.lla2ned(ref_lla, ref_lla)

        # Convert velocity from body axes to Earth Axes
        velocity_b0 = self._est_velocity
        # velocity_e0 = np.linalg.lstsq(rot_earth2body_0, velocity_b0, rcond=None)[0]
        velocity_e0 = np.linalg.solve(rot_earth2body_0, velocity_b0)

        # Convert measured acceleration from body axes to Earth axes
        # and remove Coriolis and gravity terms
        acceleration_e1 = np.linalg.solve(rot_earth2body_0, acceleration_b1) + g

        omega_tr = trans.get_transport_rate(ref_lla, velocity_e0)
        acceleration_e1 -= 2.0 * cross_prod_xy(omega_e, velocity_e0)
        acceleration_e1 -= cross_prod_xy(omega_tr, velocity_e0)  # NEW

        position_e1 = position_e0 + velocity_e0 * time_step \
            + 0.5 * acceleration_e1 * time_step ** 2

        # Velocity increments (calculated in Earth axes)
        velocity_e1 = velocity_e0 + acceleration_e1 * time_step
        angle_rate_b1 -= rot_earth2body_0 @ omega_e
        angle_rate_b1 -= rot_earth2body_0 @ omega_tr
        # angle_rate_b1 += rot_earth2body_0 @ omega_tr   # NEW

        d_theta_dt_1 = angle_rate_b1[1] * math.cos(phi_0) - angle_rate_b1[2] * math.sin(phi_0)
        d_psi_dt_1 = angle_rate_b1[1] * math.sin(phi_0) / math.cos(theta_0) + \
            angle_rate_b1[2] * math.cos(phi_0) / math.cos(theta_0)
        d_phi_dt_1 = angle_rate_b1[0] + d_psi_dt_1 * math.sin(theta_0)

        # Attitudes
        d_psi_1 = d_psi_dt_1 * time_step
        d_theta_1 = d_theta_dt_1 * time_step
        d_phi_1 = d_phi_dt_1 * time_step

        # Add attitude angle updates
        pi = math.pi
        attitude_b1 = attitude_0 + np.array([d_psi_1, d_theta_1, d_phi_1])
        attitude_b1 = np.remainder(attitude_b1 + pi, 2 * pi) - pi

        # Rotation matrix from Earth to body axes with updated rotation angles
        rot_earth2body_1 = trans.rotate_3d(*attitude_b1)

        # lat_rad = math.radians(position_e1[0])
        # g = get_gravity_xyz(position_e1[0], position_e1[2])
        # omega_e = const.OMEGA_E * np.array([math.cos(lat_rad), 0.0, -math.sin(lat_rad)])
        # omega_tr = trans.get_transport_rate(position_e1, velocity_e1)

        # Add effect of Coriolis term due to Earth's rotation (in Local NED axes)
        acceleration_e1 += 2.0 * cross_prod_xy(omega_e, velocity_e1)
        acceleration_e1 += cross_prod_xy(omega_tr, velocity_e1)

        # Convert Velocities from Earth Axes to Body Axes
        velocity_b1 = rot_earth2body_1.dot(velocity_e1)

        # Add gravity terms and convert acceleration from Earth axes to Body Axes
        acceleration_b1 = rot_earth2body_1.dot(acceleration_e1 - g)

        # Update angle rate to include Earth rotation rate
        # angle_rate_b1 += rot_earth2body_1.dot(omega_e)

        # Update the estimated variables
        self._est_position = trans.ned2lla(position_e1, ref_lla)
        self._est_velocity = velocity_b1
        self._est_acceleration = acceleration_b1
        self._est_attitude = np.degrees(attitude_b1)
        self._est_angle_rates = np.degrees(angle_rate_b1)

    def take_altimeter_measurement(self, measured_altitude: float):
        """
        Processes given altitude measurement and uses it to update the
        internal estimation states.

        :param measured_altitude: The measured altitude value (in metres
            above sea level) to fuse with the current estimated altitude.
        :type measured_altitude: float
        """
        self._est_position[2] = \
            (1 - self._altimeter_gain) * self._est_position[2] \
            + self._altimeter_gain * measured_altitude

    def set_altimeter_gain(self, new_gain: float):
        """
        Updates the gain amount for processing altimeter measurements.

        :param new_gain: The new gain amount (between 0 and 1) for fusing
            given altimeter measurements with the current estimated altitude.
        :type new_gain: float
        """
        assert 0 <= new_gain <= 1, \
            "Altimeter gain should be between 0.0 and 1.0"
        self._altimeter_gain = new_gain

    def get_estimate(self) -> tuple:
        """
        Returns the current estimated states as a tuple.

        :return: A tuple containing the estimated position, velocity,
            acceleration, attitude and angle rates respectively.
        :rtype: tuple
        """
        return (
            self._est_position,
            self._est_velocity,
            self._est_acceleration,
            self._est_attitude,
            self._est_angle_rates
        )

    def get_estimate_array(self) -> np.ndarray:
        """
        Returns the current estimated states as an array.

        :return: An array containing the estimated position, velocity,
            acceleration, attitude and angle rates respectively.
        :rtype: numpy.ndarray
        """
        return np.concatenate(self.get_estimate())

    def overwrite_estimates(self, position: np.ndarray = None,
                            velocity: np.ndarray = None, acceleration: np.ndarray = None,
                            attitude: np.ndarray = None, angle_rates: np.ndarray = None):
        """
        Overwrites selected current estimated states.
        This is namely used when estimation fusion is applied externally.

        :param position: The replacement position vector containing the
            latitude (in degrees), longitude (in degrees) and altitude
            (in metres) respectively.
        :type position: numpy.ndarray (3-elements)

        :param velocity: The replacement velocity vector for the
            North, East and Down components respectively (in m/s).
        :type velocity: numpy.ndarray (3-elements)

        :param acceleration: The replacement acceleration vector for
            the North, East and Down components respectively (in m/s^2).
        :type acceleration: numpy.ndarray (3-elements)

        :param attitude: The replacement attitude vector for the Heading,
            Pitch and Roll components respectively (in degrees).
        :type attitude: numpy.ndarray (3-elements)

        :param angle_rates: The replacement angle rate vector for the
            P, Q and R body rates respectively (in deg/s).
        :type angle_rates: numpy.ndarray (3-elements)
        """

        if position is not None:
            self._est_position = position

        if velocity is not None:
            self._est_velocity = velocity

        if acceleration is not None:
            self._est_acceleration = acceleration

        if attitude is not None:
            self._est_attitude = attitude

        if angle_rates is not None:
            self._est_angle_rates = angle_rates

    def get_state_errors(self) -> np.ndarray:
        """
        Returns the current state error matrix.

        :return: The current state error matrix.
        :rtype: numpy.ndarray (15-by-15 elements)
        """
        return self._state_errors

    def overwrite_state_errors(self, errors: np.ndarray):
        """
        Overwrites the current state error matrix.
        This is namely used when estimation fusion is applied externally.

        :param errors: The updated state error matrix.
        :type errors: numpy.ndarray (15-by-15 elements)
        """
        assert errors.shape == (15, 15), \
            "State error matrix must be of size 15-by-15!"
        self._state_errors = errors

    def get_sensor_angles(self) -> np.ndarray:
        """
        Returns the angles of the sensor placement.

        :return: The angles of the sensor placement in radians.
        :rtype: np.ndarray (3-elements)
        """
        return np.array(self._sensor_angles_rad)

    def get_lever_angles(self) -> np.ndarray:
        """
        Returns the angles of the lever arm.

        :return: The angles for the lever arm rotation.
        :rtype: np.ndarray (3-elements)
        """
        return np.array(self._lever_arm)

    # def as_dict(self) -> dict:
    #     return {
    #         'position': self._est_position,
    #         'velocity': self._est_velocity,
    #         'acceleration': self._est_acceleration,
    #         'attitude': self._est_attitude,
    #         'angle_rates': self._est_angle_rates,
    #         'gravity_model': self._gravity_model
    #         'altimeter_gain': self._altimeter_gain,
    #         'sensor_angles_rad': self._sensor_angles_rad,
    #         'lever_angles': self._lever_angles,
    #         'state_errors': self._state_errors,
    #     }
    #
    # @staticmethod
    # def from_dict(self, ins_dict: dict):
    #
    #     try:
    #         ins = StandardINS(**ins_dict)
    #
    #
    #
    #     except KeyError:
    #         raise KeyError("No state error matrix was found.")
