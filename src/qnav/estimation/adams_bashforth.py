"""
==================
adams_bashforth.py
==================

:summary:
    An alternative INS implementation that uses IMU Data from the dynamics to
    generate new navigation solution by employing the Adams-Bashforth
    integration method.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import qnav.util.transformations as trans
import qnav.util.constants as const
import numpy as np
import math

from qnav.gravity.base import GravityModel
from qnav.gravity.simple import SimpleUniform
from qnav.estimation.ins import StandardINS
from math import cos, sin, pi


class AdamsBashforth(StandardINS):
    """
    Processes the measurement vector and updates the estimation states using
    Adams-Bashforth integration.

    Does include:
        * Simple Accelerometer Measurement Model (basic - independent
          measurement errors, giving drift term)
        * 3 x Individual Accelerator Bias values (fixed)
        * 3 x Individual Accelerator Scaling errors (fixed)
        * 3D Accelerometer non-orthogonality errors (fixed) for cross-coupling
        * Simple (not position dependent) Gravity Compensation
        * Simple Gyroscope Measurement Model (basic - independent
          measurement errors, giving drift term)
        * 3 x Individual Gyroscope Bias values (fixed)
        * 3 x Individual Gyroscope Scaling errors (fixed)
        * 3D Gyroscope non-orthogonality errors (fixed) for cross-coupling
        * Sensor bias drift for accelerometers and gyroscopes
        * Kalman Filtering of measurement signals
        * WGS'84 coordinates (ellipsoidal rotating Earth)
        * Effect of Coriolis effect due to Earth's rotation

    Does NOT currently include:
        * Schuler correction loop.
        * Physics-based accelerometer sensor model
        * Physics-based gyroscope sensor model
        * Lever-arm effects from rotations not around origin/centre of the IMU
    """

    # All added properties of the class
    __slots__ = \
        '_d_est_position', \
        '_d_est_velocity', \
        '_d_est_acceleration', \
        '_d_est_attitude', \
        '_d_est_angle_rates', \
        '__requires_bootstrap'

    def __init__(self, position: np.ndarray,
                 velocity: np.ndarray,
                 acceleration: np.ndarray,
                 attitude: np.ndarray,
                 angle_rates: np.ndarray,
                 gravity_model: GravityModel = SimpleUniform,
                 altimeter_gain: float = 0.1):
        """
        Initialises the Adams Bashforth based on initial estimates.

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

        # Call the parent class's constructor.
        super().__init__(position, velocity, acceleration,
                         attitude, angle_rates,
                         gravity_model, altimeter_gain)

        # Set the estimation differences
        self._d_est_position = np.zeros(3)
        self._d_est_velocity = np.zeros(3)
        self._d_est_acceleration = np.zeros(3)
        self._d_est_attitude = np.zeros(3)
        self._d_est_angle_rates = np.zeros(3)

        # Set that bootstrapping is required.
        self.__requires_bootstrap = True

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

        # Use bootstrapping for the first run
        if self.__requires_bootstrap:
            self.__perform_bootstrapping(
                time_step, measured_acceleration, measured_angle_rates)
            self.__requires_bootstrap = False
            return

        # Calculate the estimated gravity vector
        g = self._gravity_model.calc_gravity_xyz(*self._est_position)

        lat_rad = trans.deg2rad(float(self._est_position[0]))
        omega_e = const.OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

        # Set up acceleration and angle rate measurements (Sensor axes)
        acceleration_s0 = measured_acceleration
        angle_rate_s0 = np.radians(measured_angle_rates)

        # Get lever arm and sensor angles in radians.
        lever_arm = self._lever_arm
        sensor_angles = self._sensor_angles_rad

        # Convert measured angle rates to Body axes
        rot_body2sensor = trans.rotate_3d(*sensor_angles)
        angle_rate_bs0 = np.linalg.solve(rot_body2sensor, angle_rate_s0)

        # Convert measured acceleration to Body axes
        # TODO: Use tuple functions instead
        acceleration_bs0 = np.linalg.solve(
            rot_body2sensor, acceleration_s0) - np.cross(
            angle_rate_bs0, np.cross(angle_rate_bs0, lever_arm))

        # Fix the reference Lat-Long-Altitude location
        ref_lla = self._est_position

        # Euler angles (in radians)
        attitude_0 = np.radians(self._est_attitude)
        angle_rate_b0 = np.radians(self._est_angle_rates)

        # Convert position to local NED co-ordinates
        position_e0 = np.zeros(3)

        # Obtain the body velocity and acceleration
        # Obtain the body velocity and acceleration
        velocity_b0 = self._est_velocity
        acceleration_b0 = self._est_acceleration

        # Adams-Bashforth
        # ---------------

        # Calculate ESTIMATED body axes from Earth-oriented axes.
        rot_e2b_0 = trans.rotate_3d(*attitude_0)

        # Convert velocity from body axes to Earth Axes
        velocity_e0 = np.linalg.solve(rot_e2b_0, velocity_b0)
        omega_tr = trans.get_transport_rate(ref_lla, velocity_e0)

        # Convert measured acceleration from body axes to Earth axes and remove gravity terms
        acceleration_e1 = np.linalg.solve(rot_e2b_0, acceleration_bs0) + g

        # Remove effect of Coriolis term due to Earth's rotation and transport rate (in Local NED axes)
        acceleration_e1 -= 2.0 * np.cross(omega_e, velocity_e0)
        acceleration_e1 -= np.cross(omega_tr, velocity_e0)

        # Position increments in local NED/Earth axes
        position_e1 = position_e0 + velocity_e0 * time_step

        # Velocity increments (calculated in Earth axes)
        velocity_e1 = velocity_e0 + acceleration_e1 * time_step

        # angle_rate_b1 = angle_rate_bs0 - rot_e2b_0 @ omega_e + rot_e2b_0 @ omega_tr
        angle_rate_b1 = angle_rate_bs0 - rot_e2b_0 @ omega_e - rot_e2b_0 @ omega_tr

        # Angle Rates
        d_psi_dt = angle_rate_b1[1] * math.sin(attitude_0[2]) / math.cos(attitude_0[1]) + \
            angle_rate_b1[2] * math.cos(attitude_0[2]) / math.cos(attitude_0[1])

        d_theta_dt = angle_rate_b1[1] * math.cos(attitude_0[2]) - \
            angle_rate_b1[2] * math.sin(attitude_0[2])

        d_phi_dt = angle_rate_b1[0] + angle_rate_b1[1] * \
            math.sin(attitude_0[2]) * math.tan(attitude_0[1]) + \
            angle_rate_b1[2] * math.cos(attitude_0[2]) * math.tan(attitude_0[1])

        # Add attitude angle updates
        attitude_1 = attitude_0 + np.array([d_psi_dt, d_theta_dt, d_phi_dt]) * time_step

        # Rotation matrix from Earth to body axes with NEW (updated) rotation angles
        rot_e2b_1 = trans.rotate_3d(*attitude_1)

        # Add effect of Coriolis term due to Earth's rotation (in Local NED axes)
        acceleration_e1 = acceleration_e1 + 2.0 * np.cross(omega_e, velocity_e1)

        # Convert Velocities from Earth Axes to Body Axes
        velocity_b1 = rot_e2b_1 @ velocity_e1

        # Add gravity terms and convert acceleration from Earth axes to Body Axes
        acceleration_b1 = rot_e2b_1 @ (acceleration_e1 - g)

        # Update angle rate to include Earth rotation rate
        omega_tr = trans.get_transport_rate(ref_lla, velocity_e1)
        angle_rate_b1 = angle_rate_b1 + rot_e2b_1 @ omega_e - rot_e2b_1 @ omega_tr

        # Adams-Bashforth increments
        d_position_e1 = position_e1 - position_e0
        d_velocity_b1 = velocity_b1 - velocity_b0
        d_acceleration_b1 = acceleration_b1 - acceleration_b0
        d_attitude_1 = attitude_1 - attitude_0
        d_angle_rate_b1 = angle_rate_b1 - angle_rate_b0

        d_position_e0 = self._d_est_position
        d_velocity_b0 = self._d_est_velocity
        d_acceleration_b0 = self._d_est_acceleration
        d_attitude_0 = self._d_est_attitude
        d_angle_rate_b0 = self._d_est_angle_rates

        # Sum together all Adams-Bashforth terms
        position_e2 = position_e0 + 1.5 * d_position_e1 - 0.5 * d_position_e0
        velocity_b2 = velocity_b0 + 1.5 * d_velocity_b1 - 0.5 * d_velocity_b0
        acceleration_b2 = acceleration_b0 + 1.5 * d_acceleration_b1 - 0.5 * d_acceleration_b0
        attitude_2 = attitude_0 + 1.5 * d_attitude_1 - 0.5 * d_attitude_0
        angle_rate_b2 = angle_rate_b0 + 1.5 * d_angle_rate_b1 - 0.5 * d_angle_rate_b0

        # Ensure the updated attitude values are valid
        attitude_2 = np.remainder(attitude_2 + pi, 2 * pi) - pi

        # Update the estimated variables
        self._est_position = trans.ned2lla(position_e2, ref_lla)
        self._est_velocity = velocity_b2
        self._est_acceleration = acceleration_b2
        self._est_attitude = np.degrees(attitude_2)
        self._est_angle_rates = np.degrees(angle_rate_b2)

        # Convert position from local NED co-ordinates back to Lat-Long-Altitude
        self._d_est_position = d_position_e1
        self._d_est_velocity = d_velocity_b1
        self._d_est_acceleration = d_acceleration_b1
        self._d_est_attitude = d_attitude_1
        self._d_est_angle_rates = d_angle_rate_b1

    def __perform_bootstrapping(self, time_step: float,
                                measured_acceleration: np.ndarray,
                                measured_angle_rates: np.ndarray):
        """
        An alternative processing method for the first run.
        If previous estimates are not available, this method can be used
        to bootstrap the difference in estimation values.

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

        # Obtain the current estimated position.
        ref_lla = self._est_position

        # Obtain the current estimation values.
        position_e0 = np.zeros(3)
        velocity_b0 = np.copy(self._est_velocity)
        acceleration_b0 = np.copy(self._est_acceleration)
        attitude_0 = np.radians(self._est_attitude)
        angle_rate_b0 = np.radians(self._est_attitude)

        # Use the parent class's update method.
        super().take_inertia_measurement(
            time_step, measured_acceleration, measured_angle_rates)

        # Obtain the updated estimation values.
        position_e1 = trans.lla2ned(self._est_position, ref_lla)
        velocity_b1 = np.copy(self._est_velocity)
        acceleration_b1 = np.copy(self._est_acceleration)
        attitude_1 = np.radians(self._est_attitude)
        angle_rate_b1 = np.radians(self._est_attitude)

        # Finally, initialise the estimation differences.
        self._d_est_position = position_e1 - position_e0
        self._d_est_velocity = velocity_b1 - velocity_b0
        self._d_est_acceleration = acceleration_b1 - acceleration_b0
        self._d_est_attitude = attitude_1 - attitude_0
        self._d_est_angle_rates = angle_rate_b1 - angle_rate_b0
