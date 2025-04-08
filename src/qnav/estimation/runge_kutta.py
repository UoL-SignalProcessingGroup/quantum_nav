"""
==============
runge_kutta.py
==============

:summary:
    An alternative INS implementation that uses IMU Data from the dynamics to
    generate new navigation solution by employing the Runge Kutta integration
    (4th order) method.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import qnav.util.transformations as trans
import qnav.util.constants as const
import numpy as np

# from qnav.util.transformations import cross_prod3
from qnav.util.transformations import cross_prod_xy
from qnav.estimation.ins import StandardINS
from math import cos, sin, tan, pi


class RungeKutta(StandardINS):
    """
    Processes the measurement vector and updates the estimation states using
    Runge Kutta (4th order) integration.

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

        # Define Angular velocity for Earth's rotation (in local NED axes)
        lat_rad = trans.deg2rad(float( self._est_position[0]))
        omega_e = const.OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

        # Calculate the gravity vector
        g = self._gravity_model.calc_gravity_xyz(*self._est_position)

        # Set up acceleration and angle rate measurements (Sensor axes)
        acceleration_s0 = measured_acceleration
        angle_rate_s0 = np.radians(measured_angle_rates)

        # Get lever arm and sensor angles in radians.
        lever_arm = self._lever_arm
        sensor_angles = self._sensor_angles_rad

        # Convert measured angle rates to Body axes
        rot_body2sensor = trans.rotate_3d(*sensor_angles)
        angle_rate_bs0 = np.linalg.solve(rot_body2sensor, angle_rate_s0)

        # TODO: Use angle_rate_s0 instead of angle_rate_bs0?

        # Convert measured acceleration to Body axes
        # acceleration_bs0 = np.linalg.solve(
        #     rot_body2sensor, acceleration_s0) - np.array(cross_prod3(
        #         angle_rate_bs0, np.array(cross_prod3(angle_rate_bs0, lever_arm))))

        acceleration_bs0 = np.linalg.solve(
            rot_body2sensor, acceleration_s0) - np.array(cross_prod_xy(
            angle_rate_bs0, np.array(cross_prod_xy(angle_rate_bs0, lever_arm))))

        # Fix the reference Lat-Long-Altitude location
        ref_lla = self._est_position

        # Euler angles (in radians)
        attitude_0 = np.radians(self._est_attitude)
        angle_rate_b0 = np.radians(self._est_angle_rates)

        # Convert position to local NED co-ordinates
        # position_e0 = trans.lla2ned(position, position)
        position_e0 = np.zeros(3)

        # Obtain the body velocity and acceleration
        velocity_b0 = np.copy(self._est_velocity)
        acceleration_b0 = np.copy(self._est_acceleration)

        # Runge-Kutta Setup
        # -----------------

        # The number of steps to repeat
        order_number: int = 4

        # Create arrays to stops the k terms
        k_position_e = np.zeros([order_number, 3])
        k_velocity_b = np.zeros([order_number, 3])
        k_acceleration_b = np.zeros([order_number, 3])
        k_attitude = np.zeros([order_number, 3])
        k_angle_rate_b = np.zeros([order_number, 3])

        # Record the previous estimation values to be updated
        updated_position = position_e0
        updated_velocity = velocity_b0
        updated_acceleration = acceleration_b0
        updated_attitude = attitude_0
        updated_angle_rates = angle_rate_b0

        # For each of the k steps:
        for i in range(order_number):

            # Set the increment to calculate the derivatives
            if i != 2:
                d_time_step = (0.5 * time_step)
            else:
                d_time_step = time_step

            # Calculate ESTIMATED body axes from Earth-oriented axes
            rot_earth2body_0 = trans.rotate_3d(*attitude_0)

            # Calculate the body axes from Earth-oriented axes
            velocity_e0 = np.linalg.solve(rot_earth2body_0, velocity_b0)
            # velocity_e0 = np.linalg.lstsq(rot_earth2body_0, velocity_b0, rcond=None)[0]

            # Calculate the transport rate
            omega_tr = trans.get_transport_rate(ref_lla, velocity_e0)

            # Convert measured acceleration from body axes to Earth axes
            # and remove Coriolis and gravity terms
            acceleration_e1 = np.linalg.solve(rot_earth2body_0, acceleration_bs0) + g
            # acceleration_e1 = np.linalg.lstsq(rot_earth2body_0, acceleration_bs0, rcond=None)[0] + g

            # Remove effect of Coriolis term due to Earth's rotation and transport rate (in Local NED axes)
            acceleration_e1 -= 2.0 * np.cross(omega_e, velocity_e0)
            acceleration_e1 -= np.cross(omega_tr, velocity_e0)

            # Position increments in local NED/Earth axes
            position_e1 = position_e0 + velocity_e0 * d_time_step

            # Velocity increments (calculated in Earth axes)
            velocity_e1 = velocity_e0 + acceleration_e1 * d_time_step
            # angle_rate_b1 = angle_rate_bs0 - rot_earth2body_0 @ omega_e + rot_earth2body_0 @ omega_tr
            angle_rate_b1 = angle_rate_bs0 - rot_earth2body_0 @ omega_e - rot_earth2body_0 @ omega_tr  # <<<< NEW

            # Calculate the updated angle rates
            d_psi_dt = angle_rate_b1[1] * sin(attitude_0[2]) / cos(attitude_0[1]) + \
                angle_rate_b1[2] * cos(attitude_0[2]) / cos(attitude_0[1])

            d_theta_dt = angle_rate_b1[1] * cos(attitude_0[2]) - angle_rate_b1[2] * sin(attitude_0[2])

            d_phi_dt = angle_rate_b1[0] + angle_rate_b1[1] * sin(attitude_0[2]) * tan(attitude_0[1]) + \
                angle_rate_b1[2] * cos(attitude_0[2]) * tan(attitude_0[1])

            # Add attitude angle updates
            attitude_1 = attitude_0 + np.array([d_psi_dt, d_theta_dt, d_phi_dt]) * d_time_step

            # Rotation matrix from Earth to body axes with NEW (updated) rotation angles
            rot_earth2body_1 = trans.rotate_3d(*attitude_1)

            # Add effect of Coriolis term due to Earth's rotation (in Local NED axes)
            acceleration_e1 += 2.0 * np.cross(omega_e, velocity_e1)

            # Convert Velocities from Earth Axes to Body Axes
            velocity_b1 = rot_earth2body_1 @ velocity_e1

            # Add gravity terms and convert acceleration from Earth axes to Body Axes
            acceleration_b1 = rot_earth2body_1 @ (acceleration_e1 - g)

            # Update angle rate to include Earth rotation rate
            omega_tr = trans.get_transport_rate(ref_lla, velocity_e1)
            angle_rate_b1 += rot_earth2body_1 @ omega_e - rot_earth2body_1 @ omega_tr

            # Runge_Kutta values
            k_position_e[i, :] = (position_e1 - position_e0) / d_time_step
            k_velocity_b[i, :] = (velocity_b1 - velocity_b0) / d_time_step
            k_acceleration_b[i, :] = (acceleration_b1 - acceleration_b0) / d_time_step
            k_attitude[i, :] = (attitude_1 - attitude_0) / d_time_step
            k_angle_rate_b[i, :] = (angle_rate_b1 - angle_rate_b0) / d_time_step

            # Repeat the loop using the updated values
            position_e0 = position_e1
            velocity_b0 = velocity_b1
            attitude_0 = attitude_1

        # Assign weightings to each of the mid-points
        k_weightings = np.array([1, 2, 2, 1]).reshape(-1, 1)

        # Pre-calculate the time step multiplier
        multiply_amount = time_step / 6.0

        # Sum together all Runge-Kutta terms with their assigned weightings
        updated_position += np.sum(k_position_e * k_weightings, axis=0) * multiply_amount
        updated_velocity += np.sum(k_velocity_b * k_weightings, axis=0) * multiply_amount
        updated_acceleration += np.sum(k_acceleration_b * k_weightings, axis=0) * multiply_amount
        updated_attitude += np.sum(k_attitude * k_weightings, axis=0) * multiply_amount
        updated_angle_rates += np.sum(k_angle_rate_b * k_weightings, axis=0) * multiply_amount

        # Ensure the updated attitude values are valid
        updated_attitude = np.remainder(updated_attitude + pi, 2 * pi) - pi

        # Update the estimated variables
        self._est_position = trans.ned2lla(updated_position, ref_lla)
        self._est_velocity = updated_velocity
        self._est_acceleration = updated_acceleration
        self._est_attitude = np.degrees(updated_attitude)
        self._est_angle_rates = np.degrees(updated_angle_rates)
