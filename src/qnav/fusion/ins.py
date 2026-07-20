"""
======
ins.py
======

:summary:
    Fusion method for supplying Inertial Navigation.
    Provides a selection of standard sensor fusion methods for means of basic
    internal navigation. Each of these takes measurements from accelerometer
    and gyroscope sensors (IMU).

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""


from qnav.estimation.state import EstimatedState
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.platform import SensorAxis
from qnav.measurement.sensor import SensorFusion
from qnav.measurement.sensor import FusionTrigger
from qnav.measurement.sensor import resolve_time_step
from qnav.util.transformations import cross_prod_xy
# from qnav.util.transformations import cross_prod3
from math import radians, sin, cos, tan, pi
from warnings import warn

import qnav.util.transformations as trans
import qnav.util.constants as const
import numpy as np
import math


def _adams_bashforth_weights(time_step: float,
                             previous_time_step: float) -> tuple[float, float]:
    """Return AB2 weights for derivative increments at unequal intervals."""
    step_ratio = time_step / previous_time_step
    return 1.0 + 0.5 * step_ratio, -0.5 * step_ratio ** 2


class NumericalINS(SensorFusion):
    """
    A fairly basic simple numerical solution for Internal Navigation (INS).
    This INS solution is based on simple numerical integration. While its
    estimation calculations are relatively primitive, it serves as a base
    model that other solutions can extend from. The simple implementation
    also provides straight-forward means for testing calculations.
    """

    def __init__(self, accelerometer: Accelerometer,
                 gyroscope: Gyroscope,
                 estimated_acc_axis: SensorAxis = None,
                 estimated_gyro_axis: SensorAxis = None):
        """
        Create a simple numerical solution for internal Navigation (INS).
        Processes periodic measurements from given accelerometer and
        gyroscopes to update estimated state.

        :param accelerometer: An accelerometer instance used to capture
            acceleration values during simulation.
        :type accelerometer: Accelerometer

        :param gyroscope: A gyroscope instance used to capture
            angle rate acceleration values during simulation.
        :type gyroscope: Gyroscope

        :param estimated_acc_axis: (Optional) The estimated axis for the
            accelerometer instance. If not provided, the actual
            accelerometer's axis will be used, resulting in zero misalignment.
        :type estimated_acc_axis: SensorAxis


        :param estimated_gyro_axis: (Optional) The estimated axis for the
            gyroscope instance. If not provided, the actual gyroscope's
            axis will be used, resulting in zero misalignment.
        :type estimated_acc_axis: SensorAxis
        """

        # Call parent class's constructor:
        super().__init__(FusionTrigger.ALL, accelerometer, gyroscope)

        # Copy given required arguments
        self._accelerometer = accelerometer
        self._gyroscope = gyroscope

        # Get the axis to use for the accelerometer
        if estimated_acc_axis is None:
            self._acc_axis = accelerometer.sensor_axis
        else:
            self._acc_axis = estimated_acc_axis

        # Get the axis to use for the accelerometer
        if estimated_gyro_axis is None:
            self._gyro_axis = gyroscope.sensor_axis
        else:
            self._gyro_axis = estimated_gyro_axis

        # Ensure that the sensor frequencies are compatible
        if self._accelerometer.frequency != self._gyroscope.frequency:
            warn("Accelerometer and Gyroscope have different measurement frequencies")


    def perform_fusion(self, estimated_state: EstimatedState,
                       time_step: float = None) -> None:
        """
        Performs fusion using latest accelerometer and gyroscope measurements.
        Using the latest basic inertial measurements, updates the full current
        estimated state using a simple numerical model.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState

        :param time_step: Optional elapsed interval in seconds. When omitted,
            each sensor's configured interval is used.
        :type time_step: float
        """

        # Get the latest measurement from the sensors:
        measured_acceleration = self._accelerometer.last_measurement
        measured_angle_rates = self._gyroscope.last_measurement

        # Explicit replay intervals apply to both measurements. Simulation
        # callers retain the independently configured sensor intervals.
        acc_time_step = resolve_time_step(
            time_step, self._accelerometer.time_step)
        gyro_time_step = resolve_time_step(
            time_step, self._gyroscope.time_step)

        # Unpack old estimated states
        est_position = estimated_state.position
        est_attitude = estimated_state.attitude
        est_angle_rates = estimated_state.angle_rates

        # ------------------------------------------------------------------------

        # Unpack required sensor axis values:
        acc_lever_arm = self._acc_axis.lever_arm
        acc_rot_body2sensor = self._acc_axis.body2sensor_mat
        gyro_rot_body2sensor = self._gyro_axis.body2sensor_mat

        # Calculate the estimated gravity vector:
        gravity_model = estimated_state.gravity_model
        g = gravity_model.calc_gravity_xyz(*est_position)

        # Define Angular velocity for Earth's rotation (in local NED axes)
        lat_rad = radians(est_position[0])
        omega_e = const.OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

        # Set up acceleration and angle rate measurements (Sensor axes)
        acceleration_s1 = measured_acceleration
        angle_rate_s1 = np.radians(measured_angle_rates)

        # Convert measured acceleration and angle rates to Body axes
        angle_rate_b0 = np.radians(est_angle_rates)

        # rot_body2sensor = trans.rotate_3d(*sensor_angles_rad)
        angle_rate_b1 = np.linalg.solve(gyro_rot_body2sensor, angle_rate_s1)
        acceleration_b1 = np.linalg.solve(acc_rot_body2sensor, acceleration_s1) - cross_prod_xy(
            angle_rate_b0, cross_prod_xy(angle_rate_b0, acc_lever_arm))

        # Calculate estimated body axes from Earth-oriented axes.
        attitude_0 = np.radians(est_attitude)
        psi_0, theta_0, phi_0 = attitude_0
        rot_earth2body_0 = trans.rotate_3d(*attitude_0)

        # ------------------------------------------------------------------------

        # Convert position to local NED co-ordinates
        ref_lla = est_position
        position_e0 = trans.lla2ned(ref_lla, ref_lla)

        # Convert velocity from body axes to Earth Axes
        velocity_b0 = estimated_state.velocity
        velocity_e0 = np.linalg.solve(rot_earth2body_0, velocity_b0)

        # Convert measured acceleration from body axes to Earth axes
        # and remove Coriolis and gravity terms
        acceleration_e1 = np.linalg.solve(rot_earth2body_0, acceleration_b1) + g

        omega_tr = trans.get_transport_rate(ref_lla, velocity_e0)
        acceleration_e1 -= 2.0 * cross_prod_xy(omega_e, velocity_e0)
        acceleration_e1 -= cross_prod_xy(omega_tr, velocity_e0)  # NEW

        position_e1 = position_e0 + velocity_e0 * acc_time_step \
                      + 0.5 * acceleration_e1 * acc_time_step ** 2

        # Velocity increments (calculated in Earth axes)
        velocity_e1 = velocity_e0 + acceleration_e1 * acc_time_step
        angle_rate_b1 -= rot_earth2body_0 @ omega_e
        angle_rate_b1 -= rot_earth2body_0 @ omega_tr

        d_theta_dt_1 = angle_rate_b1[1] * cos(phi_0) - angle_rate_b1[2] * sin(phi_0)
        d_psi_dt_1 = angle_rate_b1[1] * sin(phi_0) / cos(theta_0) + \
                     angle_rate_b1[2] * cos(phi_0) / cos(theta_0)
        d_phi_dt_1 = angle_rate_b1[0] + d_psi_dt_1 * sin(theta_0)

        # Attitudes
        d_psi_1 = d_psi_dt_1 * gyro_time_step
        d_theta_1 = d_theta_dt_1 * gyro_time_step
        d_phi_1 = d_phi_dt_1 * gyro_time_step

        # Add attitude angle updates
        attitude_b1 = attitude_0 + np.array([d_psi_1, d_theta_1, d_phi_1])
        attitude_b1 = np.remainder(attitude_b1 + pi, 2 * pi) - pi

        # Rotation matrix from Earth to body axes with updated rotation angles
        rot_earth2body_1 = trans.rotate_3d(*attitude_b1)

        # Add effect of Coriolis term due to Earth's rotation (in Local NED axes)
        acceleration_e1 += 2.0 * cross_prod_xy(omega_e, velocity_e1)
        acceleration_e1 += cross_prod_xy(omega_tr, velocity_e1)

        # Convert Velocities from Earth Axes to Body Axes
        velocity_b1 = rot_earth2body_1.dot(velocity_e1)

        # Add gravity terms and convert acceleration from Earth axes to Body Axes
        acceleration_b1 = rot_earth2body_1.dot(acceleration_e1 - g)

        # Update the estimated variables
        estimated_state.update_estimates(
            position=trans.ned2lla(position_e1, ref_lla),
            velocity=velocity_b1,
            acceleration=acceleration_b1,
            attitude=np.degrees(attitude_b1),
            angle_rates=np.degrees(angle_rate_b1)
        )


class RungeKutta(NumericalINS):
    """
    An extension of the numerical INS that employs Runge-Kutta methods.
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

    def perform_fusion(self, estimated_state: EstimatedState,
                       time_step: float = None) -> None:
        """
        Performs fusion using latest accelerometer and gyroscope measurements.
        Using the latest basic inertial measurements, updates the full current
        estimated state using a simple numerical model.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState

        :param time_step: Optional elapsed interval in seconds. When omitted,
            the mean configured IMU interval is used.
        :type time_step: float
        """

        # Get the latest measurement from the sensors:
        measured_acceleration = self._accelerometer.last_measurement
        measured_angle_rates = self._gyroscope.last_measurement

        # Get the timestep between measurements:
        configured_time_step = (
            self._accelerometer.time_step + self._gyroscope.time_step) / 2
        time_step = resolve_time_step(time_step, configured_time_step)

        # Define Angular velocity for Earth's rotation (in local NED axes)
        ref_lla = estimated_state.position
        lat_rad = math.radians(ref_lla[0])
        omega_e = const.OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

        # Calculate the gravity vector
        gravity_model = estimated_state.gravity_model
        g = gravity_model.calc_gravity_xyz(*ref_lla)

        # Set up acceleration and angle rate measurements (Sensor axes)
        acceleration_s0 = measured_acceleration
        angle_rate_s0 = np.radians(measured_angle_rates)

        # Unpack required sensor axis values:
        acc_lever_arm = self._acc_axis.lever_arm
        acc_rot_body2sensor = self._acc_axis.body2sensor_mat
        gyro_rot_body2sensor = self._gyro_axis.body2sensor_mat

        # Convert measured angle rates to Body axes
        # rot_body2sensor = trans.rotate_3d(*sensor_angles)
        angle_rate_bs0 = np.linalg.solve(gyro_rot_body2sensor, angle_rate_s0)

        # TODO: Use angle_rate_s0 instead of angle_rate_bs0?

        # Convert measured acceleration to Body axes
        # acceleration_bs0 = np.linalg.solve(
        #     acc_rot_body2sensor, acceleration_s0) - np.array(cross_prod3(
        #     angle_rate_bs0, np.array(cross_prod3(angle_rate_bs0, acc_lever_arm))))

        acceleration_bs0 = np.linalg.solve(
            acc_rot_body2sensor, acceleration_s0) - np.array(cross_prod_xy(
            angle_rate_bs0, np.array(cross_prod_xy(angle_rate_bs0, acc_lever_arm))))

        # Fix the reference Lat-Long-Altitude location
        # ref_lla = self._est_position

        # Euler angles (in radians)
        attitude_0 = np.radians(estimated_state.attitude)
        angle_rate_b0 = np.radians(estimated_state.angle_rates)

        # Convert position to local NED co-ordinates
        # position_e0 = trans.lla2ned(position, position)
        position_e0 = np.zeros(3)

        # Obtain the body velocity and acceleration
        velocity_b0 = np.copy(estimated_state.velocity)
        acceleration_b0 = np.copy(estimated_state.acceleration)

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
        estimated_state.update_estimates(
            position=trans.ned2lla(updated_position, ref_lla),
            velocity=updated_velocity,
            acceleration=updated_acceleration,
            attitude= np.degrees(updated_attitude),
            angle_rates=np.degrees(updated_angle_rates)
        )


class AdamsBashforth(NumericalINS):
    """
    An extension of the numerical INS that employs Adams-bashforth method.
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

    def __init__(self, accelerometer: Accelerometer, gyroscope: Gyroscope,
                 estimated_acc_axis: SensorAxis = None, estimated_gyro_axis: SensorAxis = None):

        # Call parent class's constructor
        super().__init__(accelerometer, gyroscope, estimated_acc_axis, estimated_gyro_axis)

        # Set the estimation differences
        self._d_est_position = np.zeros(3)
        self._d_est_velocity = np.zeros(3)
        self._d_est_acceleration = np.zeros(3)
        self._d_est_attitude = np.zeros(3)
        self._d_est_angle_rates = np.zeros(3)

        # Interval represented by the stored increments. This is required for
        # the variable-step Adams-Bashforth coefficients.
        self._previous_time_step = None

        # Set that bootstrapping is required.
        self.__requires_bootstrap = True

    def __perform_bootstrapping(self, estimated_state: EstimatedState,
                                time_step: float,
                                requested_time_step: float = None):
        """
        An alternative processing method for the first run.
        If previous estimates are not available, this method can be used
        to bootstrap the difference in estimation values by using the
        parent's classes original fusion method.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState
        """

        # Obtain the current estimated position.
        ref_lla = estimated_state.position

        # Obtain the current estimation values.
        position_e0 = np.zeros(3)
        velocity_b0 = np.copy(estimated_state.velocity)
        acceleration_b0 = np.copy(estimated_state.acceleration)
        attitude_0 = np.radians(estimated_state.attitude)
        angle_rate_b0 = np.radians(estimated_state.angle_rates)

        # Use the parent class's update method.
        # super().take_inertia_measurement(
        #     time_step, measured_acceleration, measured_angle_rates)
        # Passing None preserves NumericalINS's legacy handling of unequal
        # accelerometer and gyroscope frequencies when replay did not provide
        # an explicit interval.
        super().perform_fusion(estimated_state, requested_time_step)

        # Obtain the updated estimation values.
        position_e1 = trans.lla2ned(estimated_state.position, ref_lla)
        velocity_b1 = np.copy(estimated_state.velocity)
        acceleration_b1 = np.copy(estimated_state.acceleration)
        attitude_1 = np.radians(estimated_state.attitude)
        angle_rate_b1 = np.radians(estimated_state.angle_rates)

        # Finally, initialise the estimation differences.
        self._d_est_position = position_e1 - position_e0
        self._d_est_velocity = velocity_b1 - velocity_b0
        self._d_est_acceleration = acceleration_b1 - acceleration_b0
        self._d_est_attitude = attitude_1 - attitude_0
        self._d_est_angle_rates = angle_rate_b1 - angle_rate_b0
        self._previous_time_step = time_step

    def perform_fusion(self, estimated_state: EstimatedState,
                       time_step: float = None) -> None:
        """
        Performs fusion using latest accelerometer and gyroscope measurements.
        Using the latest basic inertial measurements, updates the full current
        estimated state using a simple numerical model.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState

        :param time_step: Optional elapsed interval in seconds. When omitted,
            the mean configured IMU interval is used.
        :type time_step: float
        """

        requested_time_step = time_step
        configured_time_step = (
            self._accelerometer.time_step + self._gyroscope.time_step) / 2
        time_step = resolve_time_step(time_step, configured_time_step)

        # Use bootstrapping for the first run
        if self.__requires_bootstrap:
            self.__perform_bootstrapping(
                estimated_state, time_step, requested_time_step)
            self.__requires_bootstrap = False
            return

        # Get the latest measurement from the sensors:
        measured_acceleration = self._accelerometer.last_measurement
        measured_angle_rates = self._gyroscope.last_measurement

        ref_lla = estimated_state.position

        # Calculate the estimated gravity vector
        gravity_model = estimated_state.gravity_model
        g = gravity_model.calc_gravity_xyz(*ref_lla)

        lat_rad = math.radians(ref_lla[0])
        omega_e = const.OMEGA_E * np.array([cos(lat_rad), 0.0, -sin(lat_rad)])

        # Set up acceleration and angle rate measurements (Sensor axes)
        acceleration_s0 = measured_acceleration
        angle_rate_s0 = np.radians(measured_angle_rates)

        # Unpack required sensor axis values:
        acc_lever_arm = self._acc_axis.lever_arm
        acc_rot_body2sensor = self._acc_axis.body2sensor_mat
        gyro_rot_body2sensor = self._gyro_axis.body2sensor_mat

        # Convert measured angle rates to Body axes
        angle_rate_bs0 = np.linalg.solve(gyro_rot_body2sensor, angle_rate_s0)

        # Convert measured acceleration to Body axes
        # TODO: Use tuple functions instead?
        acceleration_bs0 = np.linalg.solve(
            acc_rot_body2sensor, acceleration_s0) - np.cross(
            angle_rate_bs0, np.cross(angle_rate_bs0, acc_lever_arm))

        # Euler angles (in radians)
        attitude_0 = np.radians(estimated_state.attitude)
        angle_rate_b0 = np.radians(estimated_state.angle_rates)

        # Convert position to local NED co-ordinates
        position_e0 = np.zeros(3)

        # Obtain the body velocity and acceleration
        # Obtain the body velocity and acceleration
        velocity_b0 = estimated_state.velocity
        acceleration_b0 = estimated_state.acceleration

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

        # Variable-step AB2. The stored values are increments rather than raw
        # derivatives, hence the previous-increment coefficient is squared.
        # With equal intervals these reduce exactly to the legacy 3/2, -1/2.
        current_weight, previous_weight = _adams_bashforth_weights(
            time_step, self._previous_time_step)
        position_e2 = position_e0 + current_weight * d_position_e1 \
            + previous_weight * d_position_e0
        velocity_b2 = velocity_b0 + current_weight * d_velocity_b1 \
            + previous_weight * d_velocity_b0
        acceleration_b2 = acceleration_b0 + current_weight * d_acceleration_b1 \
            + previous_weight * d_acceleration_b0
        attitude_2 = attitude_0 + current_weight * d_attitude_1 \
            + previous_weight * d_attitude_0
        angle_rate_b2 = angle_rate_b0 + current_weight * d_angle_rate_b1 \
            + previous_weight * d_angle_rate_b0

        # Ensure the updated attitude values are valid
        attitude_2 = np.remainder(attitude_2 + pi, 2 * pi) - pi

        # Update the estimated variables
        estimated_state.update_estimates(
            position=trans.ned2lla(position_e2, ref_lla),
            velocity=velocity_b2,
            acceleration=acceleration_b2,
            attitude=np.degrees(attitude_2),
            angle_rates=np.degrees(angle_rate_b2)
        )

        # Convert position from local NED co-ordinates back to Lat-Long-Altitude
        self._d_est_position = d_position_e1
        self._d_est_velocity = d_velocity_b1
        self._d_est_acceleration = d_acceleration_b1
        self._d_est_attitude = d_attitude_1
        self._d_est_angle_rates = d_angle_rate_b1
        self._previous_time_step = time_step

