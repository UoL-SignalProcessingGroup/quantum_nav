"""
=========
fusion.py
=========

:summary:
    A collection of GPS/GNSS methods for fusing GPS measurements with INS.
    This module contains all the GPS fusion methods for merging GPS
    position and velocity updates with the INS current estimation.
    Simpler methods will simply merge the estimates, whereas more
    sophisticated methods will adapt expected errors to compensate
    for INS noise/drift.

:authors:
    | Kallum O'Hara - sgkohara@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

from abc import ABC
from typing import Collection

from qnav.gps.satellite import Satellite
from qnav.gps.sensor import GpsSensor
from qnav.estimation.state import EstimatedState
from qnav.estimation.state import KalmanEstimatedState
from qnav.measurement.sensor import SensorFusion
from qnav.measurement.sensor import FusionTrigger
from qnav.fusion.kalman import state_vector2ned
from qnav.fusion.kalman import ned2state_vector

import qnav.util.transformations as trans
import numpy as np
import math


class GpsFusion(SensorFusion, ABC):
    """
    The abstract base class for fusion methods.
    This class outlines the structure each fusion method must extend from.
    This essentially serves as a template class containing methods and
    properties that must be incorporated by each GPS fusion method.
    """

    def __init__(self, gps_sensor: GpsSensor,
                 estimated_state: EstimatedState,
                 estimated_clock_bias: float = 0.0,
                 estimated_clock_drift: float = 0.0,
                 burn_in_iterations : int = 3):
        """
        Abstract class constructor defining the inputs and default behaviour.
        Initializes the fusion with GPS sensor, initial estimated state
        and optional estimates for the clock bias and drift rates.

        :param gps_sensor: The GPS sensor used to obtain GPS measurements.
        :type gps_sensor: GpsSensor

        :param estimated_state: The initial estimated state.
        :type estimated_state: EstimatedState

        :param estimated_clock_bias: (Optional) The estimated clock bias
            error. TODO: ADD UNITS!
        :type estimated_clock_bias: float

        :param estimated_clock_drift: (Optional) The estimated clock bias
            drift rate error. TODO: ADD UNITS
        :type estimated_clock_drift: float
        """

        # Call parent class's constructor
        super().__init__(FusionTrigger.ANY, gps_sensor)
        self._gps_sensor = gps_sensor

        # Record the amount of burn in iterations requested.
        self._burn_in_iterations = burn_in_iterations

        # Unpack the estimated state
        est_position = estimated_state.position
        est_velocity = estimated_state.velocity
        est_attitude = estimated_state.attitude

        # Convert velocity from body to earth frame
        rot_earth2body = trans.rotate_3d(*np.radians(est_attitude))
        est_velocity_ned = np.linalg.solve(rot_earth2body, est_velocity)

        # Record the initial estimated position in ECEF
        self._estimated_pos_ecef = trans.lla2ecef(est_position)

        # Record the initial estimated velocity in ECEF
        self._estimated_vel_ecef = trans.vned2vecef(est_velocity_ned, est_position)

        # Record the initial estimated clock bias and drift
        self._estimated_clock_bias = estimated_clock_bias
        self._estimated_clock_drift = estimated_clock_drift

    def solve(self) -> None:
        """
        Performs solving of GPS measurements to estimated values.
        Essentially this translates the GPS satellite measurements into
        internal local estimates.
        """

        # If needed to perform burn in iterations
        while self._burn_in_iterations > 0:
            self._burn_in_iterations -= 1
            self._gps_sensor.update(0)
            self._update_estimates()

        self._update_estimates()

    def _update_estimates(self) -> None:
        """
        When internally called, updates the estimates using last measurements.
        Uses the latest GPS satellite measurements to update the internally
        estimated position, velocity, clock bias and clock drift rate. This
        is provided as a helper for the required perform_fusion method.
        """

        # Obtain the latest measurements
        # measurements = self._gps_sensor.last_measurement
        measurements = self._gps_sensor.last_measurement['sat_measurements']
        num_measurements = len(measurements)  # Count the satellites
        a = np.zeros([num_measurements, 4])  # Satellite Geometric Matrix
        r = np.zeros(num_measurements)  # Residuals Vector
        r_vel = np.zeros(num_measurements)  # Velocity Residuals Vector

        # For each satellite in the collection:
        for i, measurement in enumerate(measurements):

            # Extract the satellite measurement properties.
            pseudo_range = measurement['pseudoRangeNoisy']
            pseudo_range_rate = measurement['pseudoRangeRatesNoisy']
            sat_pos_ecef = measurement['satellitePosition']
            sat_vel_ecef = measurement['satelliteVelocity']

            # Relative position of satellite to estimated antenna position
            d_p = sat_pos_ecef - self._estimated_pos_ecef

            # Estimated pseudo range
            # est_rho = (np.dot(d_p.transpose(), d_p)) ** 0.5
            est_rho = np.linalg.norm(d_p)  # Optimised

            # Pseudo range residual
            rho_res = pseudo_range - est_rho

            # Estimated relative velocity
            est_relative_vel = (sat_vel_ecef - self._estimated_vel_ecef)

            # Estimated Line-of-sight vector
            est_los = d_p / est_rho

            # Estimated Pseudo range Rate
            est_rho_rate = est_los.dot(est_relative_vel)

            # est_rho_rate = est_rho_rate
            rho_rate_res = pseudo_range_rate - est_rho_rate

            # Update Residuals Vector
            r[i] = rho_res
            r_vel[i] = rho_rate_res

            # Update Satellite Geometric Matrix
            a[i] = [-d_p[0] / est_rho, -d_p[1] / est_rho, -d_p[2] / est_rho, 1]

        # Update the estimated position and estimated clock bias.
        update_vec_pos = (np.linalg.inv(a.T.dot(a)).dot(a.T)).dot(r)
        self._estimated_pos_ecef += update_vec_pos[0:3]
        self._estimated_clock_bias += update_vec_pos[-1]

        # Update the estimated velocity and estimated clock bias drift.
        update_vec_vel = (np.linalg.inv(a.T.dot(a)).dot(a.T)).dot(r_vel)
        self._estimated_vel_ecef += update_vec_vel[0:3]
        self._estimated_clock_drift += update_vec_vel[-1]

    def _hold_update(self) -> bool:
        """
        Helper function used to determine if GPS updates should be held.
        This is subject to the last measurements.

        :return: True if fusion updates should NOT be applied at this time.
        :rtype: bool
        """
        last_measurement = self._gps_sensor.last_measurement
        return last_measurement is None or last_measurement['is_usable']



class GpsFixedGainFusion(GpsFusion):
    """
    Performs simple fixed-gain GPS fusion.
    Updates the estimated state's position and velocity using simple fixed
    gained fusion. Can be used with any estimated state class.
    """

    def __init__(self, gps_sensor: GpsSensor,
                 estimated_state: EstimatedState,
                 gain_amount: float = 0.1):
        """
        Generates fixed-gained GPS fusion instance.
        Performs fixed-gained fusion for GPS measurements with given sensors
        and gain amount.

        :param gps_sensor: The GPS sensor used to obtain GPS measurements.
        :type gps_sensor: GpsSensor

        :param estimated_state: The initial estimated state.
        :type estimated_state: EstimatedState

        :param gain_amount: The gain amount to apply between 0.0 and 1.0.
        :type gain_amount: float
        """

        # Call parent class's constructor
        super().__init__(gps_sensor, estimated_state)

        # Record the gain amount
        if gain_amount < 0 or gain_amount > 1:
            raise ValueError('gain_amount must be between 0 and 1')
        self._gain_amount = gain_amount


    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Performs GPS correction using simple fixed-gained fusion.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState
        """

        # Update the current internal states
        self.solve()

        # TODO: New - Used for delaying zones
        if self._hold_update():
            return

        # Unpack the estimated state
        ins_position = estimated_state.position
        ins_velocity = estimated_state.velocity
        ins_attitude = estimated_state.attitude

        # Obtain the vehicle's position currently estimated by the GPS
        gps_est_position = trans.ecef2lla(self._estimated_pos_ecef)

        # Convert GPS velocity into local NED/Earth axes
        gps_velocity_ned = trans.vel_ecef2vel_ned(
            self._estimated_vel_ecef, gps_est_position)

        # GPS velocity is now produced in NED axes rather than ECEF
        attitude_rad = np.deg2rad(ins_attitude)
        rot_earth2body = trans.rotate_3d(*attitude_rad)

        # Convert GPS velocity from Earth axes to Body axes
        gps_velocity = rot_earth2body @ gps_velocity_ned

        ins_position += self._gain_amount * (gps_est_position - ins_position)
        ins_velocity += self._gain_amount * (gps_velocity - ins_velocity)

        # Finally, update the current INS estimates
        estimated_state.update_estimates(
            position=ins_position, velocity=ins_velocity)


class GpsLooseFusion(GpsFusion):
    """
    Performs loosely-coupled based GPS fusion.
    A fusion method that uses loosely-coupled fusion for updating the INS
    with given GPS estimates. To further improve the accuracy of the
    navigation solution, the error states are fed back to the INS.
    Must be used with a Kalman estimated state class.
    """

    def __init__(self, gps_sensor: GpsSensor, estimated_state: KalmanEstimatedState,
                 pos_cov_mat_ned: np.ndarray, vel_cov_mat_ned: np.ndarray):
        """
        Creates a Loosely-coupled GPS fusion instance.
        This instance is initialed by setting the initial position and
        velocity covariance matrices.

        :param gps_sensor: The GPS sensor used to obtain GPS measurements.
        :type gps_sensor: GpsSensor

        :param estimated_state: The initial estimated state.
        :type estimated_state: KalmanEstimatedState

        :param pos_cov_mat_ned: The position covariance matrix (in NED).
        :type pos_cov_mat_ned: numpy.array (3-by-3 elements)

        :param vel_cov_mat_ned: The velocity covariance matrix (in NED).
        :type vel_cov_mat_ned: numpy.array (3-by-3 elements)
        """

        super().__init__(gps_sensor, estimated_state)

        if not isinstance(estimated_state, KalmanEstimatedState):
            raise ValueError('estimated_state must be KalmanEstimatedState')

        if pos_cov_mat_ned.shape != (4,4):
            raise ValueError('pos_cov_mat_ned must be of shape (4,4)')

        if vel_cov_mat_ned.shape != (4,4):
            raise ValueError('vel_cov_mat_ned must be of shape (4,4)')

        self.__pos_cov_mat_ned = pos_cov_mat_ned
        self.__vel_cov_mat_ned = vel_cov_mat_ned

    def perform_fusion(self, kalman_state: KalmanEstimatedState) -> None:
        """
        Applies loose-coupled fusion using GPS estimates and INS solution.
        This method updates the given INS instance's estimates by fusing them
        with the given GPS estimates. The INS states are updated internally.

        :param kalman_state: The current estimated state.
        :type kalman_state: KalmanEstimatedState
        """

        # Update the current internal states
        self.solve()

        # TODO: New - Used for delaying zones
        if self._hold_update():
            return

        # Obtain the vehicle's position currently estimated by the GPS
        gps_position_lla = trans.ecef2lla(self._estimated_pos_ecef)

        # Convert GPS velocity into local NED/Earth axes
        gps_velocity_ned = trans.vel_ecef2vel_ned(
            self._estimated_vel_ecef, gps_position_lla)

        # Unpack the estimates state
        ref_lla = kalman_state.position
        gravity_model = kalman_state.gravity_model
        state_vector = kalman_state.state_vector

        # Convert estimated state to NED format
        state_vector_ned = state_vector2ned(
            state_vector, ref_lla, gravity_model)

        # Fuse GPS Position update with current INS state
        h_m = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

        # Construct measurement covariance matrix
        r_m = np.zeros((6, 6))
        r_m[0:3, 0:3] = self.__pos_cov_mat_ned[0:3, 0:3]
        r_m[3:6, 3:6] = self.__vel_cov_mat_ned[0:3, 0:3]

        # Position in local NED/Earth axes
        ins_measurement = h_m @ state_vector_ned
        gps_measurement = np.zeros(6)
        gps_measurement[0:3] = trans.lla2ned(gps_position_lla, ref_lla)
        gps_measurement[3:6] = gps_velocity_ned

        # Obtain the GPS innovation vector
        innovation_vec = gps_measurement - ins_measurement

        # Kalman Filter Equations
        # Calculate Kalman Gain
        state_errors = kalman_state.state_errors
        s_m = r_m + h_m @ state_errors @ h_m.T
        k_m = state_errors @ h_m.T @ np.linalg.inv(s_m)

        # Update state with innovation
        state_vector_ned += k_m @ innovation_vec

        # Update errors after measurement
        state_errors -= k_m @ s_m @ k_m.T

        # Convert state vector back from NED format
        state_vector = ned2state_vector(state_vector_ned, ref_lla, gravity_model)

        # Finally, update the estimated state
        kalman_state.update_estimates(
            position=state_vector[[0, 3, 6]],
            velocity=state_vector[[1, 4, 7]],
            acceleration=state_vector[[2, 5, 8]],
            attitude=state_vector[[9, 11, 13]],
            angle_rates=state_vector[[14, 12, 10]],
            state_errors=state_errors
        )


class GpsTightFusion(GpsFusion):
    """
    Tightly-coupled based GPS fusion.
    A fusion method that uses tightly-coupled fusion for updating the INS
    with given GPS estimates. This method uses a centralized Kalman filter
    that integrates the estimated Pseudo Ranges and Doppler shift from the
    GNSS receiver, along with the estimated position, velocity and attitude
    from the INS. Must be used with a Kalman estimated state class.
    """

    def __init__(self, gps_sensor: GpsSensor, estimated_state: KalmanEstimatedState,
                 tight_state_vector: np.ndarray, tight_state_errors: np.ndarray):
        """
        Creates a Tightly-coupled GPS fusion instance.
        This instance is initialed by setting the initial tight state vector
        and tight state errors.

        :param gps_sensor: The GPS sensor used to obtain GPS measurements.
        :type gps_sensor: GpsSensor

        :param estimated_state: The initial estimated state.
        :type estimated_state: KalmanEstimatedState

        :param tight_state_vector: The initial tight state vector with 17
            states (including clock static bias mean and drift rate).
            Information for all typical states will be taken from the INS.
        :type tight_state_vector: numpy.array (17-elements)

        :param tight_state_errors: The initial tight state errors for 17
            states (including clock static bias mean and drift rate).
            Information for all typical states will be taken from the INS.
        :type tight_state_errors: numpy.array (17-by-17 elements)
        """

        # Call parent class's constructor
        super().__init__(gps_sensor, estimated_state)

        if not isinstance(estimated_state, KalmanEstimatedState):
            raise ValueError('estimated_state must be KalmanEstimatedState')

        assert tight_state_vector.shape == (17,), \
            "Tight state vector must contain 17 elements!"

        assert tight_state_errors.shape == (17, 17), \
            "Tight state error must be of size 17 x 17!"

        self.__tight_state_vector = tight_state_vector
        self.__tight_state_errors = tight_state_errors
        self.__dx = 0.0001

    def perform_fusion(self, estimated_state: KalmanEstimatedState) -> None:
        """
        Applies tightly-coupled fusion using GPS estimates and INS solution.
        This method updates the given INS instance's estimates by fusing them
        with the given GPS estimates. The INS states are updated internally.

        :param estimated_state: The current estimated state.
        :type estimated_state: KalmanEstimatedState
        """

        # Update the current internal states
        self.solve()

        # TODO: New - Used for delaying zones
        if self._hold_update():
            return

        # Ensure the state vector is up-to-date
        self.__tight_state_vector[:15] = estimated_state.state_vector

        # Ensure the state errors are up-to-date
        self.__tight_state_errors[:15, :15] = np.copy(estimated_state.state_errors)

        ref_lla = estimated_state.position
        gravity_model = estimated_state.gravity_model

        # Convert State Vector from LLA/Body co-ordinates to Local NED/Earth axes.
        state_vector = np.copy(self.__tight_state_vector)
        state_vector[:15] = state_vector2ned(
            self.__tight_state_vector[:15], ref_lla, gravity_model)

        # Fuse GPS Position update with current INS state Number of satellites
        satellites = self._gps_sensor.last_measurement['sat_measurements']
        # satellites = self._gps_sensor.last_measurement


        num_sat = len(satellites)
        h_m = np.zeros([2 * num_sat, 17])
        gps_measurement = np.zeros(2 * num_sat)

        # For all satellites:
        for sat_index, sat in enumerate(satellites):

            sat_pos_ecef = sat['satellitePosition']
            sat_pos_ned = trans.ecef2ned(sat_pos_ecef, ref_lla)

            # Pseudo-ranges
            rho = np.sqrt(
                (sat_pos_ned[0] - state_vector[0]) ** 2 +
                (sat_pos_ned[1] - state_vector[3]) ** 2 +
                (sat_pos_ned[2] - state_vector[6]) ** 2) + state_vector[15]

            gps_measurement[sat_index] = sat['pseudoRangeNoisy'] - rho

            # Pseudo-range rates
            los_vector_ned = (sat_pos_ned - state_vector[[0, 3, 6]]) / rho

            # Actual user velocity (NED axes)
            imu_vel_ned = state_vector[[1, 4, 7]]

            # Satellite velocity in NED
            sat_vel_ecef = sat['satelliteVelocity']
            sat_vel = trans.vel_ecef2vel_ned(sat_vel_ecef, ref_lla)

            vel_relative_ned = sat_vel - imu_vel_ned
            rho_dot = np.dot(los_vector_ned, vel_relative_ned)
            gps_measurement[sat_index + num_sat] = sat['pseudoRangeRatesNoisy'] - rho_dot

            dx = self.__dx

            # For each of the states covered in the matrix:
            for state_index in range(17):

                state_vector_dx = np.copy(state_vector)
                state_vector_dx[state_index] += dx

                rho_dx = np.sqrt(
                    (sat_pos_ned[0] - state_vector_dx[0]) ** 2 +
                    (sat_pos_ned[1] - state_vector_dx[3]) ** 2 +
                    (sat_pos_ned[2] - state_vector_dx[6]) ** 2) + state_vector_dx[15]

                d_rho_dx = (rho_dx - rho) / dx
                h_m[sat_index][state_index] = d_rho_dx

                # Pseudo-range rates
                los_vector_ned_dx = (sat_pos_ned - state_vector_dx[[0, 3, 6]]) / rho

                # Actual user velocity (NED axes)
                imu_vel_ned_dx = state_vector_dx[[1, 4, 7]]
                vel_rel_ned_dx = sat_vel - imu_vel_ned_dx

                rho_dot_dx = np.dot(los_vector_ned_dx, vel_rel_ned_dx)
                d_rho_dot_dx = (rho_dot_dx - rho_dot) / dx

                h_m[sat_index + num_sat][state_index] = d_rho_dot_dx

        # Construct measurement covariance matrix
        r_m = np.zeros([2 * num_sat, 2 * num_sat])
        for sat_index in range(num_sat):
            r_m[sat_index][sat_index] = 1.0 ** 2
            r_m[sat_index + num_sat][sat_index + num_sat] = 0.05 ** 2

        innovation_vec = gps_measurement

        # Kalman Filter Equations
        s_m = r_m + h_m @ self.__tight_state_errors @ h_m.T

        # Calculate Kalman Gain
        k_m = self.__tight_state_errors @ h_m.T @ np.linalg.inv(s_m)

        # Update state with innovation
        state_vector += k_m @ innovation_vec

        # Update errors after measurement
        self.__tight_state_errors -= k_m @ s_m @ k_m.T

        # Update state vector for output (and convert back to LLA/Body axes)
        state_vector[:15] = ned2state_vector(
            state_vector[:15], ref_lla, gravity_model)
        self.__tight_state_vector = state_vector

        # Finally, update the estimated state
        estimated_state.update_estimates(
            position=state_vector[[0, 3, 6]],
            velocity=state_vector[[1, 4, 7]],
            acceleration=state_vector[[2, 5, 8]],
            attitude=state_vector[[9, 11, 13]],
            angle_rates=state_vector[[14, 12, 10]],
            state_errors=self.__tight_state_errors[:15, :15]
        )


def calculate_cov_mat(satellites: Collection[Satellite],
                      ant_pos_lla: np.ndarray,
                      sigma_position: float = 50,
                      sigma_velocity: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    Produces the covariance matrices for loosely-coupled fusion.
    This method produces and returns the covariance matrices for position
    and velocity based on the current position. These are namely required
    for initialising a loosely-coupled GPS fusion instance (if matrices
    are not produced manually).

    :param satellites: The GPS satellites involved in measurements.
    :type satellites: Collection[Satellite]

    :param ant_pos_lla: The true antenna position (latitude, longitude
        and altitude given in deg-deg-metres).
    :type ant_pos_lla: numpy.array (3-elements)

    :param sigma_position: The pseudo-range position error (in metres).
    :type sigma_position: float

    :param sigma_velocity: The pseudo-range velocity error (in metres/sec).
    :type sigma_velocity: float

    :return: A tuple containing the constructed position and velocity
        covariance matrices, respectively.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    num_satellites = len(satellites)
    g = np.zeros([num_satellites, 4])

    # For each satellite in the collection:
    for i, sat in enumerate(satellites):

        azimuth, elev = sat.get_location_azimuth_elevation(ant_pos_lla)
        azimuth_rad = trans.deg2rad(azimuth)
        elev_rad = trans.deg2rad(elev)
        cel = math.cos(elev_rad)
        caz = math.cos(azimuth_rad)
        sel = math.sin(elev_rad)
        saz = math.sin(azimuth_rad)
        sat_vec = (cel * caz, cel * saz, -sel)
        g[i] = [-sat_vec[0], -sat_vec[1], -sat_vec[2], 1]

    # Calculate dilution of precision matrix
    dop_mat_ned = np.linalg.inv(g.T @ g)

    pos_cov_mat = dop_mat_ned * (sigma_position ** 2)
    vel_cov_mat = dop_mat_ned * (sigma_velocity ** 2)

    return pos_cov_mat, vel_cov_mat
