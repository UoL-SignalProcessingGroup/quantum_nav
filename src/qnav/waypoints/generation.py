"""
=============
generation.py
=============

:summary:
    Handles all interpolation and generation of waypoint trajectory data.
    This module contains a collection of functions for generating waypoint
    trajectories for various vehicle models. In addition to this, waypoint
    data can be imported from CSV files or binary numpy files.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.1.0 - Includes fixes and improvements.
"""

import qnav.util.transformations as trans
import qnav.util.constants as const
import numpy as np
import math

from qnav.gravity.base import GravityModel
from qnav.gravity.simple import SimpleUniform, FixedValue
from qnav.gravity.somigliana import Somigliana
from qnav.waypoints.vehicle import Vehicle
from scipy.interpolate import pchip_interpolate
from math import radians, acos, sin, cos, sqrt
from matplotlib import pyplot as plt
from pathlib import Path


def get_positions(lat_points: np.ndarray, lon_points: np.ndarray, alt_points: np.ndarray,
                  vehicle_model: Vehicle, frequency: float = 200.0, avg_speed: float = 100.0,
                  travel_times: np.ndarray = None):
    """
    This function interpolates positions (latitude, longitude and altitude)
    between provided waypoints that are separated by either stated distance or
    time (not both). The generated route is subject to the limitations of the
    vehicle. The produced route can then be used for calculating the
    required trajectory for simulations.

    :param lat_points: The decimal latitude points for the given waypoints.
        Use negative values for southern coordinates (in degrees).
    :type lat_points: numpy.ndarray

    :param lon_points: The decimal longitude points for the given waypoints.
        Use negative values for western coordinates (in degrees).
    :type lon_points: numpy.ndarray

    :param alt_points: The altitude for the given waypoints (in metres).
    :type alt_points: numpy.ndarray

    :param vehicle_model: An initialised vehicle model containing the
        manoeuvring limitations to consider when generating the waypoint
        positions.
    :type vehicle_model: Vehicle

    :param frequency: The simulation's update frequency. This is used for
        calculating the number of intermediate points and total number of
        waypoints to return.
    :type frequency: float

    :param avg_speed: The average vehicle speed for the given waypoints. This
        is used to calculate the travel times between waypoints. This value is
        replaced by 'travel_times'. Therefore, a value for this is not
        required if the travel times are known.
    :type avg_speed: float

    :param travel_times: The time (in seconds) when each waypoint is reached.
        This is required for interpolating the position between points.
    :type travel_times: numpy.ndarray

    :return: An array containing column for latitude, longitude and altitude
        of the vehicle at each time step.
    :rtype: numpy.ndarray (n-by-4 elements)
    """

    # Obtain the gravity constants
    a = const.POLAR_AXIS_A
    b = const.POLAR_AXIS_B
    r_e = const.R_E

    # If travel times have been given:
    if travel_times is None:

        # Calculate the distance between the waypoints.
        distances = trans.haversine_vec(
            lat_points[:-1], lon_points[:-1], alt_points[:-1],
            lat_points[1:], lon_points[1:], alt_points[1:])

        # Round to avoid float-point errors
        # distances = np.around(distances, decimals=3)

        # Calculate the travel times between each waypoint.
        travel_times = np.concatenate(([0.0], distances / avg_speed))

        # Calculate the cumulative sum of the times.
        travel_time_cum_sum = np.cumsum(travel_times)

        # Calculate the number of waypoints required.
        total_num_of_waypoints = int(np.ceil(frequency * travel_time_cum_sum[-1]))

    else:

        # Calculate the number of waypoints required.
        total_num_of_waypoints = int(np.ceil(
            frequency * (travel_times[-1] - travel_times[0])))

        # Use the provided travel times.
        travel_time_cum_sum = travel_times

    # Initialise the waypoints to return
    waypoints_base = np.zeros([total_num_of_waypoints + 1, 4])
    waypoint_node = np.column_stack([travel_time_cum_sum, lat_points, lon_points, alt_points])
    waypoints_base[0, :] = waypoint_node[0, :]

    time_current = travel_times[0]
    dt = 1 / frequency
    dt_sq = dt ** 2

    e_current = [
        cos(radians(lat_points[0])) * cos(radians(lon_points[0])),
        cos(radians(lat_points[0])) * sin(radians(lon_points[0])),
        sin(radians(lat_points[0]))
    ]

    height_current = alt_points[0]

    d_theta = 0
    d_lambda = 0
    e_n = np.zeros(3)

    target_index = 1
    e_target = [
        cos(radians(lat_points[target_index])) * cos(radians(lon_points[target_index])),
        cos(radians(lat_points[target_index])) * sin(radians(lon_points[target_index])),
        sin(radians(lat_points[target_index]))
    ]

    height_records = np.zeros(total_num_of_waypoints + 1)
    height_records[0] = alt_points[0]

    # TODO: Resolve rounding issue?
    target_update_steps = (travel_time_cum_sum / dt) - (time_current / dt)
    target_update_steps = np.ceil(target_update_steps).astype(int)
    target_update_steps[target_update_steps < 1] = 1

    # target_update_step = target_update_steps[1] + 1
    target_update_steps += 1
    target_update_step = target_update_steps[1]
    max_target_index = np.size(target_update_steps) - 1

    #
    for step_count in range(1, total_num_of_waypoints + 1):

        # if step_count > target_update_step:
        if step_count > target_update_step:
            target_index += 1
            target_index = min(target_index, max_target_index)
            target_update_step = target_update_steps[target_index]

            # Update the target matrix:
            e_target = [
                cos(radians(lat_points[target_index])) * cos(radians(lon_points[target_index])),
                cos(radians(lat_points[target_index])) * sin(radians(lon_points[target_index])),
                sin(radians(lat_points[target_index]))
            ]

        # Check time to go to reach current target waypoint
        t_go = travel_time_cum_sum[target_index] - time_current + dt

        # Find axis of rotation from current vector to target vector on sphere
        if time_current < dt:

            # Move me before loop (repeated code)?
            tmp1 = __norm(e_current) * __norm(e_target)
            d_theta_des = acos(__dot(e_current, e_target) / tmp1) * (dt / t_go)
            d_theta = d_theta_des

            """
            The code below is the Python optimised version of the following:
            e_n = cross(e_current, e_target) / norm(cross(e_current, e_target))
            """
            tmp1 = __cross(e_current, e_target)
            tmp2 = __norm(__cross(e_current, e_target))
            e_n = [tmp1[0] / tmp2, tmp1[1] / tmp2, tmp1[2] / tmp2]

        else:

            """
            The code below is the Python optimised version of the following:
            d_theta_des = real(acos(dot(e_current, e_target) /
                (norm(e_current) * norm(e_target)))) * dt/t_go;
            """
            tmp1 = __norm(e_current) * __norm(e_target)
            tmp2 = max(-1, min(1, __dot(e_current, e_target) / tmp1))
            d_theta_des = acos(tmp2) * (dt / t_go)

            d_theta_des = d_theta + max(
                -vehicle_model.deceleration_max * dt_sq / r_e,
                min(vehicle_model.acceleration_max * dt_sq / r_e, d_theta_des - d_theta))

            # d_theta = d_theta + dt / vehicle_model.time_delay_acc * (d_theta_des - d_theta)
            d_theta += dt / vehicle_model.time_delay_acc * (d_theta_des - d_theta)

            """
            The code below is the Python optimised version of the following:
            e_n_des = cross(e_current, e_target) / norm(cross(e_current, e_target))
            """
            # tmp1 = __cross(e_current, e_target)
            # tmp2 = __norm(__cross(e_current, e_target))
            # e_n_des = [tmp1[0] / tmp2, tmp1[1] / tmp2, tmp1[2] / tmp2]
            tmp1 = __cross(e_current, e_target)
            tmp2 = __norm(tmp1)
            e_n_des = [tmp1[0] / tmp2, tmp1[1] / tmp2, tmp1[2] / tmp2]

            if __norm(__cross(e_n, e_n_des)) > (0.5 * dt / vehicle_model.time_delay_turn):

                """
                The code below is the Python optimised version of the following:
                d_lambda_des = min(vehicle_model.turn_rate_max * dt * pi/180.0,
                    real(acos(dot(e_n, e_n_des) / (norm(e_n) * norm(e_n_des)))))
                """
                tmp01 = __dot(e_n, e_n_des)
                tmp02 = __norm(e_n) * __norm(e_n_des)
                tmp03 = max(min(tmp01 / tmp02, 1), -1)
                d_lambda_des = min(radians(vehicle_model.turn_rate_max * dt), acos(tmp03))

                d_lambda += (dt / vehicle_model.time_delay_turn) * (d_lambda_des - d_lambda)

                """
                The code below is the Python optimised version of the following:
                e_l = cross(e_n, e_n_des) / norm(cross(e_n, e_n_des))
                """
                # tmp1 = __cross(e_n, e_n_des)
                # tmp2 = __norm(__cross(e_n, e_n_des))
                # e_l = [tmp1[0] / tmp2, tmp1[1] / tmp2, tmp1[2] / tmp2]
                tmp1 = __cross(e_n, e_n_des)
                tmp2 = __norm(tmp1)
                e_l = [tmp1[0] / tmp2, tmp1[1] / tmp2, tmp1[2] / tmp2]

                """
                The code below is the Python optimised version of the following:
                e_n_new = e_n * cos(d_lambda) + cross(e_l, e_n) * 
                    sin(d_lambda) + e_l * dot(e_l, e_n) * (1 - cos(d_lambda));
                """
                tmp1 = cos(d_lambda)
                tmp2 = [e_n[0] * tmp1, e_n[1] * tmp1, e_n[2] * tmp1]
                tmp3 = __cross(e_l, e_n)
                tmp4 = sin(d_lambda)
                tmp5 = [tmp3[0] * tmp4, tmp3[1] * tmp4, tmp3[2] * tmp4]
                tmp6 = __dot(e_l, e_n) * (1 - tmp1)
                tmp7 = [e_l[0] * tmp6, e_l[1] * tmp6, e_l[2] * tmp6]

                e_n_new = [
                    tmp2[0] + tmp5[0] + tmp7[0],
                    tmp2[1] + tmp5[1] + tmp7[1],
                    tmp2[2] + tmp5[2] + tmp7[2]
                ]

            else:
                d_lambda_des = 0
                d_lambda += (dt / vehicle_model.time_delay_turn) * (d_lambda_des - d_lambda)
                e_n_new = e_n_des

            e_n = e_n_new

        # Increase the current time
        time_current += dt
        time_current = round(time_current, 10)

        # Calculate height for each waypoint.
        height_current += ((waypoint_node[target_index, 3] - height_current) * (dt / t_go))
        height_records[step_count] = height_current

        # Rotate unit vector to new point using Rodrigue's formula.
        """
        The code below is the Python optimised version of the following:
        e_current = e_current * cos(d_theta) + cross(e_n, e_current) \
            * sin(d_theta) + e_n * dot(e_n, e_current) * (1 - cos(d_theta))
        """
        tmp1 = cos(d_theta)
        part1 = [e_current[0] * tmp1, e_current[1] * tmp1, e_current[2] * tmp1]
        tmp1 = __cross(e_n, e_current)
        tmp2 = sin(d_theta)
        part2 = [tmp1[0] * tmp2, tmp1[1] * tmp2, tmp1[2] * tmp2]
        tmp1 = __dot(e_n, e_current) * (1 - cos(d_theta))
        part3 = [e_n[0] * tmp1, e_n[1] * tmp1, e_n[2] * tmp1]

        e_current = [
            part1[0] + part2[0] + part3[0],
            part1[1] + part2[1] + part3[1],
            part1[2] + part2[2] + part3[2]
        ]

        # lat_current_2 = math.degrees(math.atan2(
        #     e_current[2], math.sqrt(e_current[0] ** 2 + e_current[1] ** 2)))
        # r_e = trans.radius84(lat_current_2)

        # [1.21246552e+02 1.31748366e+00 1.15802939e-03]
        # [1.21246553e+02 1.31747668e+00 1.15786569e-03]

        waypoints_base[step_count, 0] = time_current
        waypoints_base[step_count, 1:4] = e_current

    e_0 = waypoints_base[1:, 1]
    e_1 = waypoints_base[1:, 2]
    e_2 = waypoints_base[1:, 3]

    # Add radius and height for elliptical Earth model.
    lat_current = np.rad2deg(np.arctan2(e_2, np.sqrt(e_0 ** 2 + e_1 ** 2)))
    radius_current = trans.radius84_vec(lat_current)
    r_current = np.column_stack([
        (radius_current + height_records[1:]) * e_0,
        (radius_current + height_records[1:]) * e_1,
        (b ** 2 / a ** 2 * radius_current + height_records[1:]) * e_2
    ])

    waypoints_base[1:, 1:4] = trans.ecef2lla_vec(r_current)

    # Finally, return the produced waypoints
    return waypoints_base


def interpolate_positions(base_points: np.ndarray, frequency: float) -> np.ndarray:
    """
    Interpolates the base waypoints at the new higher frequency.
    This function can be used to speed up and stabilise the calculation of
    the waypoint positions, by interpolating the intermediate steps.

    :param base_points: The base waypoint data to interpolate.
    :type base_points: np.ndarray

    :param frequency: The frequency to interpolate the waypoint data by.
    :type frequency: float

    :return: The interpolated version of the given waypoint positions.
    :rtype: np.ndarray
    """

    start_time = base_points[0, 0]
    end_time = base_points[-1, 0]

    dt = 1 / frequency
    base_times = base_points[:, 0]
    interp_times = np.arange(start_time, end_time + dt, dt)
    waypoints = np.zeros((len(interp_times), 4))

    waypoints[:, 0] = interp_times
    waypoints[:, 1] = pchip_interpolate(base_times, base_points[:, 1], interp_times)
    waypoints[:, 2] = pchip_interpolate(base_times, base_points[:, 2], interp_times)
    waypoints[:, 3] = pchip_interpolate(base_times, base_points[:, 3], interp_times)
    return waypoints


def produce_noise(num_rows: int, first_val: float or np.ndarray,
                  noise_val: float, dt_vdp: float):
    """
    This function generated random noise that can be added to various aspects.
    For instance, this function can be used to produce noise for position,
    attitude and oscillation displacement. This function is required so that
    the noise generated is done so in a self-balancing manner such that the
    displacements do not cause permanent or increasing drift.

    :param num_rows: The number of noise values to produce.
    :type num_rows: int

    :param first_val: The first generated noise value(s).
    :type first_val: float or numpy.array

    :param noise_val: The scale of the noise values.
    :type noise_val: float

    :param dt_vdp: The dampening factor to apply. This is used to prevent the
        noise from causing permanent or excessive drift.

    :return: The generated noise subject to the given parameters.
    :rtype: numpy.array
    """

    # Obtain the number of columns to produce.
    num_cols = np.size(first_val)

    # Pre-allocate the data to return.
    to_return = np.zeros([num_rows, num_cols])
    to_return[0, :] = first_val

    # Calculate the noise in advance.
    rand_noise = noise_val * np.random.randn(num_rows, num_cols)

    # If the produced noise values
    # are not going to be all zeros:
    if np.all(rand_noise != 0) or dt_vdp != 0:

        # Generate cumulative noise:
        for i in range(1, num_rows):
            tmp = to_return[i - 1, :] + rand_noise[i]
            to_return[i, :] = tmp + dt_vdp * (first_val - tmp)

    # Finally, return the produced noise data.
    return np.squeeze(to_return)


def moving_average(input_array, window_size: int = 5):
    """
    Obtains the "sliding window average" for values of an array.
    This function calculates the moving average of a given 1D array using the
    same approach as MATLAB. A sliding window is passed over the given array.
    For the centre (right) element the moving average is calculated using the
    values covered by the window. This means that fewer elements are covered
    at the start and end of the array. The returned values are the same size
    as the given array.

    :param input_array: The values which the moving average is going to be
     calculated for. This must be a 1D array.
    :type input_array: NumPy array

    :param window_size: The size of the sliding window. Must be an integer.
    :type window_size: int

    :return: The calculated moving averages for the given array.
    :rtype: NumPy array
    """

    # Obtain the convolutional sum of elements using a window of requested
    # size, then divide each of these values by the window size. This will
    # obtain the moving average for most of the elements
    conv = np.convolve(input_array, np.ones(window_size), 'same')
    to_return = conv / window_size

    # Correct the moving average for the elements at the start of the array.
    start_range = range(math.ceil(window_size / 2), window_size)
    for i, d in enumerate(start_range):
        to_return[i] = conv[i] / d

    # Correct the moving average for the elements at the end of the array.
    end_range = range(math.floor(window_size / 2) + 1, window_size)
    for i, d in enumerate(end_range, start=1):
        to_return[-i] = conv[-i] / d

    # Finally, return the moving averages.
    return to_return


def rescale_values(values):
    """
    A simple function that allows angles to be re-scaled to prevent inaccuracy
    calculations between angles differences.

    :param values: The angle values to be rescaled.
    :type values: numpy.array

    :return: The given values re-scaled between -pi/2 and pi/2.
    :rtype: numpy.array
    """

    tmp1 = values > (np.pi / 2)
    tmp2 = (values < (-np.pi / 2)) & ~tmp1

    values[tmp1] -= 2 * np.pi
    values[tmp2] += 2 * np.pi

    # tmp1 = values > np.pi
    # tmp2 = values < -np.pi
    #
    # values[tmp1] -= 2 * np.pi
    # values[tmp2] += 2 * np.pi

    return values


def get_simple_trajectory_values(waypoints: np.ndarray,
                                 gravity_model: GravityModel,
                                 given_attitudes: np.ndarray = None,
                                 attitude_times: np.ndarray = None,
                                 smoothing_window_time: float = None,
                                 smoothing_passes: int = None,
                                 step_size: int = 2,
                                 virtual_mem_file: Path = None):
    """
    Calculates the trajectory information and properties from given waypoints.
    This function calculates the trajectory properties for each position for
    a given set of waypoints. Only simple properties are generated, without
    any added noise, vibrations or oscillations of the platform. In previous
    releases this function was refereed to as legacy waypoint generation.

    :param waypoints: The interpolated waypoint array previously generated. It
        is important any changes/updates to the waypoints (such as altitude
        changes) are applied to the waypoints before using this function.
    :type waypoints: NumPy Matrix (n-by-4 elements)

    :param gravity_model: The gravity model used to calculate local gravity.
        This is required for calculating the downwards gravitation
        acceleration and its impact on the platform for each axis.
    :type gravity_model: GravityModel

    :param given_attitudes: (Optional) The requested attitude of the vehicle at
        given times. Intermediate attitude values will be interpolated.
    :type given_attitudes: numpy.array (n-elements)

    :param attitude_times: (Optional) The requested corresponding times for
        the given attitudes. These timestamps are used to interpolate
        intermediate attitude values.
    :type attitude_times: numpy.array (n-elements)

    :param smoothing_window_time: (Optional) The size of the smoothing
        (moving average) window to use for smoothing the calculated
        waypoint velocities. This is useful for removing noise.
    :type smoothing_window_time: float

    :param smoothing_passes: (Optional) The number of smoothing passes
        (iterations) to perform on the calculated waypoint velocities.
    :type smoothing_passes: int

    :param step_size: (Optional) The step size used for calculating the
        velocity between upcoming waypoints (default=1).
    :type step_size: int

    :param virtual_mem_file: The on-disk file to use for virtual memory.
        Will be created or overwritten
    :type virtual_mem_file: bool

    :return: A matrix containing the position, velocity, acceleration,
        attitude, angle rates and gravitational acceleration (and gradient)
        for each time step.
    :rtype: NumPy Matrix (n-by-16 elements)
    """

    # Record the number of given waypoints
    num_of_rows = waypoints.shape[0] - (2 * step_size)
    output_shape = (num_of_rows, 16)

    # Pre-allocate an array for storing the returned values.
    if virtual_mem_file is None:
        data = np.zeros(output_shape)
    else:
        data = np.memmap(str(virtual_mem_file), np.float64,
                         'w+', shape=output_shape)

    # Determine if attitude values have been provided.
    attitudes_provided = given_attitudes is not None

    # Obtain the timestamps and the timestamps for the next two steps.
    timestamp = waypoints[:-(2 * step_size), 0]
    timestamp_plus_1 = waypoints[step_size:-step_size, 0]
    timestamp_plus_2 = waypoints[(2 * step_size):, 0]

    # Obtain the positions and the positions for the next two steps.
    positions = waypoints[:-(2 * step_size), 1:4]
    positions_plus_1 = waypoints[step_size:-step_size, 1:4]
    positions_plus_2 = waypoints[(2 * step_size):, 1:4]

    # Find local effective gravity for the positions
    # g = get_gravity_xyz_vec(positions[:, 0], positions[:, 2])
    g = gravity_model.calc_gravity_xyz_vec(
        positions[:, 0], positions[:, 1], positions[:, 2])

    # Radius of Earth at Latitude (metres)
    lat_rad = np.radians(positions[:, 0])

    # Define Angular velocity for Earth's rotation (in local NED axes)
    omega_e = np.zeros([num_of_rows, 3])
    omega_e[:, 0] = const.OMEGA_E * np.cos(lat_rad)
    omega_e[:, 2] = const.OMEGA_E * -np.sin(lat_rad)

    # Convert next position from LLA to NED
    position_ned_plus1 = trans.lla2ned_vec(positions_plus_1, positions)
    position_ned_plus2 = trans.lla2ned_vec(positions_plus_2, positions)

    # Calculate NED velocity
    velocity = (position_ned_plus1.T / (timestamp_plus_1 - timestamp)).T
    velocity_plus_1 = ((position_ned_plus2 - position_ned_plus1).T / (
            timestamp_plus_2 - timestamp_plus_1)).T

    # If waypoint smoothing has been enabled:
    if smoothing_window_time is not None and \
            smoothing_passes is not None:

        # Calculate the smoothing window size.
        time_step = (timestamp[1] - timestamp[0])
        window_size = math.floor(smoothing_window_time / time_step)

        # For each smoothing pass:
        if window_size > 1:
            for i in range(3):
                for _ in range(max(smoothing_passes, 0)):
                    # Calculate the moving average for the velocity values.
                    velocity[:, i] = moving_average(velocity[:, i], window_size)
                    velocity_plus_1[:, i] = moving_average(velocity_plus_1[:, i], window_size)

    # Calculate NED acceleration
    acceleration = ((velocity_plus_1 - velocity).T / (timestamp_plus_1 - timestamp)).T

    # If attitude values have not been provided:
    if not attitudes_provided:

        # Calculate NED attitude.
        attitude = np.zeros([num_of_rows, 3])
        attitude[:, 0] = np.arctan2(velocity[:, 1], velocity[:, 0])
        attitude[:, 1] = np.arctan2(-velocity[:, 2], np.sqrt(
            velocity[:, 1] ** 2 + velocity[:, 0] ** 2))

        # Calculate the derivative of the attitude
        attitude_dt = np.zeros([num_of_rows, 3])
        attitude_dt[:, 0] = np.arctan2(velocity_plus_1[:, 1], velocity_plus_1[:, 0])
        attitude_dt[:, 1] = np.arctan2(-velocity_plus_1[:, 2], np.sqrt(
            velocity_plus_1[:, 1] ** 2 + velocity_plus_1[:, 0] ** 2))

    else:

        # Otherwise, interpolate the attitude values.
        attitude_interp = pchip_interpolate(attitude_times, given_attitudes, waypoints[:, 0])
        attitude_interp = np.deg2rad(attitude_interp)
        attitude_interp = ((attitude_interp + np.pi) % (2 * np.pi)) - np.pi

        # Calculate NED attitude.
        attitude = attitude_interp[:-(2 * step_size)]
        attitude_dt = attitude_interp[step_size:-step_size]

    # # Calculate the change in attitude
    tmp = attitude_dt - attitude
    tmp = ((tmp.T + np.pi) % (2 * np.pi)) - np.pi
    tmp = (tmp / (timestamp_plus_1 - timestamp)).T

    # Obtain psi, theta, phi
    d_psi_dt = tmp[:, 0]
    d_theta_dt = tmp[:, 1]
    d_phi_dt = tmp[:, 2]

    # Angle rates in body axes
    p = d_phi_dt - np.sin(attitude[:, 1]) * d_psi_dt
    q = np.cos(attitude_dt[:, 2]) * d_theta_dt + np.sin(attitude_dt[:, 2]) * np.cos(attitude_dt[:, 1]) * d_psi_dt
    r = -np.sin(attitude_dt[:, 2]) * d_theta_dt + np.cos(attitude_dt[:, 2]) * np.cos(attitude_dt[:, 1]) * d_psi_dt
    angle_rate = np.column_stack([p, q, r])

    # Rotation matrix to move from local earth axes to body axes
    rot_earth2body = trans.rotate_3d_vec(
        attitude[:, 0], attitude[:, 1], attitude[:, 2])

    # Acceleration in Local NED Earth axes
    acceleration += 2.0 * np.cross(omega_e, velocity)

    # Convert angle rates to body axes:
    angle_rate += np.einsum('nji, ni -> nj', rot_earth2body, omega_e)

    # If attitude values have been provided:
    if attitudes_provided:

        # Include the effects of the transport rate for angle rates.
        omega_tr = trans.get_transport_rate_vec(positions, velocity)
        angle_rate += np.einsum('nji, ni -> nj', rot_earth2body, omega_tr)

    # Construct 'specific force' as acceleration (in local NED axes)
    acceleration -= g

    # Convert velocity from Earth axes to Body axes
    velocity = np.einsum('nji, ni -> nj', rot_earth2body, velocity)

    # Add gravity and convert acceleration from Earth axes to
    # Body axes and add the Coriolis terms
    acceleration = np.einsum('nji, ni -> nj', rot_earth2body, acceleration)

    data[:, 0] = waypoints[:-(2 * step_size), 0]  # Time in seconds
    data[:, 1:4] = positions  # Position (LLA)
    data[:, 4:7] = velocity  # Velocity (Body axes)
    data[:, 7:10] = acceleration  # Acceleration (Body axes)
    data[:, 10:13] = np.degrees(attitude)  # Attitudes (Local Earth axes)
    data[:, 13:16] = np.degrees(angle_rate)  # Angle Rates (Body Axes)

    # Correct the first record.
    data[0, 1:] = data[1, 1:]

    # Finally, return the data for the waypoints
    return data

def get_trajectory_values(waypoints: np.ndarray,
                          gravity_model: GravityModel,
                          vehicle_model: Vehicle,
                          given_attitudes: np.ndarray = None,
                          attitude_times: np.ndarray = None,
                          smoothing_window_time: float = None,
                          smoothing_passes: int = None,
                          step_size: int = 2,
                          virtual_mem_file: Path = None):
    """
    Using waypoints route information generated prior, this function will
    calculate the velocity, acceleration, attitude, angle rates and
    gravitational acceleration (and gradient) for each time step. This
    function calculates said values in an efficient vectorised manner
    while ensuring they abide by the given vehicle model's limitations.

    :param waypoints: The interpolated waypoint array previously generated. It
        is important any changes/updates to the waypoints (such as altitude
        changes) are applied to the waypoints before using this function.
    :type waypoints: NumPy Matrix (n-by-4 elements)

    :param gravity_model: The gravity model used to calculate local gravity.
        This is required for calculating the downwards gravitation
        acceleration and its impact on the platform for each axis.
    :type gravity_model: GravityModel

    :param vehicle_model: An initialised vehicle model containing the
        manoeuvring limitations to consider when generating the waypoint
        positions.
    :type vehicle_model: Vehicle

    :param given_attitudes: (Optional) The requested attitude of the vehicle at
        given times. Intermediate attitude values will be interpolated.
    :type given_attitudes: numpy.array (n-elements)

    :param attitude_times: (Optional) The requested corresponding times for
        the given attitudes. These timestamps are used to interpolate
        intermediate attitude values.
    :type attitude_times: numpy.array (n-elements)

    :param smoothing_window_time: (Optional) The size of the smoothing
        (moving average) window to use for smoothing the calculated
        waypoint velocities. This is useful for removing noise.
    :type smoothing_window_time: float

    :param smoothing_passes: (Optional) The number of smoothing passes
        (iterations) to perform on the calculated waypoint velocities.
    :type smoothing_passes: int

    :param step_size: (Optional) The step size used for calculating the
        velocity between upcoming waypoints (default=1).
    :type step_size: int

    :param virtual_mem_file: The on-disk file to use for virtual memory.
        Will be created or overwritten
    :type virtual_mem_file: bool

    :return: A matrix containing the position, velocity, acceleration,
        attitude, angle rates and gravitational acceleration (and gradient)
        for each time step.
    :rtype: NumPy Matrix (n-by-16 elements)
    """

    # Record the number of records to produce.
    num_of_rows = waypoints.shape[0] - (3 * step_size)
    num_of_points = waypoints.shape[0]

    # Pre-allocate an array for storing the returned values.
    if virtual_mem_file is None:
        data = np.zeros((num_of_rows, 16))
    else:
        data = np.memmap(str(virtual_mem_file), np.float64,
                         'w+', shape=(num_of_rows, 16))

    # Determine if attitude values have been provided.
    attitudes_provided = given_attitudes is not None

    # Obtain the inverse of the waypoint frequency.
    # dt = round(waypoints[1][0] - waypoints[0][0], 10)
    dt = round(float(waypoints[1][0] - waypoints[0][0]), 10)

    # Convert the noise acceleration to m/s^2.sqrt(Hz).
    vibration_noise_acceleration = \
        vehicle_model.vibration_noise_acceleration / math.sqrt(dt)

    # Convert the noise angle rates to deg/s.sqrt(Hz).
    vibration_noise_angle_rates = \
        vehicle_model.vibration_noise_angle_rates / math.sqrt(dt)

    # Set the seed for random number generation
    np.random.seed(vehicle_model.rand_seed)

    #  Set random initial phase for vehicle oscillations
    phase_pitch_osc_0 = 360.0 * np.random.rand()
    phase_roll_osc_0 = 360.0 * np.random.rand()
    initial_variation = 0.25

    pitch_osc_magnitude_0 = vehicle_model.pitch_osc_magnitude * \
        (1.0 + initial_variation * np.random.randn())

    roll_osc_magnitude_0 = vehicle_model.roll_osc_magnitude * \
        (1.0 + initial_variation * np.random.randn())

    pitch_osc_period = vehicle_model.pitch_osc_period * \
        (1.0 + initial_variation * np.random.randn())

    roll_osc_period = vehicle_model.roll_osc_period * \
        (1.0 + initial_variation * np.random.randn())

    # Extract the damping time (in seconds).
    if vehicle_model.vibration_damping_period != 0:
        dt_vdp = dt / vehicle_model.vibration_damping_period
    else:
        dt_vdp = 0

    # ----------------------------------------------------------

    position_deviations = np.zeros((num_of_points, 3))

    rand_noise = 0.5 * vibration_noise_acceleration * \
                 np.random.randn(num_of_points, 3) * dt ** 2.5

    position_deviation_0 = rand_noise[0, :]
    position_deviations[0, :] = position_deviation_0
    position_deviations[1, :] = position_deviations[0, :] + 0.5 * rand_noise[1, :]
    position_deviations[2, :] = position_deviations[1, :] + 0.5 * rand_noise[2, :]
    position_deviations[3, :] = position_deviations[2, :] + 0.5 * rand_noise[3, :]

    for i in range(4, num_of_points):
        tmp = position_deviations[i-1, :] + rand_noise[i, :]
        position_deviations[i-1, :] = tmp + dt_vdp * (position_deviation_0 - tmp)

    # ----------------------------------------------------------

    attitude_deviations = np.zeros((num_of_points, 3))
    rand_noise = 0.5 * np.pi / 180.0 * vibration_noise_angle_rates * \
        np.random.randn(num_of_points, 3) * dt ** 1.5

    attitude_deviation_0 = rand_noise[0, :]
    attitude_deviations[0, :] = attitude_deviation_0
    attitude_deviations[1, :] = attitude_deviations[0, :] + rand_noise[1, :]
    attitude_deviations[2, :] = attitude_deviations[1, :] + rand_noise[2, :]

    for i in range(3, num_of_points):
        tmp = position_deviations[i - 1, :] + rand_noise[i, :]
        attitude_deviations[i, :] = tmp + dt_vdp * (attitude_deviation_0 - tmp)

    # ----------------------------------------------------------

    phase_pitch_osc = np.zeros(num_of_points)
    rand_noise = phase_pitch_osc_0 * vehicle_model.variation_per_second2 * \
                 np.random.randn(num_of_points) * math.sqrt(dt)

    phase_pitch_osc[0] = phase_pitch_osc_0
    for i in range(1, num_of_points):
        tmp = phase_pitch_osc[i - 1] + rand_noise[i]
        phase_pitch_osc[i] = tmp + dt_vdp * (phase_pitch_osc_0 - tmp)

    # ----------------------------------------------------------

    phase_roll_osc = np.zeros(num_of_points)
    rand_noise = phase_roll_osc_0 * vehicle_model.variation_per_second2 * \
         np.random.randn(num_of_points) * math.sqrt(dt)

    phase_roll_osc[0] = phase_roll_osc_0
    for i in range(1, num_of_points):
        tmp = phase_roll_osc[i - 1] + rand_noise[i]
        phase_roll_osc[i] = tmp + dt_vdp * (phase_roll_osc_0 - tmp)

    # ----------------------------------------------------------

    pitch_osc_magnitude = np.zeros(num_of_points)
    rand_noise = pitch_osc_magnitude_0 * vehicle_model.variation_per_second1 * \
         np.random.randn(num_of_points) * math.sqrt(dt)

    pitch_osc_magnitude[0] = pitch_osc_magnitude_0
    for i in range(1, num_of_points):
        tmp = pitch_osc_magnitude[i - 1] + rand_noise[i]
        pitch_osc_magnitude[i] = tmp + dt_vdp * (pitch_osc_magnitude_0 - tmp)

    # ----------------------------------------------------------

    roll_osc_magnitude = np.zeros(num_of_points)
    rand_noise = roll_osc_magnitude_0 * vehicle_model.variation_per_second1 * \
                 np.random.randn(num_of_points) * math.sqrt(dt)

    roll_osc_magnitude[0] = roll_osc_magnitude_0
    for i in range(1, num_of_points):
        tmp = roll_osc_magnitude[i - 1] + rand_noise[i]
        roll_osc_magnitude[i] = tmp + dt_vdp * (roll_osc_magnitude_0 - tmp)

    # ----------------------------------------------------------

    # Pre-calculate the indices to assign to current and next steps.
    # This is done to enforce consistency and prevent mistakes.
    indices = [0, num_of_rows]
    indices_1 = [step_size, num_of_rows + step_size]
    indices_2 = [step_size * 2, num_of_rows + step_size * 2]
    indices_3 = [step_size * 3, num_of_rows + step_size * 3]

    # Obtain the positions and the positions for the next two steps.
    positions = waypoints[indices[0]:indices[1], 1:4]
    positions_plus_1 = waypoints[indices_1[0]:indices_1[1], 1:4]
    positions_plus_2 = waypoints[indices_2[0]:indices_2[1], 1:4]
    positions_plus_3 = waypoints[indices_3[0]:indices_3[1], 1:4]

    # The times for all waypoint positions.
    time_stamps = waypoints[:, 0]

    # Obtain the timestamps and the timestamps for the next two steps.
    time_step_dt = waypoints[indices_1[0]:indices_1[1], 0] - waypoints[indices[0]:indices[1], 0]
    time_step_dt_plus_1 = waypoints[indices_2[0]:indices_2[1], 0] - waypoints[indices_1[0]:indices_1[1], 0]
    time_steps_dt_plus_2 = waypoints[indices_3[0]:indices_3[1], 0] - waypoints[indices_2[0]:indices_2[1], 0]

    # Radius of Earth at Latitude (metres)
    lat_rad = np.deg2rad(positions[:, 0])

    # Define Angular velocity for Earth's rotation (in local NED axes)
    omega_e = np.zeros([num_of_rows, 3])
    omega_e[:, 0] = const.OMEGA_E * np.cos(lat_rad)
    omega_e[:, 2] = const.OMEGA_E * -np.sin(lat_rad)

    # Find local effective gravity for the positions
    # g = get_gravity_xyz_vec(positions[:, 0], positions[:, 2])
    g = gravity_model.calc_gravity_xyz_vec(
        positions[:, 0], positions[:, 1], positions[:, 2])

    # Convert next position from LLA to NED
    position_ned = np.zeros([num_of_rows, 3])
    position_ned_plus1 = trans.lla2ned_vec(positions_plus_1, positions)
    position_ned_plus2 = trans.lla2ned_vec(positions_plus_2, positions)
    position_ned_plus3 = trans.lla2ned_vec(positions_plus_3, positions)

    # Add the pre-generated position deviation noise
    position_ned += position_deviations[indices[0]:indices[1], :]
    position_ned_plus1 += position_deviations[indices_1[0]:indices_1[1], :]
    position_ned_plus2 += position_deviations[indices_2[0]:indices_2[1], :]
    position_ned_plus3 += position_deviations[indices_3[0]:indices_3[1], :]

    # Calculate the NED velocities
    velocities = ((position_ned_plus1 - position_ned).T / time_step_dt).T
    velocities_plus1 = ((position_ned_plus2 - position_ned_plus1).T / time_step_dt_plus_1).T
    velocities_plus2 = ((position_ned_plus3 - position_ned_plus2).T / time_steps_dt_plus_2).T

    # If waypoint smoothing has been enabled:
    if smoothing_window_time is not None and \
            smoothing_passes is not None:

        # Calculate the smoothing window size.
        window_size = math.floor(smoothing_window_time / dt)

        # For each smoothing pass:
        if window_size > 1:
            for i in range(3):
                for _ in range(max(smoothing_passes, 0)):

                    # Calculate the moving average for the velocity values.
                    velocities[:, i] = moving_average(velocities[:, i], window_size)
                    velocities_plus1[:, i] = moving_average(velocities_plus1[:, i], window_size)
                    velocities_plus2[:, i] = moving_average(velocities_plus2[:, i], window_size)

    # Calculate the NED acceleration
    accelerations = ((velocities_plus1 - velocities).T / time_step_dt).T

    # If attitude values have not been provided:
    if not attitudes_provided:

        # Calculate the current step's attitude values.
        attitudes = np.zeros([num_of_rows, 3])
        attitudes[:, 0] = np.arctan2(velocities[:, 1], velocities[:, 0])
        attitudes[:, 1] = np.arctan2(-velocities[:, 2], np.sqrt(
            velocities[:, 1] ** 2 + velocities[:, 0] ** 2))

        # Calculate the next step's attitude values.
        attitude_dt = np.zeros([num_of_rows, 3])
        attitude_dt[:, 0] = np.arctan2(velocities_plus1[:, 1], velocities_plus1[:, 0])
        attitude_dt[:, 1] = np.arctan2(-velocities_plus1[:, 2], np.sqrt(
            velocities_plus1[:, 1] ** 2 + velocities_plus1[:, 0] ** 2))

        # Calculate the next-next step's attitude values.
        attitude_dt2 = np.zeros([num_of_rows, 3])
        attitude_dt2[:, 0] = np.arctan2(velocities_plus2[:, 1], velocities_plus2[:, 0])
        attitude_dt2[:, 1] = np.arctan2(-velocities_plus2[:, 2], np.sqrt(
            velocities_plus2[:, 1] ** 2 + velocities_plus2[:, 0] ** 2))

    else:

        # Otherwise, convert the given attitude values to radians.
        attitude_interp = pchip_interpolate(attitude_times, given_attitudes, waypoints[:, 0])
        attitude_interp = np.deg2rad(attitude_interp)
        attitude_interp = ((attitude_interp + np.pi) % (2 * np.pi)) - np.pi

        # Calculate NED attitude.
        attitudes = attitude_interp[indices[0]:indices[1]]
        attitude_dt = attitude_interp[indices_1[0]:indices_1[1]]
        attitude_dt2 = attitude_interp[indices_2[0]:indices_2[1]]

    # Add the noise to the calculated/obtained attitude values.
    attitudes = attitudes + attitude_deviations[indices[0]:indices[1]]
    attitude_dt += attitude_deviations[indices_1[0]:indices_1[1]]
    attitude_dt2 += attitude_deviations[indices_2[0]:indices_2[1]]

    # Adjust attitude to allow for individual vehicle models - Constant offsets/biases.
    attitudes[:, 1] += radians(vehicle_model.pitch_constant_offset)
    attitudes[:, 2] += radians(vehicle_model.roll_constant_offset)

    # Roll when turning
    d_psi_dt = rescale_values(attitude_dt[:, 0] - attitudes[:, 0]) / time_step_dt
    attitudes[:, 2] += vehicle_model.d_roll_d_turn_rate * d_psi_dt

    #
    if pitch_osc_period != 0:

        # Calculate the pitch oscillations.
        pitch_oscillation = pitch_osc_magnitude * np.sin(
            2.0 * np.pi * time_stamps / pitch_osc_period +
            np.pi / 180.0 * phase_pitch_osc)

    else:

        # If a pitch oscillation period has not been given,
        # simply use zeros as place fillers.
        pitch_oscillation = np.zeros(pitch_osc_magnitude.shape)

    if roll_osc_period != 0.0:

        # Calculate the roll oscillations.
        roll_oscillation = roll_osc_magnitude * np.sin(
            2.0 * np.pi * time_stamps / roll_osc_period +
            np.pi / 180.0 * phase_roll_osc)
    else:

        # If a roll oscillation period has not been given,
        # simply use zeros as place fillers.
        roll_oscillation = np.zeros(roll_osc_magnitude.shape)

    # Introduce the oscillations to the attitudes.
    attitudes[:, 1] += np.radians(pitch_oscillation[indices[0]:indices[1]])
    attitudes[:, 2] += np.radians(roll_oscillation[indices[0]:indices[1]])

    # Ensure that the maximum pitch is not exceeded.
    tmp = np.abs(attitudes[:, 1]) > np.radians(vehicle_model.pitch_max)
    attitudes[tmp, 1] = np.radians(vehicle_model.pitch_max) * np.sign(attitudes[tmp, 1])

    # Ensure that the maximum roll is not exceeded.
    tmp = np.abs(attitudes[:, 2]) > np.radians(vehicle_model.roll_max)
    attitudes[tmp, 2] = np.radians(vehicle_model.roll_max) * np.sign(attitudes[tmp, 2])

    # Adjust attitude to allow for individual vehicle models - Constant offsets/biases.
    attitude_dt[:, 1] += radians(vehicle_model.pitch_constant_offset)
    attitude_dt[:, 2] += radians(vehicle_model.roll_constant_offset)

    # Roll when turning
    d_psi_dt = rescale_values(attitude_dt2[:, 0] - attitude_dt[:, 0]) / time_step_dt_plus_1
    attitude_dt[:, 2] += vehicle_model.d_roll_d_turn_rate * d_psi_dt

    # Introduce the oscillations to the attitudes.
    attitude_dt[:, 1] += np.radians(pitch_oscillation[indices_1[0]:indices_1[1]])
    attitude_dt[:, 2] += np.radians(roll_oscillation[indices_1[0]:indices_1[1]])

    # Ensure that the maximum pitch is not exceeded.
    tmp = np.abs(attitude_dt[:, 1]) > np.radians(vehicle_model.pitch_max)
    attitude_dt[tmp, 1] = np.radians(vehicle_model.pitch_max) * np.sign(attitude_dt[tmp, 1])

    # Ensure that the maximum roll is not exceeded.
    tmp = np.abs(attitude_dt[:, 2]) > np.radians(vehicle_model.roll_max)
    attitude_dt[tmp, 2] = np.radians(vehicle_model.roll_max) * np.sign(attitude_dt[tmp, 2])

    # Construct Euler Angle rates
    d_psi_dt = rescale_values(attitude_dt[:, 0] - attitudes[:, 0]) / time_step_dt
    d_theta_dt = rescale_values(attitude_dt[:, 1] - attitudes[:, 1]) / time_step_dt
    d_phi_dt = rescale_values(attitude_dt[:, 2] - attitudes[:, 2]) / time_step_dt

    # Angle rates in body axes
    p = d_phi_dt - np.sin(attitudes[:, 1]) * d_psi_dt
    q = np.cos(attitudes[:, 2]) * d_theta_dt + np.sin(attitudes[:, 2]) * np.cos(attitudes[:, 1]) * d_psi_dt
    r = -np.sin(attitudes[:, 2]) * d_theta_dt + np.cos(attitudes[:, 2]) * np.cos(attitudes[:, 1]) * d_psi_dt
    angle_rate = np.column_stack([p, q, r])

    # Rotation matrix to move from local earth axes to body axes
    rot_earth2body = trans.rotate_3d_vec(
        attitudes[:, 0], attitudes[:, 1], attitudes[:, 2])

    # Acceleration in Local NED Earth axes
    accelerations += 2.0 * np.cross(omega_e, velocities)

    # Convert angle rates to body axes:
    angle_rate += np.einsum('nji, ni -> nj', rot_earth2body, omega_e)

    # If attitude values have been provided:
    if attitudes_provided:

        # Include the effects of the transport rate for angle rates.
        omega_tr = trans.get_transport_rate_vec(positions, velocities)
        angle_rate += np.einsum('nji, ni -> nj', rot_earth2body, omega_tr)

    # Construct 'specific force' as acceleration (in local NED axes)
    accelerations -= g

    # Convert velocity from Earth axes to Body axes
    velocities = np.einsum('nji, ni -> nj', rot_earth2body, velocities)

    # Add gravity and convert acceleration from Earth axes
    # to Body axes and add the Coriolis terms
    accelerations = np.einsum('nji, ni -> nj', rot_earth2body, accelerations)

    # Include deviations applied for position points.
    position_data = trans.ned2lla_vec(position_ned, positions)

    data[:, 0] = waypoints[indices[0]:indices[1], 0]  # Time in seconds
    data[:, 1:4] = position_data  # Position (LLA)
    data[:, 4:7] = velocities  # Velocity (Body axes)
    data[:, 7:10] = accelerations  # Acceleration (Body axes)
    data[:, 10:13] = np.degrees(attitudes)  # Attitudes (Local Earth axes)
    data[:, 13:16] = np.degrees(angle_rate)  # Angle Rates (Body Axes)

    # Correct the first record.
    data[0, 1:] = data[1, 1:]

    # Finally, return the waypoint data.
    return data


def __cross(a, b):
    """
    An optimised method for computing the cross-product of two 1-by-3 tuples.
    This method is far quicker that the Numpy cross function.

    :param a: The first 1-by-3 tuple/array.
    :type a: tuple (3 elements)

    :param b: The first 1-by-3 tuple/array.
    :type b: tuple (3 elements)

    :return: The cross product of the two tuples.
    :rtype: tuple (3 elements)
    """
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ]


def __dot(a, b):
    """
    An optimised method for computing the dot-product of two 1-by-3 tuples.
    This method is far quicker that the numpy dot function.

    :param a: The first 1-by-3 tuple/array.
    :type a: tuple (3 elements)

    :param b: The first 1-by-3 tuple/array.
    :type b: tuple (3 elements)

    :return: The dot product of the two tuples.
    :rtype: tuple (3 elements)
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def __norm(a):
    """
    An optimised method for computing the norm of two 1-by-3 tuples.
    This method is far quicker that the numpy norm function.

    :param a: The first 1-by-3 tuple/array.
    :type a: tuple (3 elements)

    :return: The norm of the two tuples.
    :rtype: tuple (3 elements)
    """
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def plot_waypoints(data_values: np.ndarray, filename=None):
    """
    A simple function to plot the generated waypoints.

    :param data_values: The array of values returned by the generator
        function.
    :type data_values: numpy.ndarray (n-by-3 elements)

    :param filename: (Optional) Saves the produced plot under the given file.
    :type filename: str
    """

    # Set the backend for figure generation
    # plt_use('tkagg')

    # Plot the computed trajectory path.
    plt.figure(dpi=300, figsize=(11.51, 8.14))
    plt.plot(data_values[:, 2], data_values[:, 1], label="Trajectory")

    # Plot the starting location.
    plt.scatter(data_values[0, 2], data_values[0, 1],
                marker="o", facecolor="#1f77bf", label="Start")

    # Plot the ending/goal location.
    plt.scatter(data_values[-1, 2], data_values[-1, 1],
                marker="x", facecolor="#1f77bf", label="End")

    plt.title("Produced Waypoints", fontsize=9.5, fontweight='bold')
    plt.xlabel("Longitude (degrees)", fontsize=8.5)
    plt.ylabel("Latitude (degrees)", fontsize=8.5)
    plt.legend(loc="best", fontsize=7.5)

    plt.axis('equal')
    # plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_figures(data_values: np.ndarray, filename=None):
    """
    A simple function to plot the change in variables during the waypoint
    generation. Each of these is presented within a sub-figure.

    :param data_values: The array of values returned by the generator
        function.
    :type data_values: numpy.ndarray (n-by-16 elements)

    :param filename: (Optional) Saves the produced plot under the given file.
    :type filename: str
    """

    # Set the backend for figure generation
    # plt_use('agg')

    # Create a plot with 5 vertical sub-plots.
    fig, axs = plt.subplots(5, 1, dpi=300, figsize=(8.14, 11.51))

    # Specify the font sizes to use.
    title_font_size = 9.5
    axis_font_size = 8.5
    label_font_size = 7.5

    # Altitude against Time
    axs[0].set_title("Altitude against Time", fontsize=title_font_size, fontweight='bold')
    axs[0].set_ylabel("Altitude (m)", fontsize=axis_font_size)
    axs[0].set_xlabel('Time (s)', fontsize=axis_font_size)
    axs[0].plot(data_values[:, 0], data_values[:, 3], '-b', label="Altitude")
    axs[0].legend(loc='best', fontsize=label_font_size)

    # Velocity against Time
    axs[1].set_title("Velocity against Time", fontsize=title_font_size, fontweight='bold')
    axs[1].set_ylabel("Velocity (m/s)", fontsize=axis_font_size)
    axs[1].set_xlabel('Time (s)', fontsize=axis_font_size)
    axs[1].plot(data_values[:, 0], data_values[:, 4], '-b',
                data_values[:, 0], data_values[:, 5], '-r',
                data_values[:, 0], data_values[:, 6], '-g')
    axs[1].legend(labels=("Body Along", "Body Across", "Body Down"),
                  loc='best', fontsize=label_font_size)

    # Acceleration against Time
    axs[2].set_title("Acceleration against Time", fontsize=title_font_size, fontweight='bold')
    axs[2].set_ylabel("Acceleration (m/s²)", fontsize=axis_font_size)
    axs[2].set_xlabel('Time (s)', fontsize=axis_font_size)
    axs[2].plot(data_values[:, 0], data_values[:, 7], '-b',
                data_values[:, 0], data_values[:, 8], '-r',
                data_values[:, 0], data_values[:, 9], '-g')
    axs[2].legend(labels=("Body Along", "Body Across", "Body Down"),
                  loc='best', fontsize=label_font_size)

    # Attitude (degrees) against Time
    axs[3].set_title("Attitude against Time", fontsize=title_font_size, fontweight='bold')
    axs[3].set_ylabel("Attitude (deg)", fontsize=axis_font_size)
    axs[3].set_xlabel('Time (s)', fontsize=axis_font_size)
    axs[3].plot(data_values[:, 0], data_values[:, 10], '-b',
                data_values[:, 0], data_values[:, 11], '-r',
                data_values[:, 0], data_values[:, 12], '-g')
    axs[3].legend(labels=("Heading", "Pitch", "Roll"), loc='best',
                  fontsize=label_font_size)

    # Angle Rate (degrees) against Time
    axs[4].set_title("Angle Rates against Time", fontsize=title_font_size, fontweight='bold')
    axs[4].set_ylabel("Angle Rates (deg/s)", fontsize=axis_font_size)
    axs[4].set_xlabel('Time (s)', fontsize=axis_font_size)
    axs[4].plot(data_values[:, 0], data_values[:, 13], '-b',
                data_values[:, 0], data_values[:, 14], '-r',
                data_values[:, 0], data_values[:, 15], '-g')
    axs[4].legend(labels=("Body Rate R", "Body Rate Q", "Body Rate P"),
                  loc='best', fontsize=label_font_size)

    #
    fig.align_ylabels(axs)
    fig.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)
    else:
        plt.show()


def load_waypoint_csv(filepath: Path, down_sample_rate: int = 1):
    """
    This function will extract the values from a given trajectory file,
    formatted as comma-separated-values. Once the relevant information is
    extracted, 7 rows will be returned containing information of the extracted
    timestamps, latitude points, longitude points, altitude points and three
    attitude angles. These rows are returned within a list.

    :param filepath: The location of the CSV trajectory file to read.
    :type filepath: str or Path

    :param down_sample_rate: The down sampling multiplier to apply.
        This can be used to skip rows for cases where CSV waypoints are
        given too frequently or spaced too closely together.
    :type down_sample_rate: int

    :return: A list containing 7 arrays holding the extracted timestamps,
        latitude points, longitude points, altitude points and attitude
        angles from the given CSV trajectory file.
    :rtype: NumPy array
    """

    # Read the contents from the given file.
    file_content = np.genfromtxt(filepath, delimiter=",")

    # Find all rows that are not numeric/usable.
    valid_rows = ~np.isnan(file_content).any(axis=1)

    # Remove all invalid rows.
    file_content = file_content[valid_rows]

    # Ensure a supported number of columns is given.
    num_columns = file_content.shape[1]
    assert num_columns in [3, 4, 6, 7], \
        "Invalid column count in given CSV file!"

    # If only position data is given:
    if num_columns == 3:

        # Get the 3 columns and use None for the missing columns.
        cols = np.hsplit(file_content, 3)
        cols.insert(0, None)
        cols.extend([None, None, None])

    # If attitude values are not given:
    elif num_columns == 4:

        # Get the 4 columns and use None for the missing columns.
        cols = np.hsplit(file_content, 4)
        cols.extend([None, None, None])
        # return cols

    elif num_columns == 6:

        # Get the 6 columns and use None for the missing times.
        cols = np.hsplit(file_content, 6)
        cols.insert(0, None)

    else:

        # If attitude values are given, get the 7 columns.
        cols = np.hsplit(file_content, 7)

    # Finally, return the CSV columns as individual arrays.
    # return [column.flatten() for column in file_columns]
    csv_data = [None if col is None else col.flatten() for col in cols]

    # Apply down sampling if requested:
    if down_sample_rate > 1:
        csv_data = [None if col is None else col[::down_sample_rate] for col in csv_data]

    return csv_data


def crop_waypoints(waypoints: np.ndarray, max_time: float):
    """
    Crops waypoint data between time ranges (in seconds).

    :param waypoints: The waypoint data to crop.
    :type waypoints: numpy.ndarray

    :param max_time: The maximum time for the return waypoint data.
    :type max_time: float

    :return: The cropped waypoint data
    :rtype: numpy.ndarray
    """

    # TODO: Support start time cropping + resetting time
    if waypoints[-1, 0] > max_time:
        max_index = np.where(waypoints[:, 0] <= max_time)[0][-1] + 1
        waypoints = waypoints[:max_index, :]
    return waypoints


def save_waypoints(waypoints: np.ndarray, file_path: Path, allow_overwrite: bool = False):
    """
    Exports waypoint data to disk, allowing for reuse.

    :param waypoints: The waypoint data to save.
    :type waypoints: numpy.array

    :param file_path: The file location to save the waypoint data to.
    :type file_path: pathlib.Path

    :param allow_overwrite: Should file overwriting be permitted? This
        safeguards against accidental overwriting (default=False).
    :type allow_overwrite: bool
    """

    if not allow_overwrite and file_path.exists():
        raise FileExistsError(f"Unable to overwrite existing "
                              f"waypoint file: {file_path}")

    if not file_path.parent.exists():
        file_path.mkdir(parents=True)

    np.save(str(file_path), waypoints, allow_pickle=False)


def load_waypoints(file_path: Path, as_virtual: bool = False):
    """
    Imports waypoint data from disk, reusing saved values.

    :param file_path: The file location to load the waypoint data from.
    :type file_path: pathlib.Path

    :param as_virtual: If set to True prevent all the data being loaded at
        once and use virtual memory. This reduces RAM usage, but impacts
        runtime (default=False).
    :type as_virtual: bool

    :return: The loaded waypoint data.
    :rtype: numpy.array (n-by-16 elements)
    """

    if as_virtual:
        # return np.load(str(file_path),  mmap_mode='r', allow_pickle=False)
        return np.load(str(file_path), mmap_mode='r')  # TODO: Fix (Sometimes breaks?)
    else:
        # return np.load(str(file_path), allow_pickle=False)
        return np.load(str(file_path))

# def load_waypoints_mmap(file_path: Path):
#     return np.memmap(str(file_path), dtype=np.dtype(np.float64), mode='r')

def get_trajectory(waypoints: np.ndarray,
                   gravity_model: GravityModel):
    """
    TODO: Add documentation
    :param waypoints:
    :param gravity_model:
    :return:
    """

    # Record the number of given waypoints
    # output_shape = (num_of_rows, 20)
    num_of_rows = waypoints.shape[0] - 2
    step_size = 1

    # Obtain the timestamps and the timestamps for the next two steps.
    timestamp = waypoints[:-(2 * step_size), 0]
    timestamp_plus_1 = waypoints[step_size:-step_size, 0]
    timestamp_plus_2 = waypoints[(2 * step_size):, 0]

    # Obtain the positions and the positions for the next two steps.
    positions = waypoints[:-(2 * step_size), 1:4]
    positions_plus_1 = waypoints[step_size:-step_size, 1:4]
    positions_plus_2 = waypoints[(2 * step_size):, 1:4]

    g = gravity_model.calc_gravity_xyz_vec(
        positions[:, 0], positions[:, 1], positions[:, 2])

    # Radius of Earth at Latitude (metres)
    lat_rad = np.radians(positions[:, 0])

    # Define Angular velocity for Earth's rotation (in local NED axes)
    omega_e = np.zeros([num_of_rows, 3])
    omega_e[:, 0] = const.OMEGA_E * np.cos(lat_rad)
    omega_e[:, 2] = const.OMEGA_E * -np.sin(lat_rad)

    # Convert next position from LLA to NED
    position_ned_plus_1 = trans.lla2ned_vec(positions_plus_1, positions)
    position_ned_plus_2 = trans.lla2ned_vec(positions_plus_2, positions)

    # Calculate NED velocity
    velocity = (position_ned_plus_1.T / (timestamp_plus_1 - timestamp)).T
    velocity_plus_1 = ((position_ned_plus_2 - position_ned_plus_1).T / (
            timestamp_plus_2 - timestamp_plus_1)).T

    # Calculate NED acceleration
    acceleration = ((velocity_plus_1 - velocity).T / (timestamp_plus_1 - timestamp)).T

    # Calculate NED attitude.
    attitude = np.zeros([num_of_rows, 3])
    attitude[:, 0] = np.arctan2(velocity[:, 1], velocity[:, 0])
    attitude[:, 1] = np.arctan2(-velocity[:, 2], np.sqrt(
        velocity[:, 1] ** 2 + velocity[:, 0] ** 2))

    # Calculate the derivative of the attitude
    attitude_plus_1 = np.zeros([num_of_rows, 3])
    attitude_plus_1[:, 0] = np.arctan2(velocity_plus_1[:, 1], velocity_plus_1[:, 0])
    attitude_plus_1[:, 1] = np.arctan2(-velocity_plus_1[:, 2], np.sqrt(
        velocity_plus_1[:, 1] ** 2 + velocity_plus_1[:, 0] ** 2))

    # TODO: OLD
    # Calculate the change in attitude
    # attitude_dt = attitude_plus_1 - attitude
    # attitude_dt = ((attitude_dt.T + np.pi) % (2 * np.pi)) - np.pi
    # attitude_dt = (attitude_dt / (timestamp_plus_1 - timestamp)).T

    # TODO: NEW
    attitude_dt = __ang_diff(attitude, attitude_plus_1)
    attitude_dt /= (timestamp_plus_1 - timestamp)[:, None]

    # Obtain psi, theta, phi
    d_psi_dt = attitude_dt[:, 0]
    d_theta_dt = attitude_dt[:, 1]
    d_phi_dt = attitude_dt[:, 2]

    # Angle rates in body axes
    # TODO: CHECK ME
    # p = d_phi_dt - np.sin(attitude[:, 1]) * d_psi_dt
    # q = np.cos(attitude_plus_1[:, 2]) * d_theta_dt + np.sin(attitude_plus_1[:, 2]) * np.cos(attitude_plus_1[:, 1]) * d_psi_dt
    # r = -np.sin(attitude_plus_1[:, 2]) * d_theta_dt + np.cos(attitude_plus_1[:, 2]) * np.cos(attitude_plus_1[:, 1]) * d_psi_dt
    # angle_rate = np.column_stack([p, q, r])

    # TODO: NEW
    p = d_phi_dt - np.sin(attitude[:, 1]) * d_psi_dt
    q = np.cos(attitude[:, 2]) * d_theta_dt + np.sin(attitude[:, 2]) * np.cos(attitude[:, 1]) * d_psi_dt
    r = -np.sin(attitude[:, 2]) * d_theta_dt + np.cos(attitude[:, 2]) * np.cos(attitude[:, 1]) * d_psi_dt
    angle_rate = np.column_stack([p, q, r])

    # Rotation matrix to move from local earth axes to body axes
    rot_earth2body = trans.rotate_3d_vec(
        attitude[:, 0], attitude[:, 1], attitude[:, 2])

    # Acceleration in Local NED Earth axes
    acceleration += 2.0 * np.cross(omega_e, velocity)

    # Convert angle rates to body axes:
    angle_rate += np.einsum('nji, ni -> nj', rot_earth2body, omega_e)

    # Construct 'specific force' as acceleration (in local NED axes)
    acceleration -= g

    # Convert velocity from Earth axes to Body axes
    velocity = np.einsum('nji, ni -> nj', rot_earth2body, velocity)

    # Add gravity and convert acceleration from Earth axes to
    # Body axes and add the Coriolis terms
    acceleration = np.einsum('nji, ni -> nj', rot_earth2body, acceleration)

    # Convert attitude to degrees
    attitude = np.degrees(attitude)

    # Convert angle rates to degrees
    angle_rate = np.degrees(angle_rate)

    # Convert attitude to quaternions
    orientation = trans.euler_to_quaternion_vec(attitude)

    # TODO: Support memory mapping and referencing
    # Collect and return trajectory records
    return np.column_stack([
        timestamp,
        positions,
        velocity,
        acceleration,
        attitude,
        angle_rate,
        orientation
    ])


def __ang_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    # Radians!

    diff = y - x

    if np.any(np.abs(diff) > np.pi):
        theta = diff + np.pi
        theta_wrap = theta % (2 * np.pi)
        theta_wrap[(theta_wrap == 0) & (theta > 0)] = (2 * np.pi)
        diff = theta_wrap - np.pi

    return diff


def read_csv_waypoints(file_path: Path, delimiter: str = ",") -> dict[str, np.ndarray | None]:
    """
    Reads given CSV waypoint file and returns the records contained within it.
    For a given file path, attempts to read its content and return it's
    extracted records inside a dictionary.

    :param file_path: The CSV file to attempt to read waypoint data from.
    :type file_path: Path

    :param delimiter: The delimiter used to represent column breaks.
        By default, this is a comma, but other characters can be used.
    :type delimiter: str

    :return: A dictionary holding the extracted data.
    :rtype: dict[str, np.ndarray | None]
    """

    # Attempt to read the data
    file_data = np.genfromtxt(file_path, delimiter=delimiter)
    file_data = np.atleast_2d(file_data)
    num_cols = file_data.shape[1]

    # Validate its content
    if np.isnan(file_data).any() or np.isinf(file_data).any():
        raise ValueError("NaN or Inf values present in data")

    match num_cols:

        case 3:
            times = None
            lat_points = file_data[:, 0]
            lon_points = file_data[:, 1]
            alt_points = file_data[:, 2]
            heading = None
            pitch = None
            yaw = None

        case 4:
            times = file_data[:, 0]
            lat_points = file_data[:, 1]
            lon_points = file_data[:, 2]
            alt_points = file_data[:, 3]
            heading = None
            pitch = None
            yaw = None

        case 6:
            times = None
            lat_points = file_data[:, 0]
            lon_points = file_data[:, 1]
            alt_points = file_data[:, 2]
            heading = file_data[:, 3]
            pitch = file_data[:, 4]
            yaw = file_data[:, 5]

        case 7:
            times = file_data[:, 0]
            lat_points = file_data[:, 1]
            lon_points = file_data[:, 2]
            alt_points = file_data[:, 3]
            heading = file_data[:, 4]
            pitch = file_data[:, 5]
            yaw = file_data[:, 6]

        case _:
            raise ValueError("Invalid waypoints data shape")

    return {
        'times': times,
        'latitude': lat_points,
        'longitude': lon_points,
        'altitude': alt_points,
        'heading': heading,
        'pitch': pitch,
        'yaw': yaw
    }




if __name__ == '__main__':

    test_lats = np.array([0.0, 0.10])
    test_lons = np.array([0.0, 0.0])
    test_alts = np.array([0.0, 0.0])

    vehicle = Vehicle()
    avg_speed = 10.0
    freq = 10.0

    test_out = get_positions(test_lats, test_lons, test_alts,
                             vehicle, freq, avg_speed)

    data_out = get_trajectory(test_out, FixedValue())



# if __name__ == '__main__':
#
#     test_lats = np.array([53.407579, 53.410000, 53.700000, 53.407579])
#     test_lons = np.array([-2.967853, -2.70000, -2.80000, -2.967853])
#     test_alts = np.array([1000, 1000, 1000, 1000])
#
#     turn_rate_max = 10.0
#     roll_max = 30.0
#
#     # Modified large aircraft vehicle:
#     vehicle = Vehicle(
#         turn_rate_max=turn_rate_max,
#         acceleration_max=0.2*9.81,
#         deceleration_max=0.1*9.81,
#         time_delay_acc=0.5,
#         time_delay_turn=0.5,
#         pitch_constant_offset=2.0,
#         pitch_osc_period=0.0,
#         pitch_osc_magnitude=0.0,
#         pitch_max=45.0,
#         roll_constant_offset=0.0,
#         roll_osc_period=0.0,
#         roll_osc_magnitude=0.0,
#         roll_max=roll_max,
#         d_roll_d_turn_rate=roll_max/turn_rate_max,
#         vibration_noise_acceleration=0.0,
#         vibration_noise_angle_rates=0.0,
#         vibration_damping_period=1.0,
#         variation_per_second1=0.005,
#         variation_per_second2=0.01,
#         rand_seed=0
#     )
#
#     test_out = get_positions(test_lats, test_lons, test_alts, vehicle, 1000)
#     # plot_waypoints(test_out)
#
#     data_out = get_trajectory_values(test_out, vehicle, step_size=8)
#     plot_figures(data_out)
