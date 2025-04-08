import math

import numpy as np

from qnav.gravity.base import GravityModel
from qnav.gravity.simple import FixedValue
from qnav.util.transformations import quaternion_to_euler, ned2lla_vec
from qnav.waypoints.generation import get_positions, get_trajectory
from qnav.waypoints.trajectory import GroundTruth
from qnav.waypoints.vehicle import Vehicle


def waypoints_to_ground_truth(waypoints) -> list[GroundTruth]:

    to_return = []


    # position = record[0:3]
    #         velocity = record[3:6]
    #         acceleration = record[6:9]
    #         angle_rates = record[12:15]
    #         attitude = quaternion_to_euler(record[15:19])

    for waypoint in waypoints:
        timestamp = waypoint[0]
        position = waypoint[1:4]
        velocity = waypoint[4:7]
        acceleration = waypoint[7:10]
        angle_rates = waypoint[13:16]
        attitude = quaternion_to_euler(waypoint[16:20])

        gt = GroundTruth(timestamp, position, velocity,
                         acceleration, attitude, angle_rates)

        to_return.append(gt)

    return to_return


def generate_static_position(pos: tuple[float, float], freq: float,
                             num_records: int, gravity_model: GravityModel):

    lla = [pos[0], pos[1], 0]
    waypoints = np.zeros((num_records, 20))

    dt = 1 / freq
    time_steps = np.arange(0, num_records * dt, dt)

    waypoints[:, 0] = time_steps
    waypoints[:, 1:4] = lla
    waypoints[:, 7:10] = gravity_model.calc_gravity_xyz(*lla)
    waypoints[:, 16:20] = [1.0, 0.0, 0.0, 0.0]
    return waypoints_to_ground_truth(waypoints)



def generate_straight_line(start_pos: tuple[float, float],
                           freq: float, angle: float,
                           distance: float, velocity: float,
                           gravity_model: GravityModel):

    # Specify the start and end position
    start_lat, start_lon = start_pos
    angle_rad = math.radians(angle)
    end_lat = start_lat + distance * math.cos(angle_rad)
    end_lon = start_lon + distance * math.sin(angle_rad)

    # Format the base point coordinates
    lats = np.array([start_lat, end_lat])
    lons = np.array([start_lon, end_lon])
    alts = np.zeros_like(lats)

    # Generate and return the ground truth records
    vehicle = Vehicle()
    base_points = get_positions(lats, lons, alts, vehicle, freq, velocity)
    waypoints = get_trajectory(base_points, gravity_model)
    return waypoints_to_ground_truth(waypoints)



def generate_circle(centre_pos: tuple[float, float],
                    freq: float, radius: float, velocity: float,
                    gravity_model: GravityModel):


    num_points = 100
    angles = np.linspace(0, 2 * math.pi, num_points)
    # approx_circum = 2 * math.pi * radius

    # Calculate the position along the racetrack
    x = radius * np.sin(angles)
    y = radius * np.cos(angles)
    z = np.zeros_like(angles)

    ned = np.column_stack([x, y, z])
    ref_lla = np.array([centre_pos[0], centre_pos[1], 0])
    lla_points = ned2lla_vec(ned, ref_lla)

    lats = lla_points[:, 0]
    lons = lla_points[:, 1]
    alts = lla_points[:, 2]

    # Generate and return the ground truth records
    vehicle = Vehicle()
    base_points = get_positions(lats, lons, alts, vehicle, freq, velocity)
    waypoints = get_trajectory(base_points, gravity_model)
    return waypoints_to_ground_truth(waypoints)




if __name__ == '__main__':
    # static = generate_static_position((0,0), 10, 100, FixedValue())
    # line = generate_straight_line((0, 0), 10, 0, 0.1, 100, FixedValue())
    circle = generate_circle((0, 0), 10, 1000, 100, FixedValue())






# def generate_simple_line(start_pos: tuple[float, float],
#                          freq: float, distance: float,
#                          angle: float, velocity: float,
#                          gravity_model: GravityModel):
#
#     start_lat, start_lon = start_pos
#     angle_rad = math.radians(angle)
#
#     position = np.array([start_pos[0], start_pos[1], 0])


