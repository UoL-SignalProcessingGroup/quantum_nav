
import pytest

import math
import numpy as np
import qnav.waypoints.generation as ways

from random import random

from qnav.util.transformations import lla2ned_vec, haversine_vec
from qnav.util.constants import R_E


@pytest.fixture
def random_line() -> dict:

    start_lat = round(random() * 180 - 90)
    start_lon = round(random() * 360 - 180)
    start_alt = 1000

    azimuth = random() * math.pi * 2
    end_lat = math.sin(azimuth) + start_lat
    end_lon = math.cos(azimuth) + start_lon

    return {
        'lat_points': np.array((start_lat, end_lat)),
        'lon_points': np.array((start_lon, end_lon)),
        'alt_points': np.array((start_alt, start_alt))
    }


@pytest.mark.parametrize('frequency,speed', ((100, 100), (1000, 1000)))
@pytest.mark.xfail(reason="Identified differences from WGS-84 parameters")
def test_generate_straight_line(random_line: dict, frequency: float, speed: float) -> None:

    vehicle_model = ways.Vehicle()
    base_points = ways.get_positions(
        **random_line, vehicle_model=vehicle_model,
        frequency=frequency, avg_speed=speed)

    base_lats = random_line['lat_points']
    base_lons = random_line['lon_points']
    base_alts = random_line['alt_points']
    num_points = base_points.shape[0]

    est_time = np.linspace(0, base_points[-1, 0], num_points)
    est_lats = np.linspace(base_lats[0], base_lats[-1], num_points)
    est_lons = np.linspace(base_lons[0], base_lons[-1], num_points)
    est_alts = np.linspace(base_alts[0], base_alts[-1], num_points)

    est_positions = np.column_stack((est_lats, est_lons, est_alts))
    gen_positions = base_points[:, 1:]

    position_errors = lla2ned_vec(est_positions, gen_positions)
    max_position_errors = np.max(np.abs(position_errors), 0)
    time_errors = base_points[:, 0] - est_time

    np.testing.assert_array_less(time_errors, 1 / frequency)
    np.testing.assert_array_less(max_position_errors, 10)


@pytest.mark.parametrize('args', (
        {'frequency': 100, 'speed': 100, 'num_points': 3},
))
def test_multiple_points(random_line: dict, args: dict) -> None:

    num_add_points = args['num_points'] - 2

    lat_range = random_line['lat_points']
    lon_range = random_line['lon_points']
    alt_range = random_line['alt_points']

    rand_lats = np.random.uniform(*lat_range, num_add_points)
    rand_lons = np.random.uniform(*lon_range, num_add_points)
    rand_alts = np.random.uniform(*alt_range, num_add_points)

    points = {
        'lat_points': np.insert(lat_range, 1, rand_lats),
        'lon_points': np.insert(lon_range, 1, rand_lons),
        'alt_points': np.insert(alt_range, 1, rand_alts)
    }

    vehicle_model = ways.Vehicle()
    base_points = ways.get_positions(
        **points, vehicle_model=vehicle_model,
        frequency=args['frequency'], avg_speed=args['speed'])

    positions = base_points[:, 1:]
    for i in range(args['num_points']):

        ref_lla = np.array([
            points['lat_points'][i],
            points['lon_points'][i],
            points['alt_points'][i]
        ])

        distances = np.linalg.norm(
            lla2ned_vec(positions, ref_lla), axis=1)

        assert np.min(distances) < 100


@pytest.mark.parametrize('lat_range', (0.5, 5.0, 90.0))
@pytest.mark.parametrize('lon', (0.0,))
@pytest.mark.parametrize('alt', (1000.0,))
@pytest.mark.parametrize('freq', (1000.0,))
@pytest.mark.parametrize('speed', (100.0,))
def test_latitude_altitude(lat_range: float, lon: float, alt: float, freq: float, speed: float) -> None:

    points = {
        'lat_points': np.array([-1, 1]) * lat_range,
        'lon_points': np.zeros(2) + lon,
        'alt_points': np.zeros(2) + alt
    }

    vehicle_model = ways.Vehicle()
    base_points = ways.get_positions(
        **points, vehicle_model=vehicle_model,
        frequency=freq, avg_speed=speed)

    alt_errors = np.abs(base_points[:, 3] - alt)
    assert np.max(alt_errors) <= 5e-2



@pytest.mark.parametrize('lat', (0.0, 45.0))
@pytest.mark.parametrize('lon_range', (0.5, 5.0, 180.0))
@pytest.mark.parametrize('alt', (1000.0,))
@pytest.mark.parametrize('freq', (1000.0,))
@pytest.mark.parametrize('speed', (100.0,))
def test_longitude_altitude(lat: float, lon_range: float, alt: float, freq: float, speed: float) -> None:

    points = {
        'lat_points': np.zeros(2) + lat,
        'lon_points': np.array([-1, 1]) * lon_range,
        'alt_points': np.zeros(2) + alt
    }

    vehicle_model = ways.Vehicle()
    base_points = ways.get_positions(
        **points, vehicle_model=vehicle_model,
        frequency=freq, avg_speed=speed)

    alt_errors = np.abs(base_points[:, 3] - alt)
    assert np.max(alt_errors) <= 5e-2


@pytest.mark.parametrize('lat_points', ((0.0, 0.1), (1.0, 1.1)))
@pytest.mark.parametrize('lon_points', ((-179.9, 179.9), (1.0, 1.1)))
@pytest.mark.parametrize('alt', (1000.0,))
@pytest.mark.parametrize('freq', (1000.0,))
@pytest.mark.parametrize('speed', (100.0,))
def test_longitude_overlap(lat_points: tuple[float, float],
                           lon_points: tuple[float, float],
                           alt: float, freq: float, speed: float) -> None:

    points = {
        'lat_points': np.array(lat_points),
        'lon_points': np.array(lon_points),
        'alt_points': np.array([alt, alt]),
    }

    # print(points)

    vehicle_model = ways.Vehicle()
    base_points = ways.get_positions(
        **points, vehicle_model=vehicle_model,
        frequency=freq, avg_speed=speed)

    assert np.all(np.abs(base_points[:, 1]) <= 90), 'Latitude overlap found!'
    assert np.all(np.abs(base_points[:, 2]) <= 180), 'Longitude overlap found!'

    A = haversine_vec(
        base_points[0, 1], base_points[0, 2], base_points[0, 3],
        base_points[-1, 1], base_points[-1, 2], base_points[-1, 3])

    B = arc_length(base_points[0, 1], base_points[0, 2],
                   base_points[:, 1], base_points[:, 2])

    # dist_from_start = haversine_vec(
    #     base_points[0, 1], base_points[0, 2], base_points[0, 3],
    #     base_points[:, 1], base_points[:, 2], base_points[:, 3])
    #
    # dist_to_end = haversine_vec(
    #     base_points[-1, 1], base_points[-1, 2], base_points[-1, 3],
    #     base_points[:, 1], base_points[:, 2], base_points[:, 3])




def arc_length(lat1: np.ndarray, lon1: np.ndarray,
               lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    r = np.acos(
        np.sin(lat1_rad) * np.sin(lat2_rad) +
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.cos(lon1_rad - lon2_rad)
    )

    return R_E * np.degrees(r)
