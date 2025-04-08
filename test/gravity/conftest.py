

import pytest
import random
import numpy as np

from pathlib import Path


def sort_via_tiles(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray):
    """
    Sorts points into one degree tiles.
    """

    lat_score = np.floor(lat)
    lon_score = (lon + 180) / 360
    alt_score = alt

    # ind = np.lexsort((lat_score, lon_score, alt_score))
    ind = np.lexsort((alt_score, lon_score, lat_score))
    return lat[ind], lon[ind], alt[ind]

# def lex_sort(*values: np.ndarray):
#     """
#     Sorts values in presented key order
#     """
#
#     ind = np.lexsort(values)
#     return (v[ind] for v in values)






@pytest.fixture
def valid_lla_range() -> dict[str, float]:
    """
    Defines the range for valid LLA coordinates.

    :return: A dictionary with the range of supported LLA positions.
    :rtype: dict[str, float]
    """
    return {
        'min_lat': -90.0, 'max_lat': 90.0,
        'min_lon': -180.0, 'max_lon': 180.0,
        'min_alt': 0.0, 'max_alt': 10000.0
    }


@pytest.fixture
def invalid_lla_range() -> dict[str, float]:
    """
    Defines the range for invalid/unsupported LLA coordinates.

    :return: A dictionary with the range of unsupported LLA positions.
    :rtype: dict[str, float]
    """
    return {
        'min_lat': 0.0, 'max_lat': 0.0,
        'min_lon': 0.0, 'max_lon': 0.0,
        'min_alt': 0.0, 'max_alt': 10000.0
    }


@pytest.fixture
def random_lla(valid_lla_range) -> tuple[float, float, float]:
    """
    Returns a random latitude, longitude, altitude values.
    Latitude values are between +/- 90 degrees, longitude values are
    between +/- 180 degrees and altitude is between 0 and 1000.

    :param valid_lla_range: A dictionary defining supported LLA positions.
    :type valid_lla_range: dict[str, float]

    :return: Tuple containing random latitude, longitude, altitude values.
    :rtype: tuple[float, float, float]
    """

    # Unpack the valid LLA value range
    min_lat = valid_lla_range['min_lat']
    max_lat = valid_lla_range['max_lat']
    min_lon = valid_lla_range['min_lon']
    max_lon = valid_lla_range['max_lon']
    min_alt = valid_lla_range['min_alt']
    max_alt = valid_lla_range['max_alt']

    # Generate the random coordinates
    lat = random.uniform(min_lat, max_lat)
    lon = random.uniform(min_lon, max_lon)
    alt = random.uniform(min_alt, max_alt)
    return lat, lon, alt


@pytest.fixture
def random_lla_points(valid_lla_range, request) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return random latitude, longitude, altitude values.
    These values are stacked in a vertical

    :param valid_lla_range: A dictionary defining supported LLA positions.
    :type valid_lla_range: dict[str, float]

    :param request: Used to specify the number of points to generate.
    :type request: SubRequest

    :return: Tuple containing random latitude, longitude, altitude arrays.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    # Unpack the valid LLA value range
    min_lat = valid_lla_range['min_lat']
    max_lat = valid_lla_range['max_lat']
    min_lon = valid_lla_range['min_lon']
    max_lon = valid_lla_range['max_lon']
    min_alt = valid_lla_range['min_alt']
    max_alt = valid_lla_range['max_alt']

    # Produce required number of random coordinates
    num_points = request.param
    lat = np.random.uniform(min_lat, max_lat, num_points)
    lon = np.random.uniform(min_lon, max_lon, num_points)
    alt = np.random.uniform(min_alt, max_alt, num_points)
    # return lat, lon, alt
    return sort_via_tiles(lat, lon, alt)


@pytest.fixture
def random_outside_lla(invalid_lla_range) -> tuple[float, float, float]:
    """
    Returns an unsupported latitude, longitude, altitude values.
    This is used for obtaining a position which is known to be
    outside of a gravity map.

    :return: Tuple containing random latitude, longitude, altitude values.
    :rtype: tuple[float, float, float]
    """

    # Unpack the valid LLA value range
    min_lat = invalid_lla_range['min_lat']
    max_lat = invalid_lla_range['max_lat']
    min_lon = invalid_lla_range['min_lon']
    max_lon = invalid_lla_range['max_lon']
    min_alt = invalid_lla_range['min_alt']
    max_alt = invalid_lla_range['max_alt']

    # Generate the random coordinates
    lat = random.uniform(min_lat, max_lat)
    lon = random.uniform(min_lon, max_lon)
    alt = random.uniform(min_alt, max_alt)
    return lat, lon, alt


@pytest.fixture
def random_outside_lla_points(invalid_lla_range, request) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns a supported latitude, longitude, altitude values.
    This is used for obtaining a position which is known to be
    outside of a gravity map.

    :param invalid_lla_range: A dictionary defining unsupported LLA positions.
    :type invalid_lla_range: dict[str, float]

    :param request: Used to specify the number of points to generate.
    :type request: SubRequest

    :return: Tuple containing random latitude, longitude, altitude arrays.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    # Unpack the valid LLA value range
    min_lat = invalid_lla_range['min_lat']
    max_lat = invalid_lla_range['max_lat']
    min_lon = invalid_lla_range['min_lon']
    max_lon = invalid_lla_range['max_lon']
    min_alt = invalid_lla_range['min_alt']
    max_alt = invalid_lla_range['max_alt']

    # Produce required number of random coordinates
    num_points = request.param
    lat = np.random.uniform(min_lat, max_lat, num_points)
    lon = np.random.uniform(min_lon, max_lon, num_points)
    alt = np.random.uniform(min_alt, max_alt, num_points)
    # return lat, lon, alt
    return sort_via_tiles(lat, lon, alt)


@pytest.fixture(scope='module')
def database_dir(project_dir) -> Path:
    """
    Returns the local path for gravity database files.

    :param project_dir: The root directory of the project.
    :type project_dir: pathlib.Path

    :return: Local path for gravity database files.
    :rtype: pathlib.Path
    """
    return project_dir / 'databases' / 'gravity'
