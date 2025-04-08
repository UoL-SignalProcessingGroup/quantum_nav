
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from random import randint

import pytest

from qnav.gps import ephemeris
from qnav.gps.satellite import Satellite


@pytest.fixture()
def local_rinex_file() -> Path:
    """
    Returns the path for a local Rinex file to perform testing with.

    :return: The path for a local pre-downloaded Rinex file.
    :rtype: Path
    """
    test_file = Path(__file__).parent / '2024_278.rnx'
    if not test_file.exists():
        pytest.skip("Missing example Rinex file")
    return test_file


@pytest.fixture()
def tmp_date() -> datetime:
    """
    Generates and returns a random supported datetime instance.
    Produces a random datetime instance from 01/01/2020 to the current system
    date. Precision is only given to day (no seconds). Used for loading adhoc
    data from Rinex file testing.

    :return: A random date from 01/01/2020 to the current system date.
    :rtype: datetime.datetime
    """
    min_date = datetime(year=2020, month=1, day=1)
    max_date = datetime.now()
    return _get_rand_date_between(min_date, max_date)


@pytest.fixture()
def invalid_date() -> datetime:
    """
    Generates and returns a random unsupported datetime instance.
    Produces a random date before the existence of GPS. Used for
    attempting to load invalid adhoc data for testing.

    :return: A random date from 01/01/0000 and 21/02/1978.
    :rtype: datetime.datetime
    """
    min_date = datetime(year=1900, month=1, day=1)
    max_date = datetime(year=1978, month=2, day=21)
    return _get_rand_date_between(min_date, max_date)


@pytest.fixture
def perfect_satellites(local_rinex_file) -> list[Satellite]:
    """
    Satellites extracted from the local test rinex file.
    A set of satellite instances that we are familiar with.

    :param local_rinex_file: The location of a test rinex file.
    :type local_rinex_file: Path

    :return: Satellites from local Rinex file.
    :rtype: list[Satellite]
    """

    # Extract satellite instance from local rinex file.
    rinex_data = ephemeris.read_rinex_file(local_rinex_file)
    parameters = ephemeris.rinex_to_satellite_params(rinex_data)
    return [Satellite(p) for p in parameters]


@pytest.fixture
def noisy_satellites(local_rinex_file, request):
    """
    Satellites extracted from the local test rinex file.
    A set of satellite instances that we are familiar with.

    :param local_rinex_file: The location of a test rinex file.
    :type local_rinex_file: Path

    :param request: Additional fixture parameters.
        These contain the noise properties to apply.
    :type request: pytest.Fixture

    :return: Satellites from local Rinex file.
    :rtype: list[Satellite]
    """
    # Use default GPS time.
    gps_time = None

    # Extract the requested noise properties.
    rho_noise, rho_rate = request.param
    rinex_data = ephemeris.read_rinex_file(local_rinex_file)
    parameters = ephemeris.rinex_to_satellite_params(rinex_data)

    # Extract satellite instance from local rinex file with parameters
    return [Satellite(p, gps_time, rho_noise, rho_rate) for p in parameters]


def _get_rand_date_between(min_date: datetime, max_date: datetime) -> datetime:
    """
    Generates a random date between two dates.
    Given two dates that define a minimum and maximum bound, this function
    randomly chooses and returns a date between them.

    :param min_date: The earliest of the two dates.
    :type min_date: datetime.datetime

    :param max_date: The latest of the two dates.
    :type max_date: datetime.datetime

    :return: A random date between given date bounds.
    :rtype: datetime.datetime
    """
    num_days = (max_date - min_date).days
    rand_days = randint(0, num_days)
    return min_date + timedelta(days=rand_days)
