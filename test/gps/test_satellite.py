

import pytest
import numpy as np

from qnav.gps import ephemeris
from qnav.gps.ephemeris import SatelliteType
from qnav.gps.satellite import Satellite

from qnav.util.constants import POLAR_AXIS_A
from qnav.util.constants import POLAR_AXIS_B

from datetime import datetime, timedelta
from typing import Optional
from copy import deepcopy
from pathlib import Path


GPS = SatelliteType.GPS

def extract_satellites(rinex_file: Path, rinex_filter: SatelliteType = GPS) -> list[Satellite]:
    """

    :param rinex_file:
    :param rinex_filter:
    :return:
    """
    # Attempt to read the given rinex file.
    rinex_data = ephemeris.read_rinex_file(rinex_file)

    # Convert to satellite parameters for GPS satellites
    params = ephemeris.rinex_to_satellite_params(
        rinex_data, rinex_filter)

    # Obtain satellite instances for each
    satellites = [Satellite(p) for p in params]
    return satellites


def test_satellite_init(local_rinex_file):

    # Obtain satellite instances from given file
    satellites = extract_satellites(local_rinex_file)

    # Get the position and velocity for each
    pos_ecef = np.array([s.position for s in satellites])
    vel_ecef = np.array([s.velocity for s in satellites])

    # Ensure all values are finite
    assert np.all(np.isfinite(pos_ecef)), 'non-finite position values'
    assert np.all(np.isfinite(vel_ecef)), 'non-finite velocity values'


@pytest.mark.parametrize("dt", (0.1, 1.0, 100.0, 10000.0))
def test_satellite_update(local_rinex_file, dt: float):

    # Obtain satellite instances from given file
    satellites = extract_satellites(local_rinex_file)

    # Get the position and velocity for each
    pos_before = np.array([s.position for s in satellites])
    vel_before = np.array([s.velocity for s in satellites])

    # Update satellites by one second.
    [s.update(dt) for s in satellites]

    # Get the position and velocity for each
    pos_after = np.array([s.position for s in satellites])
    vel_after = np.array([s.velocity for s in satellites])

    assert np.all(pos_before != pos_after), \
        'some positions have not changed'

    assert np.all(vel_before != vel_after), \
        'some velocities have not changed'


@pytest.mark.parametrize("dt,num_steps", [(0.1, 10), (1.0, 10), (10.0, 100)])
def test_satellite_update_steps(local_rinex_file, dt: float, num_steps: int):

    # Obtain satellite instances and a copy of the instances
    satellites_1 = extract_satellites(local_rinex_file)
    satellites_2 = deepcopy(satellites_1)

    # Update the first set of satellites
    [s.update(dt) for s in satellites_1]

    # Update the second set of satellites
    for _ in range(num_steps):
        [s.update(dt / num_steps) for s in satellites_2]

    # Get the positions for each satellite set
    positions_1 = np.array([s.position for s in satellites_1])
    positions_2 = np.array([s.position for s in satellites_2])

    # Ensure the updated positions are the same
    np.testing.assert_array_equal(positions_1, positions_2)

    # Get the velocities for each satellite set
    velocity_1 = np.array([s.velocity for s in satellites_1])
    velocity_2 = np.array([s.velocity for s in satellites_2])

    # Ensure the updated velocities are the same
    np.testing.assert_array_equal(velocity_1, velocity_2)


@pytest.mark.parametrize("sat_time", [
    datetime(2000, 1, 2, 3,4, 5),
    datetime(2030, 6, 7, 8,9, 10),
    None
])
def test_satellite_clock(local_rinex_file, sat_time: Optional[datetime]):

    # Attempt to read the given rinex file.
    rinex_data = ephemeris.read_rinex_file(local_rinex_file)

    # Convert to satellite parameters for GPS satellites
    params = ephemeris.rinex_to_satellite_params(rinex_data)

    # Obtain satellite instances for each
    satellites = [Satellite(p, sat_time) for p in params]

    # For each parameter set and satellite instance
    for param, sat in zip(params, satellites):

        expected = sat_time
        if sat_time is None:
            expected = datetime(
                param.year, param.month, param.day,
                param.hour, param.minute, param.second)

        assert sat.sat_time == expected

@pytest.mark.parametrize("rho_noise", [0, 100])
@pytest.mark.parametrize("rho_noise_rate", [0, 100])
def test_satellite_errors(local_rinex_file, rho_noise: float, rho_noise_rate: float):

    # Attempt to read the given rinex file.
    rinex_data = ephemeris.read_rinex_file(local_rinex_file)

    # Convert to satellite parameters for GPS satellites
    params = ephemeris.rinex_to_satellite_params(rinex_data)

    # Get remaining satellite parameters
    gps_time = None

    # Obtain satellite instances for each
    satellites = [Satellite(p, gps_time, rho_noise, rho_noise_rate) for p in params]

    # Ensure parameters are set correctly
    for sat in satellites:
        assert sat.pseudo_range_noise == rho_noise
        assert sat.pseudo_range_rate_noise == rho_noise_rate

@pytest.mark.slow
@pytest.mark.parametrize("dt", [1],)
@pytest.mark.parametrize("duration", [timedelta(days=7),])
def test_satellite_long_duration(local_rinex_file, dt: float, duration: timedelta):

    earth_radius = (POLAR_AXIS_A + POLAR_AXIS_B) / 2
    max_radius = 3.6e8 + 1000
    max_velocity = 28.8e6 + 1000

    # Obtain satellite instances from given file
    satellites = extract_satellites(local_rinex_file)
    total_updates: int = int(duration.total_seconds() // dt)

    # For each time step
    for i in range(total_updates):

        # Get the update positions and velocities
        [sat.update(dt) for sat in satellites]
        pos = np.vstack([sat.position for sat in satellites])
        vel = np.vstack([sat.velocity for sat in satellites])

        pos_norm = np.linalg.norm(pos, axis=1)
        vel_norm = np.linalg.norm(vel, axis=1)

        np.testing.assert_array_less(
            earth_radius, pos_norm,
            "Satellite distance less than earth radius")

        np.testing.assert_array_less(
            pos_norm, max_radius,
            "Satellite distance too large")

        np.testing.assert_array_less(
            vel_norm, max_velocity,
            "Satellite distance too large")




# @pytest.mark.parametrize("rho_noise", [0, 1])
# @pytest.mark.parametrize("rho_rate", [0, 1])
# def test_rho_errors(local_rinex_file, rho_noise: float, rho_rate: float):
#
#
#     # Attempt to read the given rinex file.
#     rinex_data = ephemeris.read_rinex_file(local_rinex_file)
#
#     # Obtain satellite instances for each with given parameters
#     sat_time = None
#     params = ephemeris.rinex_to_satellite_params(rinex_data)
#     satellites = [Satellite(p, sat_time, rho_noise, rho_rate) for p in params]
#
#     for sat in satellites:
#         measurement = sat.get_measurements()



# def test_thing(local_rinex_file):
#
#
#     # Attempt to read the pre-downloaded rinex file.
#     rinex_filter = ephemeris.SatelliteType.GPS
#     rinex_data = ephemeris.read_rinex_file(local_rinex_file)
#
#     # Convert to satellite parameters for GPS satellites
#     params = ephemeris.rinex_to_satellite_params(
#         rinex_data, rinex_filter)
#
#     # Obtain satellite instances for each
#     satellites = [Satellite(p) for p in params]
#
#     pos = np.array([52, -3, 1000])
#     att = np.array([0.0, 0.0, 0.0])
#     vel = np.array([0.0, 0.0, 0.0])
#
#     # Get the position and velocity for each
#     measurements = [s.get_measurements(pos, att, vel) for s in satellites]

