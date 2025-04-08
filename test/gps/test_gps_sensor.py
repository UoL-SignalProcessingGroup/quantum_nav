
import qnav.util.transformations as trans
import numpy as np
import pytest


from copy import deepcopy
from numpy.testing import assert_raises

from qnav.gps.sensor import GpsSensor
from qnav.waypoints.trajectory import GroundTruth



@pytest.mark.parametrize('freq', (0.1, 1.0, 100.0))
def test_timer(perfect_satellites, freq: float):

    num_iterations: int = 5
    sensor = GpsSensor(freq, perfect_satellites)

    dt = 1 / freq
    expected = dt

    for _ in range(num_iterations):
        assert sensor.next_update == expected
        sensor.update()
        expected += dt


@pytest.mark.repeat(3)
def test_last_measurement(perfect_satellites, random_truth: GroundTruth):

    sensor = GpsSensor(1.0, perfect_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    ant_lla = random_truth.position
    ant_att = random_truth.attitude
    ant_vel = random_truth.velocity

    for measurement, sat in zip(measurements, perfect_satellites):
        expected = sat.get_measurements(ant_lla, ant_att, ant_vel)

        for k, v in expected.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(measurement[k], v)
            else:
                assert measurement[k] == v


@pytest.mark.parametrize('dt', (0.1, 1.0, 10.0))
def test_update(perfect_satellites, dt: float):

    sensor = GpsSensor(1 / dt, perfect_satellites)
    satellites_copy = deepcopy(perfect_satellites)

    sensor.update()
    [s.update(dt) for s in satellites_copy]

    pos_1 = np.array([s.position for s in perfect_satellites])
    pos_2 = np.array([s.position for s in satellites_copy])
    np.testing.assert_array_equal(pos_1, pos_2)

    vel_1 = np.array([s.velocity for s in perfect_satellites])
    vel_2 = np.array([s.velocity for s in satellites_copy])
    np.testing.assert_array_equal(vel_1, vel_2)

    times_1 = [s.sat_time for s in perfect_satellites]
    times_2 = [s.sat_time for s in satellites_copy]
    for t1, t2 in zip(times_1, times_2):
        assert t1 == t2


@pytest.mark.repeat(3)
def test_perfect_measurement(perfect_satellites, random_truth: GroundTruth):

    sensor = GpsSensor(1.0, perfect_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    ant_lla = random_truth.position
    ant_att = random_truth.attitude
    ant_vel = random_truth.velocity

    for measurement, sat in zip(measurements, perfect_satellites):
        expected = sat.get_measurements(ant_lla, ant_att, ant_vel)

        for k, v in expected.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(measurement[k], v)
            else:
                assert measurement[k] == v


@pytest.mark.parametrize("noisy_satellites", [(0,0), (1,0), (0,1), (1,1)], indirect=True)
def test_noisy_measurement(noisy_satellites, random_truth: GroundTruth):

    sensor = GpsSensor(1.0, noisy_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    ant_lla = random_truth.position
    ant_att = random_truth.attitude
    ant_vel = random_truth.velocity

    for measurement, sat in zip(measurements, noisy_satellites):
        expected = sat.get_measurements(ant_lla, ant_att, ant_vel)

        for k, v in expected.items():
            if isinstance(v, np.ndarray):
                np.testing.assert_array_equal(measurement[k], v)
            else:
                assert measurement[k] == v


@pytest.mark.parametrize("noisy_satellites", [(1,1), (1,100), (100,1)], indirect=True)
def test_rho_errors(perfect_satellites, noisy_satellites, random_truth: GroundTruth):

    perfect_sensor = GpsSensor(1.0, perfect_satellites)
    noisy_sensor = GpsSensor(1.0, noisy_satellites)

    perfect_sensor.take_measurement(0, random_truth)
    noisy_sensor.take_measurement(0, random_truth)

    perfect_measurements = perfect_sensor.last_measurement
    noisy_measurements = noisy_sensor.last_measurement

    expect_different = ['pseudoRangeNoisy', 'pseudoRangeRatesNoisy']
    expect_same = ['time', 'satellitePosition', 'satelliteVelocity']

    for measurement_1, measurement_2 in zip(perfect_measurements, noisy_measurements):

        # Validate that properties which are expected different are different:
        for k in expect_same:
            if isinstance(measurement_1[k], np.ndarray):
                np.testing.assert_array_equal(measurement_1[k], measurement_2[k])
            else:
                assert measurement_1[k] == measurement_2[k]

        for k in expect_different:
            if isinstance(measurement_1[k], np.ndarray):
                assert_raises(AssertionError, np.testing.assert_array_equal,
                              measurement_1[k], measurement_2[k])
            else:
                assert measurement_1[k] != measurement_2[k]

@pytest.mark.repeat(100)
def test_perfect_pseudo_ranges(perfect_satellites, random_truth: GroundTruth):

    # Initialise and set-up the satellite sensor
    sensor = GpsSensor(1.0, perfect_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    # Obtain the antenna position
    ant_pos_lla = random_truth.position
    ant_pos_ecef = trans.lla2ecef(ant_pos_lla)

    # For each satellite measurement
    for measurement in measurements:

        # Verify the distances are the same
        sat_pos_ecef = measurement['satellitePosition']
        pseudo_range = measurement['pseudoRangeNoisy']
        true_range = np.linalg.norm(sat_pos_ecef - ant_pos_ecef)
        np.testing.assert_array_almost_equal(
            pseudo_range, true_range, err_msg='Ranges do not match')


@pytest.mark.parametrize("noisy_satellites", [(1,0), (1,1)], indirect=True)
def test_noisy_pseudo_ranges(noisy_satellites, random_truth: GroundTruth):

    # Initialise and set-up the satellite sensor
    sensor = GpsSensor(1.0, noisy_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    # Obtain the antenna position
    ant_pos_lla = random_truth.position
    ant_pos_ecef = trans.lla2ecef(ant_pos_lla)

    # For each satellite measurement
    for measurement in measurements:

        # Verify the distances are the same
        sat_pos_ecef = measurement['satellitePosition']
        pseudo_range = measurement['pseudoRangeNoisy']
        true_range = np.linalg.norm(sat_pos_ecef - ant_pos_ecef)
        assert pseudo_range != true_range, 'Ranges should not match'


@pytest.mark.repeat(100)
def test_perfect_pseudo_range_rates(perfect_satellites, random_truth: GroundTruth):

    # Initialise and set-up the satellite sensor
    sensor = GpsSensor(1.0, perfect_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    # Obtain the antenna position, velocity and attitude
    ant_pos_lla = random_truth.position
    ant_vel_body = random_truth.velocity
    ant_att_ned = random_truth.attitude

    # Convert values from local to ECEF frame
    rot_mat = trans.rotate_3d(*np.radians(ant_att_ned))
    ant_pos_ecef = trans.lla2ecef(ant_pos_lla)
    ant_vel_ned = np.linalg.inv(rot_mat) @ ant_vel_body
    ant_vel_ecef = trans.vned2vecef(ant_vel_ned, ant_pos_lla)
    # ant_pos_ecef_next = ant_pos_ecef + ant_vel_ecef

    # For each satellite measurement
    for measurement in measurements:

        # Calculate differences in position and velocity
        sat_pos_ecef = measurement['satellitePosition']
        sat_vel_ecef = measurement['satelliteVelocity']
        pseudo_range_rate = measurement['pseudoRangeRatesNoisy']

        diff_pos = sat_pos_ecef - ant_pos_ecef
        diff_vel = sat_vel_ecef - ant_vel_ecef
        los = diff_pos / np.linalg.norm(diff_pos)
        true_range_rate = los.dot(diff_vel)

        # sat_pos_ecef_next = sat_pos_ecef + sat_vel_ecef
        # range_before = np.linalg.norm(sat_pos_ecef - ant_pos_ecef)
        # range_after = np.linalg.norm(sat_pos_ecef_next - ant_pos_ecef_next)
        # true_range_rate = range_after - range_before

        # Verify the range rates are the same
        np.testing.assert_array_almost_equal(
            pseudo_range_rate, true_range_rate,
            err_msg='Range rates do not match')


@pytest.mark.parametrize("noisy_satellites", [(0,1), (1,1)], indirect=True)
def test_noisy_pseudo_range_rates(noisy_satellites, random_truth: GroundTruth):

    # Initialise and set-up the satellite sensor
    sensor = GpsSensor(1.0, noisy_satellites)
    sensor.take_measurement(0, random_truth)
    measurements = sensor.last_measurement

    # Obtain the antenna position, velocity and attitude
    ant_pos_lla = random_truth.position
    ant_vel_body = random_truth.velocity
    ant_att_ned = random_truth.attitude

    # Convert values from local to ECEF frame
    rot_mat = trans.rotate_3d(*np.radians(ant_att_ned))
    ant_pos_ecef = trans.lla2ecef(ant_pos_lla)
    ant_vel_ned = np.linalg.inv(rot_mat) @ ant_vel_body
    ant_vel_ecef = trans.vned2vecef(ant_vel_ned, ant_pos_lla)

    # For each satellite measurement
    for measurement in measurements:

        # Calculate differences in position and velocity
        sat_pos_ecef = measurement['satellitePosition']
        sat_vel_ecef = measurement['satelliteVelocity']
        pseudo_range_rate = measurement['pseudoRangeRatesNoisy']

        diff_pos = sat_pos_ecef - ant_pos_ecef
        diff_vel = sat_vel_ecef - ant_vel_ecef
        los = diff_pos / np.linalg.norm(diff_pos)
        true_range_rate = los.dot(diff_vel)

        # Verify the range rates are the same
        assert pseudo_range_rate != true_range_rate, \
            'Range rates should not match'

