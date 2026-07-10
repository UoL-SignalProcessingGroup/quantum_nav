from unittest.mock import Mock

import numpy as np
import pytest

from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.guassian_markov import AccelerometerGM, GyroscopeGM


@pytest.mark.parametrize("sensor_type", [AccelerometerGM, GyroscopeGM])
def test_explicit_update_time_drives_gaussian_markov_clock(sensor_type):
    sensor = sensor_type(
        frequency=10.0,
        error_profile=ErrorProperties(avg_meas_noise=1.0),
        correlation_time=5.0,
        pow_spec_density=1.0,
        rand_seed=1,
    )
    sensor._gm_rng = Mock()
    sensor._gm_rng.get_random.return_value = np.ones(3)

    sensor.update(0.35)

    sensor._gm_rng.get_random.assert_called_once_with(0.35)
    assert sensor.last_update == pytest.approx(0.35)


@pytest.mark.parametrize("sensor_type", [AccelerometerGM, GyroscopeGM])
def test_default_update_preserves_nominal_gaussian_markov_clock(sensor_type):
    sensor = sensor_type(
        frequency=10.0,
        error_profile=ErrorProperties(avg_meas_noise=1.0),
        correlation_time=5.0,
        pow_spec_density=1.0,
        rand_seed=1,
    )
    sensor._gm_rng = Mock()
    sensor._gm_rng.get_random.return_value = np.ones(3)

    sensor.update()

    sensor._gm_rng.get_random.assert_called_once_with(sensor.time_step)
    assert sensor.last_update == pytest.approx(sensor.time_step)
