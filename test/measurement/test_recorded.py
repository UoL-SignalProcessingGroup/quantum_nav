import numpy as np
import pytest

from qnav.measurement.platform import SensorAxis
from qnav.measurement.recorded import (
    RecordedGravityGradiometer,
    RecordedQuantumImu,
    RecordedScalarSensor,
    RecordedVectorSensor,
)


def test_vector_push_updates_measurement_and_clock():
    axis = SensorAxis(sensor_angles=[1, 2, 3])
    sensor = RecordedVectorSensor(frequency=10, sensor_axis=axis)
    source = np.array([1.0, 2.0, 3.0])

    sensor.push(4.0, source)
    source[:] = 0
    np.testing.assert_array_equal(sensor.last_measurement, [1, 2, 3])
    assert sensor.timestamp == 4.0
    assert sensor.sensor_axis is axis

    sensor.push(4.25, [4, 5, 6])
    assert sensor.time_step == pytest.approx(0.25)
    assert sensor.frequency == pytest.approx(4.0)
    assert sensor.last_update == pytest.approx(4.25)
    assert sensor.next_update == pytest.approx(4.5)


def test_recorded_sensor_rejects_bad_values_and_backwards_time():
    sensor = RecordedVectorSensor()
    sensor.push(1, [1, 2, 3])

    with pytest.raises(ValueError, match="monotonic"):
        sensor.push(0.5, [1, 2, 3])
    with pytest.raises(ValueError, match="3 finite"):
        sensor.push(2, [1, np.nan, 3])


def test_scalar_reset_only_consumes_value():
    sensor = RecordedScalarSensor(frequency=2)
    sensor.push(3, 123.5)
    sensor.reset()

    assert sensor.last_measurement is None
    assert sensor.timestamp == 3
    assert sensor.time_step == 0.5


def test_quantum_and_gradiometer_proxy_shapes():
    quantum = RecordedQuantumImu()
    quantum.push(1, [1, 2, 3], [4, 5, 6])
    acceleration, rates = quantum.last_measurement
    np.testing.assert_array_equal(acceleration, [1, 2, 3])
    np.testing.assert_array_equal(rates, [4, 5, 6])
    assert quantum.num_active_steps == quantum.num_steps == 1

    gradiometer = RecordedGravityGradiometer()
    assert gradiometer.push(2, (0.1, 0.2)) == (0.1, 0.2)
    assert gradiometer.num_steps == 1

