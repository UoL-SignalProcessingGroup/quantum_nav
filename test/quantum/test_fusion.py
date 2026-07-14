from unittest.mock import patch

import numpy as np

from qnav.estimation.state import EstimatedState
from qnav.fusion.ins import NumericalINS
from qnav.measurement.recorded import (
    RecordedAccelerometer,
    RecordedGyroscope,
    RecordedQuantumImu,
)
from qnav.quantum.base import ConceptQuantumFusion
from qnav.util import transformations as trans
from qnav.waypoints.trajectory import GroundTruth


def _state() -> EstimatedState:
    return EstimatedState(GroundTruth(
        timestamp=0.0,
        position=np.array([52.0, -2.0, 100.0]),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        attitude=np.zeros(3),
        angle_rates=np.zeros(3),
    ))


def _fusion() -> tuple[
        ConceptQuantumFusion, RecordedQuantumImu,
        RecordedAccelerometer, RecordedGyroscope]:
    accelerometer = RecordedAccelerometer(frequency=3.0)
    gyroscope = RecordedGyroscope(frequency=3.0)
    quantum_imu = RecordedQuantumImu(frequency=3.0, full_frequency=1.0)
    fusion = ConceptQuantumFusion(
        quantum_imu, accelerometer, gyroscope)
    return fusion, quantum_imu, accelerometer, gyroscope


def test_asynchronous_quantum_measurement_consumes_completed_imu_window():
    fusion, quantum_imu, accelerometer, gyroscope = _fusion()
    state = _state()
    for timestamp in (0.0, 1.0, 2.0):
        accelerometer.push(timestamp, np.zeros(3))
        gyroscope.push(timestamp, np.zeros(3))
        assert not fusion.perform_fusion(state)

    quantum_imu.push(2.1, np.zeros(3), np.zeros(3))
    with patch.object(ConceptQuantumFusion, "_perform_correction") as correction:
        assert fusion.apply_pending_measurement(state)

    correction.assert_called_once()
    assert quantum_imu.last_measurement is None
    assert fusion._step == 0
    assert not fusion._window_full


def test_delayed_quantum_measurement_does_not_drop_simultaneous_imu_pair():
    fusion, quantum_imu, accelerometer, gyroscope = _fusion()
    state = _state()
    for timestamp in (0.0, 1.0, 2.0):
        accelerometer.push(timestamp, np.full(3, timestamp))
        gyroscope.push(timestamp, np.full(3, timestamp))
        assert not fusion.perform_fusion(state)

    quantum_imu.push(3.0, np.zeros(3), np.zeros(3))
    accelerometer.push(3.0, np.full(3, 3.0))
    gyroscope.push(3.0, np.full(3, 3.0))
    with patch.object(ConceptQuantumFusion, "_perform_correction"):
        assert fusion.apply_pending_measurement(state)
    assert not fusion.perform_fusion(state)

    assert fusion._step == 1
    np.testing.assert_allclose(fusion._imu_acc_data[0], 3.0)
    np.testing.assert_allclose(fusion._imu_gyro_data[0], 3.0)


def test_quantum_correction_preserves_aiding_applied_inside_window():
    fusion, _, _, _ = _fusion()
    fusion._estimated_state = _state()
    live_state = _state()
    live_state.update_estimates(
        timestamp=2.0,
        position=np.array([53.0, -1.0, 321.0]),
        velocity=np.array([4.0, 5.0, 6.0]),
        acceleration=np.array([0.1, 0.2, 0.3]),
        attitude=np.array([20.0, -5.0, 3.0]),
        angle_rates=np.array([1.0, 2.0, 3.0]),
    )

    with patch.object(NumericalINS, "perform_fusion"):
        fusion._perform_correction(np.zeros(3), np.zeros(3), live_state)

    assert live_state.timestamp == 2.0
    np.testing.assert_allclose(live_state.position, [53.0, -1.0, 321.0])
    np.testing.assert_allclose(live_state.velocity, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(live_state.acceleration, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(live_state.attitude, [20.0, -5.0, 3.0])
    np.testing.assert_allclose(live_state.angle_rates, [1.0, 2.0, 3.0])


def test_quantum_delta_composes_aided_vectors_in_navigation_frame():
    fusion, _, _, _ = _fusion()
    fusion._estimated_state = _state()
    nominal_state = _state()
    nominal_state.update_estimates(
        velocity=np.array([1.0, 0.0, 0.0]),
        acceleration=np.array([0.5, 0.0, 0.0]),
    )
    corrected_state = _state()
    corrected_attitude = np.array([90.0, 0.0, 0.0])
    corrected_rotation = trans.rotate_3d(*np.radians(corrected_attitude))
    corrected_state.update_estimates(
        attitude=corrected_attitude,
        velocity=corrected_rotation @ np.array([1.0, 0.0, 0.0]),
        acceleration=corrected_rotation @ np.array([0.5, 0.0, 0.0]),
    )
    live_state = _state()
    live_state.update_estimates(
        velocity=np.array([2.0, 0.0, 0.0]),
        acceleration=np.array([1.5, 0.0, 0.0]),
    )

    with patch.object(
            fusion, "_reprocess_window",
            side_effect=[nominal_state, corrected_state]):
        fusion._perform_correction(np.zeros(3), np.zeros(3), live_state)

    np.testing.assert_allclose(live_state.attitude, corrected_attitude)
    np.testing.assert_allclose(
        live_state.velocity,
        corrected_rotation @ np.array([2.0, 0.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(
        live_state.acceleration,
        corrected_rotation @ np.array([1.5, 0.0, 0.0]), atol=1e-12)
