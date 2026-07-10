from types import SimpleNamespace

import numpy as np

from qnav.estimation.state import EstimatedState, KalmanEstimatedState
from qnav.gps.recorded import (
    RecordedGnssFixedGainFusion,
    RecordedGnssLooseFusion,
    RecordedGnssSensor,
)
from qnav.util.transformations import lla2ned, ned2lla
from qnav.waypoints.trajectory import GroundTruth


def _truth(position=(53.0, -2.0, 100.0)):
    return GroundTruth(
        timestamp=0,
        position=np.asarray(position, dtype=float),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        attitude=np.zeros(3),
        angle_rates=np.zeros(3),
    )


def _kalman_state(truth, covariance=1e4):
    return KalmanEstimatedState(
        truth,
        np.zeros((6, 15)),
        np.eye(6),
        np.eye(15),
        np.eye(15),
        np.eye(15) * covariance,
    )


def test_fixed_gain_consumes_duck_typed_fix():
    state = EstimatedState(_truth())
    sensor = RecordedGnssSensor()
    fix = {
        "latitude": 54,
        "longitude": -1,
        "altitude": 110,
        "velocity_ned": [2, 3, 4],
        "valid": True,
    }
    sensor.push(1, fix)
    fusion = RecordedGnssFixedGainFusion(sensor, state, gain_amount=0.25)

    fusion.perform_fusion(state)

    np.testing.assert_allclose(state.position, [53.25, -1.75, 102.5])
    np.testing.assert_allclose(state.velocity, [0.5, 0.75, 1.0])


def test_fixed_gain_ignores_explicitly_invalid_fix():
    state = EstimatedState(_truth())
    original = state.as_numpy()
    sensor = RecordedGnssSensor()
    sensor.push(1, SimpleNamespace(position=[54, -1, 110], valid=False))

    RecordedGnssFixedGainFusion(sensor, state, gain_amount=1).perform_fusion(state)

    np.testing.assert_array_equal(state.as_numpy(), original)


def test_missing_validity_is_usable_but_rejected_ingestion_fix_is_not():
    state = EstimatedState(_truth())
    sensor = RecordedGnssSensor()
    sensor.push(
        1,
        SimpleNamespace(position=[54, -1, 110], valid=None, accepted=True),
    )
    fusion = RecordedGnssFixedGainFusion(sensor, state, gain_amount=1)
    fusion.perform_fusion(state)
    np.testing.assert_array_equal(state.position, [54, -1, 110])

    sensor.push(
        2,
        SimpleNamespace(position=[55, 0, 120], valid=True, accepted=False),
    )
    fusion.perform_fusion(state)
    np.testing.assert_array_equal(state.position, [54, -1, 110])


def test_loose_fusion_uses_fix_accuracy_and_reduces_position_error():
    truth = _truth()
    state = _kalman_state(truth)
    state.update_estimates(position=ned2lla([100, -50, 20], truth.position))
    sensor = RecordedGnssSensor()
    sensor.push(
        1,
        SimpleNamespace(
            position=truth.position,
            velocity=np.zeros(3),
            valid=True,
            horizontal_accuracy=1,
            vertical_accuracy=2,
            speed_accuracy=0.5,
        ),
    )
    before = np.linalg.norm(lla2ned(state.position, truth.position))

    RecordedGnssLooseFusion(sensor, state).perform_fusion(state)

    after = np.linalg.norm(lla2ned(state.position, truth.position))
    assert after < before
    assert np.trace(state.state_errors) < 15 * 1e4


def test_loose_fusion_supports_position_only_fix_and_fallback_sigmas():
    truth = _truth()
    state = _kalman_state(truth)
    state.update_estimates(position=ned2lla([10, 0, 0], truth.position))
    sensor = RecordedGnssSensor()
    sensor.push(1, SimpleNamespace(position_lla=truth.position))

    RecordedGnssLooseFusion(
        sensor, state, sigma_position=2, sigma_vertical=3
    ).perform_fusion(state)

    assert np.linalg.norm(lla2ned(state.position, truth.position)) < 10
