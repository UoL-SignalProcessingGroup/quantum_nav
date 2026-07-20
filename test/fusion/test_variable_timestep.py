import inspect

import numpy as np
import pytest

import qnav.util.transformations as trans
from qnav.estimation.kalman import generate_matrices
from qnav.estimation.state import EstimatedState
from qnav.fusion.ins import (
    AdamsBashforth,
    NumericalINS,
    RungeKutta,
    _adams_bashforth_weights,
)
from qnav.fusion.kalman import (
    KalmanINS,
    _nominal_time_step,
    _process_noise_for_time_step,
    _transition_for_time_step,
)
from qnav.gravity.base import GravityModel
from qnav.measurement.platform import SensorAxis
from qnav.measurement.sensor import resolve_time_step
from qnav.waypoints.trajectory import GroundTruth


class ZeroGravity(GravityModel):
    def __str__(self) -> str:
        return "Zero gravity"

    def calc_gravity_z(self, lat, lon, alt) -> float:
        return 0.0


class RecordedSensor:
    """Minimal sensor interface for deterministic fusion tests."""

    def __init__(self, measurement, time_step=0.1):
        self.last_measurement = np.asarray(measurement, dtype=float)
        self.time_step = time_step
        self.frequency = 1.0 / time_step
        self.sensor_axis = SensorAxis()


def make_state() -> EstimatedState:
    truth = GroundTruth(
        timestamp=0.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        attitude=np.zeros(3),
        angle_rates=np.zeros(3),
    )
    return EstimatedState(truth, ZeroGravity())


def make_ins(ins_type):
    accelerometer = RecordedSensor([2.0, 0.0, 0.0])
    gyroscope = RecordedSensor([0.0, 0.0, 0.0])
    return ins_type(accelerometer, gyroscope)


@pytest.mark.parametrize("solver", [NumericalINS, RungeKutta,
                                     AdamsBashforth, KalmanINS])
def test_fusion_signature_keeps_optional_time_step(solver):
    parameter = inspect.signature(solver.perform_fusion).parameters["time_step"]
    assert parameter.default is None


@pytest.mark.parametrize("invalid", [0.0, -0.1, np.inf, np.nan, "0.1"])
def test_resolve_time_step_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="positive finite"):
        resolve_time_step(invalid, 0.1)


def test_numerical_ins_defaults_to_sensor_interval():
    default_state = make_state()
    explicit_state = make_state()

    make_ins(NumericalINS).perform_fusion(default_state)
    make_ins(NumericalINS).perform_fusion(explicit_state, time_step=0.1)

    np.testing.assert_allclose(default_state.as_numpy(),
                               explicit_state.as_numpy())


def test_adams_bootstrap_preserves_unequal_sensor_intervals():
    accelerometer = RecordedSensor([2.0, 0.0, 0.0], time_step=0.1)
    gyroscope = RecordedSensor([0.0, 0.0, 1.0], time_step=0.2)
    numerical_state = make_state()
    adams_state = make_state()

    with pytest.warns(UserWarning, match="different measurement frequencies"):
        numerical = NumericalINS(accelerometer, gyroscope)
    with pytest.warns(UserWarning, match="different measurement frequencies"):
        adams = AdamsBashforth(accelerometer, gyroscope)
    numerical.perform_fusion(numerical_state)
    adams.perform_fusion(adams_state)

    np.testing.assert_allclose(numerical_state.as_numpy(),
                               adams_state.as_numpy())


@pytest.mark.parametrize("solver", [NumericalINS, RungeKutta])
def test_single_step_solvers_use_explicit_interval(solver):
    initial_position = np.zeros(3)
    short_state = make_state()
    long_state = make_state()

    make_ins(solver).perform_fusion(short_state, time_step=0.1)
    make_ins(solver).perform_fusion(long_state, time_step=0.2)

    short_displacement = trans.lla2ned(short_state.position, initial_position)
    long_displacement = trans.lla2ned(long_state.position, initial_position)
    assert long_state.velocity[0] == pytest.approx(
        2.0 * short_state.velocity[0], rel=2e-4)
    assert long_displacement[0] == pytest.approx(
        4.0 * short_displacement[0], rel=2e-4)


def test_variable_step_adams_bashforth_weights():
    assert _adams_bashforth_weights(0.1, 0.1) == (1.5, -0.5)
    assert _adams_bashforth_weights(0.2, 0.1) == (2.0, -2.0)
    assert _adams_bashforth_weights(0.05, 0.1) == (1.25, -0.125)


def test_kalman_matrices_are_adjusted_without_mutating_state_matrices():
    _, _, transition, process_noise = generate_matrices(
        0.1, 1e-3, 1e-3)
    original_transition = transition.copy()
    original_process_noise = process_noise.copy()

    nominal_dt = _nominal_time_step(transition)
    adjusted_transition = _transition_for_time_step(
        transition, nominal_dt, 0.25)
    adjusted_process_noise = _process_noise_for_time_step(
        process_noise, nominal_dt, 0.25)

    assert nominal_dt == pytest.approx(0.1)
    assert adjusted_transition[0, 1] == pytest.approx(0.25)
    assert adjusted_transition[0, 2] == pytest.approx(0.5 * 0.25 ** 2)
    np.testing.assert_allclose(adjusted_process_noise, process_noise * 2.5)
    np.testing.assert_array_equal(transition, original_transition)
    np.testing.assert_array_equal(process_noise, original_process_noise)
