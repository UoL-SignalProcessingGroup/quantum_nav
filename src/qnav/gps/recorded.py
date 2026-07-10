"""Fusion of recorded GNSS receiver fixes with navigation estimates."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import numpy as np

from qnav.estimation.state import EstimatedState, KalmanEstimatedState
from qnav.fusion.kalman import ned2state_vector, state_vector2ned
from qnav.measurement.recorded import _RecordedSensor
from qnav.measurement.sensor import FusionTrigger, SensorFusion
import qnav.util.transformations as trans


_MISSING = object()


def _field(fix: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(fix, Mapping) and name in fix:
            return fix[name]
        if hasattr(fix, name):
            return getattr(fix, name)
    if default is not _MISSING:
        return default
    raise ValueError(f"GNSS fix is missing required field {names[0]!r}")


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"GNSS {name} must have 3 finite values")
    return result


def _position(fix: Any) -> np.ndarray:
    value = _field(fix, "position", "position_lla", "lla", default=None)
    if value is None:
        value = (
            _field(fix, "latitude", "lat"),
            _field(fix, "longitude", "lon", "lng"),
            _field(fix, "altitude", "alt"),
        )
    position = _vector(value, "position")
    if not -90 <= position[0] <= 90:
        raise ValueError("GNSS latitude must be between -90 and 90 degrees")
    if not -180 <= position[1] <= 180:
        raise ValueError("GNSS longitude must be between -180 and 180 degrees")
    return position


def _velocity_ned(fix: Any) -> np.ndarray | None:
    value = _field(fix, "velocity_ned", "velocity", default=None)
    return None if value is None else _vector(value, "NED velocity")


def _is_usable(fix: Any) -> bool:
    valid_value = _field(fix, "valid", "is_usable", default=True)
    accepted_value = _field(fix, "accepted", default=True)
    # A missing validity column is represented by None and is not an explicit
    # receiver rejection.  Only an actual false value holds the update.
    valid = True if valid_value is None else bool(valid_value)
    accepted = True if accepted_value is None else bool(accepted_value)
    return valid and accepted


def _accuracy(fix: Any, names: tuple[str, ...], fallback: float) -> float:
    value = _field(fix, *names, default=None)
    result = fallback if value is None else float(value)
    if not isfinite(result) or result < 0:
        raise ValueError(f"GNSS {names[0]} must be a non-negative finite value")
    return result


class RecordedGnssSensor(_RecordedSensor):
    """Sensor-shaped holder for a pushed, already-solved receiver fix."""

    def push(self, timestamp: float, fix: Any) -> Any:
        # Validate required data at the ingestion boundary.  Optional fields
        # remain duck typed so the raw input layer can own its public record.
        _position(fix)
        velocity = _field(fix, "velocity_ned", "velocity", default=None)
        if velocity is not None:
            _vector(velocity, "NED velocity")
        return self._push(timestamp, fix)


class RecordedGnssFixedGainFusion(SensorFusion):
    """Apply fixed-gain corrections from receiver LLA/NED solutions."""

    def __init__(
        self,
        gnss_sensor: RecordedGnssSensor,
        estimated_state: EstimatedState | None = None,
        gain_amount: float = 0.1,
        velocity_gain: float | None = None,
    ):
        super().__init__(FusionTrigger.ANY, gnss_sensor)
        if not 0 <= gain_amount <= 1:
            raise ValueError("gain_amount must be between 0 and 1")
        if velocity_gain is None:
            velocity_gain = gain_amount
        if not 0 <= velocity_gain <= 1:
            raise ValueError("velocity_gain must be between 0 and 1")
        self._gnss_sensor = gnss_sensor
        self._gain_amount = float(gain_amount)
        self._velocity_gain = float(velocity_gain)

    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        fix = self._gnss_sensor.last_measurement
        if fix is None or not _is_usable(fix):
            return

        ins_position = estimated_state.position
        fix_position = _position(fix)
        position = ins_position + self._gain_amount * (fix_position - ins_position)

        updates: dict[str, np.ndarray] = {"position": position}
        fix_velocity_ned = _velocity_ned(fix)
        if fix_velocity_ned is not None:
            rotation_earth_to_body = trans.rotate_3d(
                *np.radians(estimated_state.attitude)
            )
            fix_velocity_body = rotation_earth_to_body @ fix_velocity_ned
            ins_velocity = estimated_state.velocity
            updates["velocity"] = ins_velocity + self._velocity_gain * (
                fix_velocity_body - ins_velocity
            )

        estimated_state.update_estimates(**updates)


class RecordedGnssLooseFusion(SensorFusion):
    """Covariance-aware loose Kalman update from receiver fixes."""

    def __init__(
        self,
        gnss_sensor: RecordedGnssSensor,
        estimated_state: KalmanEstimatedState,
        sigma_position: float = 50.0,
        sigma_velocity: float = 0.3,
        sigma_vertical: float | None = None,
    ):
        super().__init__(FusionTrigger.ANY, gnss_sensor)
        if not isinstance(estimated_state, KalmanEstimatedState):
            raise ValueError("estimated_state must be KalmanEstimatedState")
        if sigma_vertical is None:
            sigma_vertical = sigma_position
        for name, value in (
            ("sigma_position", sigma_position),
            ("sigma_vertical", sigma_vertical),
            ("sigma_velocity", sigma_velocity),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative finite value")
        self._gnss_sensor = gnss_sensor
        self._sigma_position = float(sigma_position)
        self._sigma_vertical = float(sigma_vertical)
        self._sigma_velocity = float(sigma_velocity)

    @staticmethod
    def _measurement_matrix(include_velocity: bool) -> np.ndarray:
        rows = 6 if include_velocity else 3
        matrix = np.zeros((rows, 15))
        matrix[0, 0] = 1
        matrix[1, 3] = 1
        matrix[2, 6] = 1
        if include_velocity:
            matrix[3, 1] = 1
            matrix[4, 4] = 1
            matrix[5, 7] = 1
        return matrix

    def perform_fusion(self, estimated_state: KalmanEstimatedState) -> None:
        if not isinstance(estimated_state, KalmanEstimatedState):
            raise ValueError("estimated_state must be KalmanEstimatedState")
        fix = self._gnss_sensor.last_measurement
        if fix is None or not _is_usable(fix):
            return

        fix_position = _position(fix)
        fix_velocity = _velocity_ned(fix)
        include_velocity = fix_velocity is not None
        h_matrix = self._measurement_matrix(include_velocity)

        horizontal_sigma = _accuracy(
            fix, ("horizontal_accuracy", "horizontal_sigma"), self._sigma_position
        )
        vertical_sigma = _accuracy(
            fix, ("vertical_accuracy", "vertical_sigma"), self._sigma_vertical
        )
        variances = [horizontal_sigma**2, horizontal_sigma**2, vertical_sigma**2]
        if include_velocity:
            speed_sigma = _accuracy(
                fix, ("speed_accuracy", "velocity_accuracy"), self._sigma_velocity
            )
            variances.extend([speed_sigma**2] * 3)
        measurement_covariance = np.diag(variances)

        reference_lla = estimated_state.position
        gravity_model = estimated_state.gravity_model
        state_ned = state_vector2ned(
            estimated_state.state_vector, reference_lla, gravity_model
        )
        measurement = np.zeros(6 if include_velocity else 3)
        measurement[:3] = trans.lla2ned(fix_position, reference_lla)
        if include_velocity:
            measurement[3:] = fix_velocity
        innovation = measurement - h_matrix @ state_ned

        state_errors = estimated_state.state_errors
        innovation_covariance = (
            measurement_covariance + h_matrix @ state_errors @ h_matrix.T
        )
        gain = state_errors @ h_matrix.T @ np.linalg.inv(innovation_covariance)
        state_ned += gain @ innovation

        # Joseph form preserves symmetry and positive semi-definiteness better
        # than the abbreviated covariance update.
        identity = np.eye(state_errors.shape[0])
        residual = identity - gain @ h_matrix
        state_errors = (
            residual @ state_errors @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        state_errors = (state_errors + state_errors.T) / 2

        state = ned2state_vector(state_ned, reference_lla, gravity_model)
        estimated_state.update_estimates(
            position=state[[0, 3, 6]],
            velocity=state[[1, 4, 7]],
            acceleration=state[[2, 5, 8]],
            attitude=state[[9, 11, 13]],
            angle_rates=state[[14, 12, 10]],
            state_errors=state_errors,
        )


# Short aliases are convenient for callers that already know they are in a
# recorded-input code path.
GnssFixedGainFusion = RecordedGnssFixedGainFusion
GnssLooseFusion = RecordedGnssLooseFusion


__all__ = [
    "GnssFixedGainFusion",
    "GnssLooseFusion",
    "RecordedGnssFixedGainFusion",
    "RecordedGnssLooseFusion",
    "RecordedGnssSensor",
]
