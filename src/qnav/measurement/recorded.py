"""Adapters that expose recorded measurements through the sensor interface.

The simulation sensors derive measurements from ground truth.  Replay has the
opposite requirement: a value and its timestamp already exist and need to be
presented to the existing fusion algorithms without adding simulated errors.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
import numpy.typing as npt

from qnav.measurement.platform import SensorAxis
from qnav.measurement.sensor import Sensor
from qnav.waypoints.trajectory import GroundTruth


class _RecordedSensor(Sensor):
    """Common clock and lifecycle behaviour for pushed measurements."""

    def __init__(
        self,
        frequency: float = 1.0,
        sensor_axis: SensorAxis | None = None,
        start_time: float = 0.0,
    ):
        super().__init__(frequency, start_time)
        self._sensor_axis = sensor_axis if sensor_axis is not None else SensorAxis()
        self._last_measurement = None
        self._has_timestamp = False

    def _push(self, timestamp: float, value: Any) -> Any:
        timestamp = float(timestamp)
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be a non-negative finite value")

        if self._has_timestamp:
            if timestamp < self._time_elapsed:
                raise ValueError("recorded sensor timestamps must be monotonic")
            interval = timestamp - self._time_elapsed
            if interval > 0:
                self._time_step = interval
                self._frequency = 1.0 / interval

        self._time_elapsed = timestamp
        self._next_update = timestamp + self._time_step
        self._last_measurement = value
        self._has_timestamp = True
        return self.last_measurement

    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        raise RuntimeError("recorded sensors receive measurements through push()")

    def update(self, time_sec: float = None):
        """Do nothing; :meth:`push` owns the replay clock."""

    def reset(self) -> None:
        """Mark the current value as consumed without rewinding the clock."""
        self._last_measurement = None

    @property
    def last_measurement(self) -> Any:
        value = self._last_measurement
        if isinstance(value, np.ndarray):
            return np.copy(value)
        return value

    @property
    def timestamp(self) -> float | None:
        """Timestamp of the latest pushed measurement, if one exists."""
        return self._time_elapsed if self._has_timestamp else None

    @property
    def sensor_axis(self) -> SensorAxis:
        return self._sensor_axis


class RecordedVectorSensor(_RecordedSensor):
    """Replay adapter for a finite three-axis measurement."""

    def push(self, timestamp: float, value: npt.ArrayLike) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("recorded vector measurement must have 3 finite values")
        return self._push(timestamp, np.copy(vector))


class RecordedScalarSensor(_RecordedSensor):
    """Replay adapter for a finite scalar measurement, such as altitude."""

    def push(self, timestamp: float, value: float) -> float:
        scalar = float(value)
        if not isfinite(scalar):
            raise ValueError("recorded scalar measurement must be finite")
        return self._push(timestamp, scalar)


class RecordedQuantumImu(_RecordedSensor):
    """Replay adapter for completed quantum acceleration/rate measurements."""

    def __init__(
        self,
        frequency: float = 1.0,
        full_frequency: float | None = None,
        sensor_axis: SensorAxis | None = None,
        start_time: float = 0.0,
    ):
        super().__init__(frequency, sensor_axis, start_time)
        if full_frequency is None:
            self._num_steps = 1
        else:
            if full_frequency <= 0 or full_frequency > frequency:
                raise ValueError(
                    "quantum frequency must be positive and no higher than "
                    "the conventional IMU frequency")
            self._num_steps = max(1, int(frequency / full_frequency))

    @staticmethod
    def _vector(value: npt.ArrayLike, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"{name} must have 3 finite values")
        return np.copy(vector)

    def push(
        self,
        timestamp: float,
        value: tuple[npt.ArrayLike, npt.ArrayLike] | npt.ArrayLike,
        angle_rates: npt.ArrayLike | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if angle_rates is None:
            try:
                acceleration, angle_rates = value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "quantum measurement must contain acceleration and angle rates"
                ) from exc
        else:
            acceleration = value

        measurement = (
            self._vector(acceleration, "acceleration"),
            self._vector(angle_rates, "angle_rates"),
        )
        return self._push(timestamp, measurement)

    @property
    def last_measurement(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._last_measurement is None:
            return None
        acceleration, angle_rates = self._last_measurement
        return np.copy(acceleration), np.copy(angle_rates)

    @property
    def num_active_steps(self) -> int:
        return self._num_steps

    @property
    def num_steps(self) -> int:
        return self._num_steps


class RecordedGravityGradiometer(_RecordedSensor):
    """Replay adapter for upper/lower gravity interferometer signals."""

    def push(
        self,
        timestamp: float,
        value: tuple[float, float] | float,
        bottom_signal: float | None = None,
    ) -> tuple[float, float]:
        if bottom_signal is None:
            try:
                top_signal, bottom_signal = value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "gradiometer measurement must contain top and bottom signals"
                ) from exc
        else:
            top_signal = value

        measurement = float(top_signal), float(bottom_signal)
        if not all(isfinite(signal) for signal in measurement):
            raise ValueError("gradiometer signals must be finite")
        return self._push(timestamp, measurement)

    @property
    def num_steps(self) -> int:
        return 1


class RecordedAccelerometer(RecordedVectorSensor):
    """Named vector replay adapter for accelerometer data."""


class RecordedGyroscope(RecordedVectorSensor):
    """Named vector replay adapter for gyroscope data."""


class RecordedAltimeter(RecordedScalarSensor):
    """Named scalar replay adapter for altimeter data."""


__all__ = [
    "RecordedAccelerometer",
    "RecordedAltimeter",
    "RecordedGravityGradiometer",
    "RecordedGyroscope",
    "RecordedQuantumImu",
    "RecordedScalarSensor",
    "RecordedVectorSensor",
]
