"""Run navigation algorithms against timestamped recorded measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Iterable

import json
import numpy as np

from qnav.estimation.state import EstimatedState, KalmanEstimatedState
from qnav.gps.recorded import (
    RecordedGnssFixedGainFusion,
    RecordedGnssLooseFusion,
    RecordedGnssSensor,
)
from qnav.gravity.map import set_default_grid
from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini import estimation_config as est_conf
from qnav.input.ini import gravity_config as grav_conf
from qnav.input.ini import vehicle_config as vehicle_conf
from qnav.input.raw import MeasurementEvent, RawDataset, load_raw_dataset
from qnav.measurement.recorded import RecordedAccelerometer, RecordedGyroscope
from qnav.output.data import ResultsTable
from qnav.util import transformations as trans
from qnav.waypoints.trajectory import GroundTruth


@dataclass
class ReplayResults:
    """Estimate records and diagnostics produced by a recorded-data run."""

    time_steps: np.ndarray
    states: np.ndarray
    utc_time: np.ndarray | None
    report: object

    def write(self, estimates: ResultsTable) -> None:
        fields = {
            "position": self.states[:, 1:4],
            "velocity": self.states[:, 4:7],
            "acceleration": self.states[:, 7:10],
            "attitude": self.states[:, 10:13],
            "angle_rates": self.states[:, 13:16],
        }
        if self.utc_time is not None:
            fields["utc_time"] = self.utc_time
        estimates.write("Estimated State", self.time_steps, **fields)

    def write_report(self, path: Path) -> None:
        report = self.report.as_dict() if hasattr(self.report, "as_dict") else dict(self.report)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")


class _Recorder:
    def __init__(self, duration: float, frequency: float, utc_origin: float | None):
        self._frequency = max(float(frequency), np.finfo(float).eps)
        size = max(2, int(np.ceil(max(duration, 0.0) * self._frequency)) + 2)
        self._times = np.full(size, np.nan)
        self._states = np.full((size, 16), np.nan)
        self._utc = np.full(size, np.nan) if utc_origin is not None else None
        self._utc_origin = utc_origin

    def record(self, timestamp: float, state: EstimatedState) -> None:
        index = min(int(np.floor(timestamp * self._frequency + 1e-10)), len(self._times) - 1)
        if index < 0:
            return
        if not np.isnan(self._times[index]):
            self._times[index] = timestamp
            self._states[index] = state.as_numpy()
            if self._utc is not None:
                self._utc[index] = self._utc_origin + timestamp
            return
        self._times[index] = timestamp
        self._states[index] = state.as_numpy()
        if self._utc is not None:
            self._utc[index] = self._utc_origin + timestamp

    def finish(self, report: object) -> ReplayResults:
        used = ~np.isnan(self._times)
        return ReplayResults(
            self._times[used], self._states[used],
            None if self._utc is None else self._utc[used], report)


def _initial_state(config: ConfigHandler, dataset: RawDataset, discovery):
    initial = dataset.config.initial_state
    if initial.attitude is None or any(value is None for value in initial.attitude):
        raise NavConfigError(
            "RawInitialState", "heading/pitch/roll", "MissingValue",
            "Raw replay requires configured initial heading, pitch and roll")

    configured_position = (
        initial.position is not None
        and all(value is not None for value in initial.position))
    if configured_position:
        start_event = discovery.earliest_event
        position = np.asarray(initial.position, dtype=float)
    else:
        start_event = discovery.first_gnss_event
        if start_event is None:
            raise NavConfigError(
                "RawInitialState", "latitude/longitude/altitude", "MissingValue",
                "Configure an initial position or provide an accepted GNSS fix")
        fix = start_event.payload
        position = np.array([fix.latitude, fix.longitude, fix.altitude], dtype=float)

    if start_event is None:
        raise ValueError("Raw dataset contains no usable measurements")

    velocity = None
    if initial.velocity is not None and all(value is not None for value in initial.velocity):
        velocity = np.asarray(initial.velocity, dtype=float)
    elif discovery.first_gnss_event is not None and discovery.first_gnss_event.payload.velocity is not None:
        velocity_ned = np.asarray(discovery.first_gnss_event.payload.velocity, dtype=float)
        attitude = np.radians(initial.attitude)
        velocity = trans.rotate_3d(*attitude) @ velocity_ned
    if velocity is None:
        velocity = np.zeros(3)

    acceleration = np.array([
        0.0 if value is None else value
        for value in (initial.acceleration or (0.0, 0.0, 0.0))])
    angle_rates = np.array([
        0.0 if value is None else value
        for value in (initial.angle_rates or (0.0, 0.0, 0.0))])
    attitude = np.asarray(initial.attitude, dtype=float)
    truth = GroundTruth(0.0, position, velocity, acceleration, attitude, angle_rates)

    gravity = grav_conf.get_estimated_gravity_model(config)
    set_default_grid(**grav_conf.get_gradient_grid(config))
    state = est_conf.get_estimated_state(config, truth, gravity)
    est_conf.add_estimation_errors(config, state)
    return state, start_event


def _build_fusions(config: ConfigHandler, kinds: set[str], state: EstimatedState):
    sensors = {}
    fusions = {}
    axis = vehicle_conf.get_sensor_axis(config)

    if "accelerometer" in kinds:
        sensors["accelerometer"] = RecordedAccelerometer(sensor_axis=axis)
    if "gyroscope" in kinds:
        sensors["gyroscope"] = RecordedGyroscope(sensor_axis=axis)
    if ("accelerometer" in sensors) != ("gyroscope" in sensors):
        raise NavConfigError(
            "RawData", "IMU", "Missing stream",
            "Raw INS replay requires both accelerometer and gyroscope streams")
    if "accelerometer" in sensors:
        fusions["ins"] = est_conf.get_ins(
            config, sensors["accelerometer"], sensors["gyroscope"])

    if "gnss" in kinds:
        sensor = RecordedGnssSensor()
        sensors["gnss"] = sensor
        method = config.get_str_alpha("GPS", "gpsFusionMethod", "fixedgain")
        if method == "fixedgain":
            gain = config.get_float("GPS", "gpsFixedGain", 0.05)
            fusions["gnss"] = RecordedGnssFixedGainFusion(sensor, state, gain)
        elif method == "loose":
            if not isinstance(state, KalmanEstimatedState):
                raise NavConfigError(
                    "GPS", "gpsFusionMethod", "Incompatible state",
                    "Loose GNSS replay requires estimatedStateModel=kalman")
            fusions["gnss"] = RecordedGnssLooseFusion(
                sensor, state,
                config.get_float("GPS", "gpsSigmaPosition", 50.0),
                config.get_float("GPS", "gpsSigmaVelocity", 0.3))
        else:
            raise NavConfigError(
                "GPS", "gpsFusionMethod", "Unsupported raw mode",
                "Receiver-fix replay supports fixedgain or loose fusion")
    return sensors, fusions


def _event_groups(events: Iterable[MeasurementEvent]):
    for timestamp, group in groupby(events, key=lambda event: event.timestamp):
        yield timestamp, list(group)


def run_raw_replay(config: ConfigHandler) -> ReplayResults:
    """Load configured recorded data and run the selected navigation methods."""
    dataset = load_raw_dataset(config)
    discovery = dataset.discover()
    if discovery.earliest_event is None or discovery.latest_event is None:
        raise ValueError("Raw dataset contains no accepted measurements")

    state, start_event = _initial_state(config, dataset, discovery)
    origin = start_event.timestamp
    utc_origin = start_event.absolute_time.timestamp() if start_event.absolute_time else None
    duration = max(0.0, discovery.latest_event.timestamp - origin)
    output_frequency = config.get_float("Measurement", "imuMeasurementFreq", 100.0)
    output_frequency /= max(1, config.get_int("Output", "resultsDownSampleRate", 1))
    recorder = _Recorder(duration, output_frequency, utc_origin)
    recorder.record(0.0, state)

    kinds = {spec.kind for spec in dataset.specs}
    sensors, fusions = _build_fusions(config, kinds, state)
    last_ins_time = 0.0
    discarded = 0

    for source_time, events in _event_groups(dataset.iter_events()):
        if source_time < origin:
            discarded += len(events)
            continue
        elapsed = source_time - origin
        imu_changed = False
        did_fuse = False

        for event in events:
            if event.kind in ("accelerometer", "gyroscope"):
                sensors[event.kind].push(elapsed, event.payload)
                imu_changed = True
            elif event.kind == "gnss":
                sensors["gnss"].push(elapsed, event.payload)

        if imu_changed and "ins" in fusions:
            acc_ready = sensors["accelerometer"].last_measurement is not None
            gyro_ready = sensors["gyroscope"].last_measurement is not None
            dt = elapsed - last_ins_time
            if acc_ready and gyro_ready and dt > 0:
                fusions["ins"].perform_fusion(state, time_step=dt)
                last_ins_time = elapsed
                did_fuse = True

        if any(event.kind == "gnss" for event in events) and "gnss" in fusions:
            fusions["gnss"].perform_fusion(state)
            did_fuse = True

        if did_fuse:
            state.update_estimates(timestamp=elapsed)
            recorder.record(elapsed, state)

    dataset.report.discarded_pre_initialization = discarded
    print(
        f"Raw input: {dataset.report.accepted} events accepted, "
        f"{dataset.report.rejected} rejected, {discarded} before initialization")
    return recorder.finish(dataset.report)


__all__ = ["ReplayResults", "run_raw_replay"]
