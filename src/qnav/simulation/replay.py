"""Run navigation algorithms against timestamped recorded measurements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Iterable

import csv
import json
import numpy as np
import shutil
import tempfile

from qnav.estimation.state import EstimatedState, KalmanEstimatedState
from qnav.gps.recorded import (
    RecordedGnssFixedGainFusion,
    RecordedGnssLooseFusion,
    RecordedGnssSensor,
)
from qnav.gravity.map import set_default_grid
from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini import altimeter_config as altimeter_conf
from qnav.input.ini import estimation_config as est_conf
from qnav.input.ini import gravity_config as grav_conf
from qnav.input.ini import measurement_config as measurement_conf
from qnav.input.ini import quantum_config as quantum_conf
from qnav.input.ini import quantum_grav_config as quantum_grav_conf
from qnav.input.ini import vehicle_config as vehicle_conf
from qnav.input.raw import MeasurementEvent, RawDataset, load_raw_dataset
from qnav.measurement.recorded import (
    RecordedAccelerometer,
    RecordedAltimeter,
    RecordedGravityGradiometer,
    RecordedGyroscope,
    RecordedQuantumImu,
)
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
    processed_imu: "_ProcessedImuCapture | None" = None
    reference: "dict[str, np.ndarray] | None" = None

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

    def write_reference(self, reference: ResultsTable) -> bool:
        """Write aligned reference fields, returning whether any exist."""
        if not self.reference:
            return False
        reference.write("Ground Truth", self.time_steps, **self.reference)
        return True

    def write_processed_imu(self, output_dir: Path) -> None:
        if self.processed_imu is not None:
            self.processed_imu.copy_to(output_dir)


class _ProcessedImuCapture:
    """Stream generated IMU measurements to temporary CSV files."""

    def __init__(self):
        self._directory = tempfile.TemporaryDirectory(prefix="qnav_processed_imu_")
        base = Path(self._directory.name)
        self._paths = {
            "accelerometer": base / "processed_accelerometer.csv",
            "gyroscope": base / "processed_gyroscope.csv",
        }
        self._handles = {}
        self._writers = {}
        for kind, path in self._paths.items():
            handle = path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(("time", "utc_time", "x", "y", "z"))
            self._handles[kind] = handle
            self._writers[kind] = writer

    def record(self, kind: str, timestamp: float,
               utc_time: float | None, value) -> None:
        self._writers[kind].writerow((
            timestamp, "" if utc_time is None else utc_time,
            *np.asarray(value, dtype=float)))

    def close(self) -> None:
        for handle in self._handles.values():
            if not handle.closed:
                handle.close()

    def copy_to(self, output_dir: Path) -> None:
        self.close()
        for path in self._paths.values():
            shutil.copyfile(path, output_dir / path.name)

    def cleanup(self) -> None:
        self.close()
        self._directory.cleanup()

    def __del__(self):
        self.cleanup()


class _Recorder:
    def __init__(self, duration: float, frequency: float, utc_origin: float | None):
        self._frequency = max(float(frequency), np.finfo(float).eps)
        size = max(2, int(np.ceil(max(duration, 0.0) * self._frequency)) + 2)
        self._times = np.full(size, np.nan)
        self._states = np.full((size, 16), np.nan)
        self._utc = np.full(size, np.nan) if utc_origin is not None else None
        self._utc_origin = utc_origin

    def record(self, timestamp: float, state: EstimatedState) -> int | None:
        index = min(int(np.floor(timestamp * self._frequency + 1e-10)), len(self._times) - 1)
        if index < 0:
            return None
        if not np.isnan(self._times[index]):
            return None
        self._times[index] = timestamp
        self._states[index] = state.as_numpy()
        if self._utc is not None:
            self._utc[index] = self._utc_origin + timestamp
        return index

    def finish(self, report: object) -> ReplayResults:
        used = ~np.isnan(self._times)
        return ReplayResults(
            self._times[used], self._states[used],
            None if self._utc is None else self._utc[used], report)


_REFERENCE_FIELDS = {
    "position": ("latitude", "longitude", "altitude"),
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
    "acceleration": ("acceleration_x", "acceleration_y", "acceleration_z"),
    "attitude": ("heading", "pitch", "roll"),
    "angle_rates": ("angular_rate_x", "angular_rate_y", "angular_rate_z"),
}


def _reference_record(payload: dict) -> dict[str, np.ndarray]:
    """Group one canonical flat reference payload for interpolation."""
    record = {}
    for field, names in _REFERENCE_FIELDS.items():
        if all(name in payload for name in names):
            record[field] = np.asarray([payload[name] for name in names], dtype=float)
    quaternion_names = (
        "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z")
    if all(name in payload for name in quaternion_names):
        record["quaternion"] = np.asarray(
            [payload[name] for name in quaternion_names], dtype=float)
    elif "attitude" in record:
        record["quaternion"] = trans.euler_to_quaternion(record["attitude"])
    return record


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path interpolation of scalar-first unit quaternions."""
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        return result / np.linalg.norm(result)
    angle = np.arccos(dot)
    return (
        np.sin((1.0 - fraction) * angle) * first
        + np.sin(fraction * angle) * second
    ) / np.sin(angle)


def _interpolate_reference(first: dict[str, np.ndarray],
                           second: dict[str, np.ndarray],
                           fraction: float) -> dict[str, np.ndarray]:
    result = {}
    for field in ("velocity", "acceleration", "angle_rates"):
        if field in first and field in second:
            result[field] = first[field] + fraction * (second[field] - first[field])
    if "position" in first and "position" in second:
        xyz_first = trans.lla2ecef(first["position"])
        xyz_second = trans.lla2ecef(second["position"])
        result["position"] = trans.ecef2lla(
            xyz_first + fraction * (xyz_second - xyz_first))
    if "quaternion" in first and "quaternion" in second:
        quaternion = _slerp(first["quaternion"], second["quaternion"], fraction)
        result["attitude"] = trans.quaternion_to_euler(quaternion)
    elif "attitude" in first and "attitude" in second:
        # This branch is retained for callers that construct grouped records
        # directly; raw reference payloads always acquire a quaternion above.
        delta = (second["attitude"] - first["attitude"] + 180.0) % 360.0 - 180.0
        result["attitude"] = first["attitude"] + fraction * delta
    return result


class _ReferenceAligner:
    """Align a streaming reference onto the retained estimate time grid."""

    def __init__(self):
        self._previous = {field: None for field in _REFERENCE_FIELDS}
        self._current = {field: None for field in _REFERENCE_FIELDS}
        self._rows: list[dict[str, np.ndarray]] = []
        self._target_times: list[float] = []
        self._seen_fields = set()
        self._pending = {field: [] for field in _REFERENCE_FIELDS}

    @staticmethod
    def _same_time(first: float, second: float) -> bool:
        return bool(np.isclose(first, second, rtol=0.0, atol=1e-12))

    def add_target(self, timestamp: float) -> None:
        index = len(self._rows)
        self._rows.append({})
        self._target_times.append(timestamp)
        for field in self._seen_fields:
            if not self._resolve(field, index, timestamp):
                self._pending[field].append((index, timestamp))

    def push(self, timestamp: float, payload: dict) -> None:
        record = _reference_record(payload)
        for field in _REFERENCE_FIELDS:
            if field not in record:
                continue
            first_sample = field not in self._seen_fields
            self._seen_fields.add(field)
            if self._current[field] is not None:
                self._previous[field] = self._current[field]
            self._current[field] = (timestamp, record)

            pending = (
                list(enumerate(self._target_times))
                if first_sample else self._pending[field]
            )
            remaining = []
            for index, target in pending:
                if not self._resolve(field, index, target) and target > timestamp:
                    remaining.append((index, target))
                # Older unbracketed targets remain empty: reference data is
                # never extrapolated.
            self._pending[field] = remaining

    def _resolve(self, field: str, index: int, timestamp: float) -> bool:
        current = self._current[field]
        previous = self._previous[field]
        if current is not None and self._same_time(timestamp, current[0]):
            self._rows[index][field] = current[1][field].copy()
            return True
        if (previous is not None and current is not None
                and previous[0] <= timestamp <= current[0]):
            self._rows[index][field] = self._between(
                field, timestamp, previous, current)
            return True
        return False

    def _between(self, field: str, timestamp: float, previous, current) \
            -> np.ndarray:
        first_time, first = previous
        second_time, second = current
        if self._same_time(first_time, second_time):
            return second[field].copy()
        fraction = (timestamp - first_time) / (second_time - first_time)
        return _interpolate_reference(first, second, fraction)[field]

    def finish(self) -> dict[str, np.ndarray] | None:
        fields = set()
        for row in self._rows:
            fields.update(row)
        fields.discard("quaternion")
        if not fields:
            return None
        return {
            field: np.vstack([
                row.get(field, np.full(3, np.nan)) for row in self._rows
            ])
            for field in sorted(fields)
        }


def _initial_state(config: ConfigHandler, dataset: RawDataset, discovery):
    initial = dataset.config.initial_state
    reference_event = discovery.first_reference_event
    reference = (
        _reference_record(reference_event.payload)
        if reference_event is not None else {})

    def configured_group(name: str, values):
        if values is None:
            return None
        present = [value is not None for value in values]
        if any(present) and not all(present):
            raise NavConfigError(
                "RawInitialState", name, "PartialValue",
                f"Raw initial {name} must configure all three components")
        return np.asarray(values, dtype=float) if all(present) else None

    configured_attitude = configured_group(
        "heading/pitch/roll", initial.attitude)
    used_reference = False
    if configured_attitude is not None:
        attitude = configured_attitude
    elif "attitude" in reference:
        attitude = reference["attitude"].copy()
        used_reference = True
    else:
        raise NavConfigError(
            "RawInitialState", "heading/pitch/roll", "MissingValue",
            "Raw replay requires explicit or reference heading, pitch and roll")

    configured_position = configured_group(
        "latitude/longitude/altitude", initial.position)
    used_gnss = False
    if configured_position is not None:
        position = configured_position
    elif "position" in reference:
        position = reference["position"].copy()
        used_reference = True
    else:
        if discovery.first_gnss_event is None:
            raise NavConfigError(
                "RawInitialState", "latitude/longitude/altitude", "MissingValue",
                "Configure an initial position, reference, or accepted GNSS fix")
        fix = discovery.first_gnss_event.payload
        position = np.array([fix.latitude, fix.longitude, fix.altitude], dtype=float)
        used_gnss = True

    velocity = configured_group("velocity", initial.velocity)
    if velocity is None:
        if "velocity" in reference:
            velocity = reference["velocity"].copy()
            used_reference = True
        elif (discovery.first_gnss_event is not None
              and discovery.first_gnss_event.payload.velocity is not None):
            velocity_ned = np.asarray(
                discovery.first_gnss_event.payload.velocity, dtype=float)
            velocity = trans.rotate_3d(*np.radians(attitude)) @ velocity_ned
            used_gnss = True
    if velocity is None:
        velocity = np.zeros(3)

    acceleration = configured_group("acceleration", initial.acceleration)
    if acceleration is None:
        if "acceleration" in reference:
            acceleration = reference["acceleration"].copy()
            used_reference = True
        else:
            acceleration = np.zeros(3)
    angle_rates = configured_group("angular rates", initial.angle_rates)
    if angle_rates is None:
        if "angle_rates" in reference:
            angle_rates = reference["angle_rates"].copy()
            used_reference = True
        else:
            angle_rates = np.zeros(3)

    source_events = []
    if used_reference:
        source_events.append(reference_event)
    if used_gnss:
        source_events.append(discovery.first_gnss_event)
    start_event = (
        max(source_events, key=lambda event: event.timestamp)
        if source_events else discovery.earliest_event)
    if start_event is None:
        raise ValueError("Raw dataset contains no usable measurements")

    truth = GroundTruth(0.0, position, velocity, acceleration, attitude, angle_rates)

    gravity = grav_conf.get_estimated_gravity_model(config)
    set_default_grid(**grav_conf.get_gradient_grid(config))
    state = est_conf.get_estimated_state(config, truth, gravity)
    est_conf.add_estimation_errors(config, state)
    return state, start_event


def _build_fusions(config: ConfigHandler, kinds: set[str], state: EstimatedState,
                   imu_input_level: str):
    sensors = {}
    fusions = {}
    axis = vehicle_conf.get_sensor_axis(config)
    imu_frequency = measurement_conf.get_imu_frequency(config)

    if "accelerometer" in kinds:
        sensors["accelerometer"] = (
            measurement_conf.get_accelerometer(config)
            if imu_input_level == "truth"
            else RecordedAccelerometer(
                frequency=imu_frequency, sensor_axis=axis))
    if "gyroscope" in kinds:
        sensors["gyroscope"] = (
            measurement_conf.get_gyroscope(config)
            if imu_input_level == "truth"
            else RecordedGyroscope(
                frequency=imu_frequency, sensor_axis=axis))
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

    if "altimeter" in kinds:
        sensor = RecordedAltimeter()
        sensors["altimeter"] = sensor
        fusions["altimeter"] = altimeter_conf.get_fusion(config, sensor)

    if "quantum_imu" in kinds:
        if "accelerometer" not in sensors:
            raise NavConfigError(
                "RawQuantumImu", "IMU", "Missing stream",
                "Raw quantum IMU fusion requires conventional accelerometer "
                "and gyroscope streams")
        sensor = RecordedQuantumImu(
            frequency=imu_frequency,
            full_frequency=config.get_float(
                "Quantum", "quantumImuFrequency", 1.0),
            sensor_axis=axis,
        )
        sensors["quantum_imu"] = sensor
        fusions["quantum_imu"] = quantum_conf.get_quantum_imu_fusion(
            config, sensor, sensors["accelerometer"], sensors["gyroscope"])

    if "gradiometer" in kinds:
        sensor = RecordedGravityGradiometer(sensor_axis=axis)
        sensors["gradiometer"] = sensor
        fusions["gradiometer"] = quantum_grav_conf.get_quantum_grav_fusion(
            config, sensor)
    return sensors, fusions


def _event_groups(events: Iterable[MeasurementEvent]):
    for timestamp, group in groupby(events, key=lambda event: event.timestamp):
        yield timestamp, list(group)


def _normalize_imu_frame(event: MeasurementEvent, sensor,
                         input_level: str) -> np.ndarray:
    """Convert one raw IMU vector to the frame expected by its sensor path."""
    value = np.asarray(event.payload, dtype=float)
    default_frame = "body" if input_level == "truth" else "sensor"
    frame = (event.frame or default_frame).strip().lower()
    if frame not in {"body", "sensor"}:
        raise ValueError(
            f"raw {event.kind} frame must be 'body' or 'sensor', got '{frame}'")

    body_to_sensor = sensor.sensor_axis.body2sensor_mat
    if input_level == "truth":
        # Simulated sensor models consume ideal body-frame truth and apply the
        # configured mounting themselves.
        return (np.linalg.solve(body_to_sensor, value)
                if frame == "sensor" else value)

    # Recorded adapters feed fusion directly, which expects sensor-frame
    # measurements and removes the configured mounting there.
    return body_to_sensor @ value if frame == "body" else value


def _normalize_quantum_imu_frame(event: MeasurementEvent, sensor) \
        -> tuple[np.ndarray, np.ndarray]:
    """Convert a recorded quantum IMU pair to its configured sensor frame."""
    acceleration = np.asarray(event.payload["acceleration"], dtype=float)
    angle_rates = np.asarray(event.payload["angular_rate"], dtype=float)
    frame = (event.frame or "sensor").strip().lower()
    if frame not in {"body", "sensor"}:
        raise ValueError(
            f"raw quantum_imu frame must be 'body' or 'sensor', got '{frame}'")
    if frame == "body":
        body_to_sensor = sensor.sensor_axis.body2sensor_mat
        acceleration = body_to_sensor @ acceleration
        angle_rates = body_to_sensor @ angle_rates
    return acceleration, angle_rates


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

    kinds = {spec.kind for spec in dataset.specs}
    reference_aligner = _ReferenceAligner() if "reference" in kinds else None
    if recorder.record(0.0, state) is not None and reference_aligner is not None:
        reference_aligner.add_target(0.0)
    imu_input_level = dataset.config.imu_input_level
    sensors, fusions = _build_fusions(config, kinds, state, imu_input_level)
    save_processed_imu = config.get_bool("Output", "saveProcessedImu", False)
    if save_processed_imu and imu_input_level != "truth":
        raise NavConfigError(
            "Output", "saveProcessedImu", "Incompatible input",
            "Processed IMU output is only available for imuInputLevel=truth")
    processed_imu = _ProcessedImuCapture() if save_processed_imu else None
    ideal_imu = {"accelerometer": None, "gyroscope": None}
    last_ins_time = None
    quantum_imu_ready = {"accelerometer": False, "gyroscope": False}
    discarded = 0

    for source_time, events in _event_groups(dataset.iter_events()):
        if source_time < origin:
            if reference_aligner is not None:
                for event in events:
                    if event.kind == "reference":
                        reference_aligner.push(
                            source_time - origin, event.payload)
            discarded += len(events)
            continue
        elapsed = source_time - origin
        imu_event_kinds = set()
        imu_updated_kinds = set()
        did_fuse = False

        for event in events:
            if event.kind in ("accelerometer", "gyroscope"):
                normalized = _normalize_imu_frame(
                    event, sensors[event.kind], imu_input_level)
                if imu_input_level == "truth":
                    ideal_imu[event.kind] = normalized
                else:
                    sensors[event.kind].push(elapsed, normalized)
                imu_event_kinds.add(event.kind)
            elif event.kind == "gnss":
                sensors["gnss"].push(elapsed, event.payload)
            elif event.kind == "altimeter":
                sensors["altimeter"].push(elapsed, event.payload)
            elif event.kind == "quantum_imu":
                acceleration, angle_rates = _normalize_quantum_imu_frame(
                    event, sensors["quantum_imu"])
                sensors["quantum_imu"].push(
                    elapsed, acceleration, angle_rates)
            elif event.kind == "gradiometer":
                sensors["gradiometer"].push(elapsed, event.payload)
            elif event.kind == "reference" and reference_aligner is not None:
                reference_aligner.push(elapsed, event.payload)

        if (imu_input_level == "truth" and imu_event_kinds
                and all(value is not None for value in ideal_imu.values())):
            truth = GroundTruth(
                elapsed, state.position, state.velocity,
                ideal_imu["accelerometer"], state.attitude,
                ideal_imu["gyroscope"])
            utc_time = next((event.absolute_time.timestamp() for event in events
                             if event.absolute_time is not None), None)
            for kind in ("accelerometer", "gyroscope"):
                if kind in imu_event_kinds or sensors[kind].last_measurement is None:
                    sensor = sensors[kind]
                    sensor.take_measurement(elapsed, truth)
                    imu_updated_kinds.add(kind)
                    if processed_imu is not None:
                        processed_imu.record(
                            kind, elapsed, utc_time, sensor.last_measurement)
                    sensor.update(elapsed)
        elif imu_input_level != "truth":
            imu_updated_kinds.update(imu_event_kinds)

        if imu_updated_kinds and "ins" in fusions:
            acc_ready = sensors["accelerometer"].last_measurement is not None
            gyro_ready = sensors["gyroscope"].last_measurement is not None
            if acc_ready and gyro_ready:
                if last_ins_time is None:
                    # The first complete pair establishes the IMU clock. It
                    # cannot describe motion over any earlier initialization
                    # or sensor-start gap.
                    last_ins_time = elapsed
                else:
                    dt = elapsed - last_ins_time
                    if dt > 0:
                        fusions["ins"].perform_fusion(state, time_step=dt)
                        last_ins_time = elapsed
                        did_fuse = True

        if any(event.kind == "gnss" for event in events) and "gnss" in fusions:
            fusions["gnss"].perform_fusion(state)
            did_fuse = True

        if (any(event.kind == "altimeter" for event in events)
                and "altimeter" in fusions):
            fusions["altimeter"].perform_fusion(state)
            did_fuse = True

        for kind in imu_updated_kinds:
            quantum_imu_ready[kind] = True
        quantum_event = any(event.kind == "quantum_imu" for event in events)
        if "quantum_imu" in fusions and quantum_event:
            quantum_applied = fusions[
                "quantum_imu"].apply_pending_measurement(state)
            did_fuse = bool(quantum_applied) or did_fuse
        if ("quantum_imu" in fusions
                and all(quantum_imu_ready.values())
                and sensors["accelerometer"].last_measurement is not None
                and sensors["gyroscope"].last_measurement is not None):
            quantum_applied = fusions["quantum_imu"].perform_fusion(state)
            quantum_imu_ready = {"accelerometer": False, "gyroscope": False}
            did_fuse = bool(quantum_applied) or did_fuse

        if (any(event.kind == "gradiometer" for event in events)
                and "gradiometer" in fusions):
            fusions["gradiometer"].perform_fusion(state)
            did_fuse = True

        if did_fuse:
            state.update_estimates(timestamp=elapsed)
            recorded = recorder.record(elapsed, state)
            if recorded is not None and reference_aligner is not None:
                reference_aligner.add_target(elapsed)

    dataset.report.discarded_pre_initialization = discarded
    print(
        f"Raw input: {dataset.report.accepted} events accepted, "
        f"{dataset.report.rejected} rejected, {discarded} before initialization")
    results = recorder.finish(dataset.report)
    results.processed_imu = processed_imu
    if reference_aligner is not None:
        results.reference = reference_aligner.finish()
    return results


__all__ = ["ReplayResults", "run_raw_replay"]
