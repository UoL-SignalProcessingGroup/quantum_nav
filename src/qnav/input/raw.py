"""Configuration-driven, bounded-memory loading of recorded sensor data.

The loader deliberately stops at a small, sensor-neutral event interface.  It
does not construct simulated sensors or apply sensor error models: values read
from disk are the measurements to be replayed.
"""

from __future__ import annotations

from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import math
import numpy as np
import pandas as pd

from qnav.input.config_handler import ConfigHandler
from qnav.util.transformations import (
    quaternion_to_euler,
    rotate_3d,
    vel_ecef2vel_ned,
)


_G = 9.80665
_MAX_REPORTED_ERRORS = 100
_SENSOR_SECTIONS = {
    "RawAccelerometer": "accelerometer",
    "RawGyroscope": "gyroscope",
    "RawGnss": "gnss",
    "RawAltimeter": "altimeter",
    "RawQuantumImu": "quantum_imu",
    "RawGravityGradiometer": "gradiometer",
    "RawReference": "reference",
}


@dataclass(frozen=True)
class GnssFix:
    """A normalized receiver fix (degrees, metres, and metres/second)."""

    latitude: float
    longitude: float
    altitude: float
    velocity: tuple[float, float, float] | None = None
    velocity_frame: str = "ned"
    valid: bool | None = None
    fix_type: float | None = None
    satellite_count: int | None = None
    horizontal_accuracy: float | None = None
    vertical_accuracy: float | None = None
    speed_accuracy: float | None = None
    accepted: bool = True


@dataclass(frozen=True)
class MeasurementEvent:
    """One recorded measurement on a time-ordered replay stream.

    ``timestamp`` is the source time converted to seconds.  Datetime sources
    use Unix epoch seconds and additionally populate ``absolute_time``.
    ``payload`` is a three-tuple for conventional IMU streams, a
    :class:`GnssFix` for GNSS, a float for altimeters, a dictionary for quantum
    IMU/reference values, and a two-tuple for gradiometers.
    """

    timestamp: float | Fraction
    kind: str
    payload: Any
    absolute_time: datetime | None = None
    source: Path | None = None
    row_number: int | None = None
    frame: str | None = None

    @property
    def data(self) -> Any:
        """Alias retained for event consumers that call the payload data."""
        return self.payload


@dataclass
class IngestionReport:
    """Counters produced by one complete pass over a raw dataset."""

    rows_read: int = 0
    accepted: int = 0
    rejected: int = 0
    quality_filtered: int = 0
    duplicate_timestamps: int = 0
    backward_timestamps: int = 0
    partial_records: int = 0
    source_files: tuple[Path, ...] = ()
    earliest_timestamp: float | Fraction | None = None
    latest_timestamp: float | Fraction | None = None
    errors: list[str] = field(default_factory=list)
    errors_omitted: int = 0

    @property
    def accepted_events(self) -> int:
        return self.accepted

    @property
    def rejected_records(self) -> int:
        return self.rejected

    def as_dict(self) -> dict[str, Any]:
        result = vars(self).copy()
        result["source_files"] = [str(path) for path in self.source_files]
        for name in ("earliest_timestamp", "latest_timestamp"):
            if isinstance(result[name], Fraction):
                result[name] = float(result[name])
        return result


@dataclass(frozen=True)
class CsvOptions:
    delimiter: str = ","
    quotechar: str = '"'
    comment: str | None = None
    encoding: str = "utf-8"
    decimal: str = "."


@dataclass(frozen=True)
class SensorSpec:
    section: str
    kind: str
    file: Path
    columns: Mapping[str, str]
    required: tuple[str, ...]
    options: CsvOptions
    time_unit: str = "auto"
    datetime_format: str | None = None
    timezone: str = "UTC"
    units: Mapping[str, str] = field(default_factory=dict)
    frame: str | None = None
    signs: tuple[float, float, float] = (1.0, 1.0, 1.0)
    quality: Mapping[str, float] = field(default_factory=dict)
    quaternion_direction: str | None = None


@dataclass(frozen=True)
class RawInputConfig:
    """Resolved raw-input settings, suitable for inspection and replay."""

    specs: tuple[SensorSpec, ...]
    chunk_size: int = 100_000
    invalid_record_policy: str = "strict"
    initial_state: "RawInitialState" = field(default_factory=lambda: RawInitialState())
    main_config: Path | None = None
    raw_config: Path | None = None
    mapping_config: ConfigParser | None = field(default=None, repr=False, compare=False)
    declaring_base: Path | None = None
    imu_input_level: str = "measurement"

    def spec_for(self, kind: str) -> SensorSpec | None:
        """Return the enabled specification for a sensor kind, if present."""
        return next((spec for spec in self.specs if spec.kind == kind), None)

    def is_enabled(self, kind: str) -> bool:
        return self.spec_for(kind) is not None


@dataclass(frozen=True)
class RawInitialState:
    """Optional navigation state supplied for recorded replay."""

    position: tuple[float | None, float | None, float | None] | None = None
    velocity: tuple[float | None, float | None, float | None] | None = None
    acceleration: tuple[float | None, float | None, float | None] | None = None
    attitude: tuple[float | None, float | None, float | None] | None = None
    angle_rates: tuple[float | None, float | None, float | None] | None = None

    def as_dict(self) -> dict[str, tuple[float | None, float | None, float | None]]:
        return {name: value for name in ("position", "velocity", "acceleration",
                                          "attitude", "angle_rates")
                if (value := getattr(self, name)) is not None}


@dataclass(frozen=True)
class RawDiscovery:
    earliest_event: MeasurementEvent | None
    latest_event: MeasurementEvent | None
    first_gnss_event: MeasurementEvent | None
    report: IngestionReport
    first_reference_event: MeasurementEvent | None = None

    @property
    def first_gnss_fix(self) -> GnssFix | None:
        if self.first_gnss_event is None:
            return None
        return self.first_gnss_event.payload


class RawDataset:
    """A repeatable dataset whose event passes never retain the input file."""

    def __init__(self, config: RawInputConfig):
        self.config = config
        self.report = IngestionReport(
            source_files=tuple(dict.fromkeys(spec.file for spec in config.specs)))

    @property
    def specs(self) -> tuple[SensorSpec, ...]:
        return self.config.specs

    @property
    def mapping_config(self) -> ConfigParser | None:
        return self.config.mapping_config

    @property
    def declaring_base(self) -> Path | None:
        return self.config.declaring_base

    def __iter__(self) -> Iterator[MeasurementEvent]:
        return self.iter_events()

    def iter_events(self, update_report: bool = True) -> Iterator[MeasurementEvent]:
        """Return a new, bounded-memory, globally ordered event iterator."""
        report = IngestionReport(
            source_files=tuple(dict.fromkeys(spec.file for spec in self.specs)))

        def generate() -> Iterator[MeasurementEvent]:
            groups: dict[tuple[Path, CsvOptions], list[SensorSpec]] = {}
            for spec in self.specs:
                groups.setdefault((spec.file, spec.options), []).append(spec)
            streams = [iter(_iter_file(path, specs, self.config, report))
                       for (path, _), specs in groups.items()]
            heap: list[tuple[float, int, MeasurementEvent, Iterator[MeasurementEvent]]] = []
            serial = 0
            for stream in streams:
                try:
                    event = next(stream)
                except StopIteration:
                    continue
                heappush(heap, (event.timestamp, serial, event, stream))
                serial += 1
            while heap:
                _, _, event, stream = heappop(heap)
                report.accepted += 1
                report.earliest_timestamp = (
                    event.timestamp if report.earliest_timestamp is None
                    else min(report.earliest_timestamp, event.timestamp))
                report.latest_timestamp = (
                    event.timestamp if report.latest_timestamp is None
                    else max(report.latest_timestamp, event.timestamp))
                yield event
                try:
                    following = next(stream)
                except StopIteration:
                    continue
                heappush(heap, (following.timestamp, serial, following, stream))
                serial += 1

        iterator = generate()
        try:
            yield from iterator
        finally:
            if update_report:
                self.report = report

    def discover(self) -> RawDiscovery:
        """Scan timing and initialization facts without materializing events."""
        earliest = latest = first_gnss = first_reference = None
        time_domain = None
        # Keep the discovery diagnostics available to its caller without
        # overwriting the report belonging to the later replay pass.
        previous_report = self.report
        for event in self.iter_events(update_report=True):
            event_domain = "absolute" if event.absolute_time is not None else "relative"
            if time_domain is None:
                time_domain = event_domain
            elif event_domain != time_domain:
                raise ValueError(
                    "Raw input mixes relative numeric and absolute timestamps; "
                    "use a consistent time domain and a unix_* timeUnit for "
                    "numeric Unix timestamps")
            if earliest is None:
                earliest = event
            latest = event
            if first_gnss is None and event.kind == "gnss":
                fix = event.payload
                if isinstance(fix, GnssFix) and fix.accepted:
                    first_gnss = event
            if first_reference is None and event.kind == "reference":
                first_reference = event
        discovery_report = self.report
        self.report = previous_report
        return RawDiscovery(
            earliest, latest, first_gnss, discovery_report, first_reference)

    # A name useful to callers performing an explicit prepass.
    summary = discover


class _ConfigView:
    def __init__(self, main: ConfigParser, main_path: Path,
                 raw: ConfigParser | None = None, raw_path: Path | None = None):
        self.main = main
        self.main_path = main_path
        self.raw = raw
        self.raw_path = raw_path

    def parser_for(self, section: str) -> tuple[ConfigParser, Path]:
        if self.raw is not None and self.raw.has_section(section):
            return self.raw, self.raw_path
        return self.main, self.main_path

    def _parsers_for(self, section: str):
        """Yield option sources in precedence order.

        A reusable raw mapping overrides options it declares, while the main
        run configuration can still supply options absent from that mapping.
        """
        if self.raw is not None and self.raw.has_section(section):
            yield self.raw, self.raw_path
        if self.main.has_section(section):
            yield self.main, self.main_path

    def has_section(self, section: str) -> bool:
        return any(True for _ in self._parsers_for(section))

    def get(self, section: str, names: str | Iterable[str],
            fallback: Any = None) -> Any:
        if isinstance(names, str):
            names = (names,)
        for parser, _ in self._parsers_for(section):
            for name in names:
                if parser.has_option(section, name):
                    value = parser.get(section, name).strip()
                    if value != "":
                        return value
        return fallback

    def get_with_base(self, section: str, names: str | Iterable[str],
                      fallback: Any = None) -> tuple[Any, Path | None]:
        """Return an option together with the file that declared it."""
        if isinstance(names, str):
            names = (names,)
        for parser, path in self._parsers_for(section):
            for name in names:
                if parser.has_option(section, name):
                    value = parser.get(section, name).strip()
                    if value != "":
                        return value, path
        return fallback, None

    def base(self, section: str) -> Path:
        return self.parser_for(section)[1].parent


def is_raw_mode(config: ConfigHandler | ConfigParser | Path | str) -> bool:
    """Return whether ``[Input] inputMode`` selects recorded input."""
    parser, _ = _main_parser(config)
    return parser.get("Input", "inputMode", fallback="simulation").strip().lower() == "raw"


def load_raw_dataset(config: ConfigHandler | ConfigParser | Path | str) -> RawDataset:
    """Resolve raw configuration and return a repeatable streaming dataset."""
    raw_config = _load_raw_config(config)
    _validate_headers(raw_config)
    return RawDataset(raw_config)


def _main_parser(config: ConfigHandler | ConfigParser | Path | str) -> tuple[ConfigParser, Path]:
    if isinstance(config, ConfigHandler):
        return config.parser, Path(config.config_file).resolve()
    if isinstance(config, ConfigParser):
        return config, Path.cwd() / "configuration.ini"
    path = Path(config).resolve()
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    if not parser.read(path):
        raise FileNotFoundError(f"Raw input configuration not found: {path}")
    return parser, path


def _load_raw_config(config: ConfigHandler | ConfigParser | Path | str) -> RawInputConfig:
    main, main_path = _main_parser(config)
    raw_path = None
    raw_parser = None
    raw_name = main.get("Input", "rawDataConfig", fallback="").strip()
    if raw_name:
        raw_path = Path(raw_name)
        if not raw_path.is_absolute():
            raw_path = main_path.parent / raw_path
        raw_path = raw_path.resolve()
        raw_parser = ConfigParser(interpolation=ExtendedInterpolation())
        if not raw_parser.read(raw_path):
            raise FileNotFoundError(f"Raw data sub-configuration not found: {raw_path}")
    view = _ConfigView(main, main_path, raw_parser, raw_path)
    chunk_size = _integer(view.get("RawData", "chunkSize", "100000"), "chunkSize")
    if chunk_size <= 0:
        raise ValueError("[RawData] chunkSize must be positive")
    policy = str(view.get("RawData", "invalidRecordPolicy", "strict")).lower()
    if policy not in {"strict", "drop"}:
        raise ValueError("[RawData] invalidRecordPolicy must be 'strict' or 'drop'")
    imu_input_level = str(
        view.get("RawData", "imuInputLevel", "measurement")).strip().lower()
    if imu_input_level not in {"measurement", "truth"}:
        raise ValueError(
            "[RawData] imuInputLevel must be 'measurement' or 'truth'")
    specs = tuple(_build_spec(view, section, kind) for section, kind in _SENSOR_SECTIONS.items()
                  if _section_enabled(view, section))
    if not specs:
        raise ValueError("Raw input has no enabled sensor sections")
    initial = _initial_state(view)
    effective = ConfigParser(interpolation=ExtendedInterpolation())
    effective.read_dict({section: dict(main.items(section)) for section in main.sections()})
    if raw_parser is not None:
        for section in raw_parser.sections():
            if not effective.has_section(section):
                effective.add_section(section)
            for name, value in raw_parser.items(section):
                effective.set(section, name, value)
    declaring_base = (raw_path or main_path).parent
    return RawInputConfig(
        specs, chunk_size, policy, initial, main_path, raw_path,
        effective, declaring_base, imu_input_level)


def _section_enabled(view: _ConfigView, section: str) -> bool:
    if not view.has_section(section):
        return False
    value = str(view.get(section, "enabled", "true")).lower()
    if value not in {"true", "false", "yes", "no", "1", "0", "on", "off"}:
        raise ValueError(f"[{section}] enabled must be a boolean")
    return value in {"true", "yes", "1", "on"}


def _build_spec(view: _ConfigView, section: str, kind: str) -> SensorSpec:
    global_file, global_path = view.get_with_base(
        "RawData", ("file", "combinedFile"))
    file_name, file_declaring_path = view.get_with_base(
        section, ("file", "inputFile"), global_file)
    if file_declaring_path is None:
        file_declaring_path = global_path
    if not file_name:
        raise ValueError(f"[{section}] needs file or [RawData] combinedFile")
    file_path = Path(str(file_name))
    if not file_path.is_absolute():
        declaring_path = file_declaring_path or view.main_path
        file_path = declaring_path.parent / file_path
    file_path = file_path.resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"[{section}] input file not found: {file_path}")

    timestamp = _column(view, section, "timestamp", "timestampColumn", "timeColumn")
    columns: dict[str, str] = {"timestamp": timestamp}
    required: tuple[str, ...]
    if kind in {"accelerometer", "gyroscope"}:
        columns.update(_vector_columns(view, section, ""))
        required = ("x", "y", "z")
    elif kind == "gnss":
        columns.update({
            "latitude": _column(view, section, "latitude", "latitudeColumn", "latColumn"),
            "longitude": _column(view, section, "longitude", "longitudeColumn", "lonColumn"),
            "altitude": _column(view, section, "altitude", "altitudeColumn", "heightColumn"),
        })
        for field_name, aliases in {
            "velocity_x": ("velocityXColumn", "velocityNorthColumn", "northVelocityColumn"),
            "velocity_y": ("velocityYColumn", "velocityEastColumn", "eastVelocityColumn"),
            "velocity_z": ("velocityZColumn", "velocityDownColumn", "downVelocityColumn"),
            "valid": ("validColumn", "validityColumn"),
            "fix_type": ("fixTypeColumn",), "satellite_count": ("satelliteCountColumn", "satellitesColumn"),
            "horizontal_accuracy": ("horizontalAccuracyColumn",),
            "vertical_accuracy": ("verticalAccuracyColumn",),
            "speed_accuracy": ("speedAccuracyColumn",),
        }.items():
            value = view.get(section, aliases)
            if value:
                columns[field_name] = value
        velocity_fields = ("velocity_x", "velocity_y", "velocity_z")
        mapped_velocity = [name in columns for name in velocity_fields]
        if any(mapped_velocity) and not all(mapped_velocity):
            raise ValueError(
                "[RawGnss] velocity mappings must include all three components")
        required = ("latitude", "longitude", "altitude")
    elif kind == "altimeter":
        columns["altitude"] = _column(view, section, "altitude", "altitudeColumn", "heightColumn")
        required = ("altitude",)
    elif kind == "quantum_imu":
        columns.update(_vector_columns(view, section, "acceleration", "acceleration"))
        columns.update(_vector_columns(view, section, "angular_rate", "angularRate"))
        required = ("acceleration_x", "acceleration_y", "acceleration_z",
                    "angular_rate_x", "angular_rate_y", "angular_rate_z")
    elif kind == "gradiometer":
        columns["top_signal"] = _column(view, section, "topSignal", "topSignalColumn", "topColumn")
        columns["bottom_signal"] = _column(view, section, "bottomSignal", "bottomSignalColumn", "bottomColumn")
        required = ("top_signal", "bottom_signal")
    else:
        aliases = {
            "latitude": ("latitudeColumn", "latColumn"), "longitude": ("longitudeColumn", "lonColumn"),
            "altitude": ("altitudeColumn", "heightColumn"),
            "velocity_x": ("velocityXColumn", "velocityNorthColumn"),
            "velocity_y": ("velocityYColumn", "velocityEastColumn"),
            "velocity_z": ("velocityZColumn", "velocityDownColumn"),
            "acceleration_x": ("accelerationXColumn",), "acceleration_y": ("accelerationYColumn",),
            "acceleration_z": ("accelerationZColumn",), "heading": ("headingColumn",),
            "pitch": ("pitchColumn",), "roll": ("rollColumn",),
            "quaternion_w": ("quaternionWColumn",),
            "quaternion_x": ("quaternionXColumn",),
            "quaternion_y": ("quaternionYColumn",),
            "quaternion_z": ("quaternionZColumn",),
            "angular_rate_x": ("angularRateXColumn",), "angular_rate_y": ("angularRateYColumn",),
            "angular_rate_z": ("angularRateZColumn",),
        }
        for field_name, names in aliases.items():
            value = view.get(section, names)
            if value:
                columns[field_name] = value
        groups = {
            "position": ("latitude", "longitude", "altitude"),
            "velocity": ("velocity_x", "velocity_y", "velocity_z"),
            "acceleration": ("acceleration_x", "acceleration_y", "acceleration_z"),
            "Euler attitude": ("heading", "pitch", "roll"),
            "quaternion attitude": (
                "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"),
            "angular rate": ("angular_rate_x", "angular_rate_y", "angular_rate_z"),
        }
        mapped_groups = []
        for group_name, fields in groups.items():
            mapped = [field_name in columns for field_name in fields]
            if any(mapped) and not all(mapped):
                raise ValueError(
                    f"[RawReference] {group_name} mappings must be complete")
            if all(mapped):
                mapped_groups.append(group_name)
        if not mapped_groups:
            raise ValueError("[RawReference] must map at least one complete state group")
        if "Euler attitude" in mapped_groups and "quaternion attitude" in mapped_groups:
            raise ValueError(
                "[RawReference] map either Euler or quaternion attitude, not both")
        required = tuple(name for name in columns if name != "timestamp")

    options = _csv_options(view, section)
    angle_fallback = view.get(section, "angleUnits", "degrees")
    units = {
        "acceleration": str(view.get(section, "accelerationUnits",
                                     view.get(section, "units", "m/s^2"))).lower(),
        "angular_rate": str(view.get(section, ("angularRateUnits", "rateUnits"),
                                     view.get(section, "units", "deg/s"))).lower(),
        # Keep angle for callers/specs created before coordinate and attitude
        # units were made independent.
        "angle": str(angle_fallback).lower(),
        "coordinate_angle": str(
            view.get(section, "coordinateUnits", angle_fallback)).lower(),
        "attitude_angle": str(
            view.get(section, "attitudeUnits", angle_fallback)).lower(),
        "length": str(view.get(section, ("lengthUnits", "altitudeUnits"),
                                 view.get(section, "units", "m"))).lower(),
        "velocity": str(view.get(section, "velocityUnits", "m/s")).lower(),
    }
    frame = view.get(section, ("frame", "sourceFrame", "velocityFrame"))
    signs = tuple(_number(view.get(section, f"{axis}Sign", "1"), f"{axis}Sign")
                  for axis in "xyz")
    if any(sign not in {-1.0, 1.0} for sign in signs):
        raise ValueError(f"[{section}] axis signs must be +1 or -1")
    quality = {}
    for name in ("minimumFixType", "minimumSatelliteCount", "maximumHorizontalAccuracy",
                 "maximumVerticalAccuracy", "maximumSpeedAccuracy"):
        value = view.get(section, name)
        if value is not None:
            quality[name] = _number(value, name)
    for name in ("maximumHorizontalAccuracy", "maximumVerticalAccuracy"):
        if name in quality:
            quality[name] *= _length_factor(units["length"])
    if "maximumSpeedAccuracy" in quality:
        quality["maximumSpeedAccuracy"] *= _velocity_factor(units["velocity"])
    quaternion_direction = None
    if kind == "reference" and "quaternion_w" in columns:
        direction = view.get(section, "quaternionDirection")
        if direction is None:
            raise ValueError(
                "[RawReference] quaternionDirection is required for quaternion mappings")
        normalized_direction = _normalize_direction(str(direction))
        if normalized_direction not in {"bodytoned", "nedtobody"}:
            raise ValueError(
                "[RawReference] quaternionDirection must be bodyToNed or nedToBody")
        quaternion_direction = normalized_direction
    return SensorSpec(
        section, kind, file_path, columns, required, options,
        str(view.get(section, "timeUnit", view.get("RawData", "timeUnit", "auto"))).lower(),
        view.get(section, ("datetimeFormat", "timestampFormat"),
                 view.get("RawData", ("datetimeFormat", "timestampFormat"))),
        str(view.get(section, ("timezone", "sourceTimezone"),
                     view.get("RawData", ("timezone", "sourceTimezone"), "UTC"))),
        units, str(frame).lower() if frame else None, signs, quality,
        quaternion_direction)


def _column(view: _ConfigView, section: str, field_name: str, *aliases: str) -> str:
    value = view.get(section, aliases)
    if not value:
        raise ValueError(f"[{section}] missing {aliases[0]} mapping for {field_name}")
    return value


def _vector_columns(view: _ConfigView, section: str, prefix: str,
                    option_prefix: str = "") -> dict[str, str]:
    result = {}
    for axis in "xyz":
        field_name = f"{prefix}_{axis}" if prefix else axis
        camel = f"{option_prefix}{axis.upper()}Column" if option_prefix else f"{axis}Column"
        result[field_name] = _column(view, section, field_name, camel)
    return result


def _csv_options(view: _ConfigView, section: str) -> CsvOptions:
    def inherited(names: str | tuple[str, ...], fallback: Any = None) -> Any:
        return view.get(section, names, view.get("RawData", names, fallback))
    delimiter = str(inherited(("delimiter", "separator"), ","))
    quotechar = str(inherited("quotechar", '"'))
    comment = inherited("comment")
    decimal = str(inherited("decimal", "."))
    if len(delimiter) != 1 or len(quotechar) != 1 or len(decimal) != 1:
        raise ValueError("CSV delimiter, quotechar, and decimal must be one character")
    if comment is not None and len(str(comment)) != 1:
        raise ValueError("CSV comment must be one character")
    return CsvOptions(delimiter, quotechar, str(comment) if comment else None,
                      str(inherited("encoding", "utf-8")), decimal)


def _initial_state(view: _ConfigView) -> RawInitialState:
    if not view.has_section("RawInitialState"):
        return RawInitialState()

    def vector(names: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]):
        values = []
        for aliases in names:
            value = view.get("RawInitialState", aliases)
            values.append(None if value is None else _number(value, aliases[0]))
        return tuple(values) if any(value is not None for value in values) else None

    return RawInitialState(
        position=vector((("latitude",), ("longitude",), ("altitude",))),
        velocity=vector((("velocityX", "velocityNorth"), ("velocityY", "velocityEast"),
                         ("velocityZ", "velocityDown"))),
        acceleration=vector((("accelerationX",), ("accelerationY",), ("accelerationZ",))),
        attitude=vector((("heading",), ("pitch",), ("roll",))),
        angle_rates=vector((("angularRateX",), ("angularRateY",), ("angularRateZ",))),
    )


def _validate_headers(config: RawInputConfig) -> None:
    grouped: dict[tuple[Path, CsvOptions], set[str]] = {}
    for spec in config.specs:
        grouped.setdefault((spec.file, spec.options), set()).update(spec.columns.values())
    for (path, options), wanted in grouped.items():
        header = pd.read_csv(path, nrows=0, sep=options.delimiter,
                             quotechar=options.quotechar, comment=options.comment,
                             encoding=options.encoding, decimal=options.decimal)
        missing = wanted.difference(header.columns)
        if missing:
            raise ValueError(f"{path}: missing mapped CSV columns: {', '.join(sorted(missing))}")


def _iter_file(path: Path, specs: list[SensorSpec], config: RawInputConfig,
               report: IngestionReport) -> Iterator[MeasurementEvent]:
    """Merge independently ordered sensor streams declared in one CSV.

    A combined file may map a different timestamp column for each sensor.
    Reading one row at a time and yielding that row's events is not sufficient:
    a later row for one sensor can precede an earlier row for another.  Each
    sensor projection is therefore streamed independently and heap-merged.
    This retains bounded memory and does not assume row-level time alignment.
    """
    timing = {
        (spec.columns["timestamp"], spec.time_unit,
         spec.datetime_format, spec.timezone)
        for spec in specs
    }
    if len(timing) == 1:
        yield from _iter_shared_timestamp_file(path, specs, config, report)
        return

    streams = [
        iter(_iter_spec_file(path, spec, config, report, index == 0))
        for index, spec in enumerate(specs)
    ]
    heap: list[tuple[float, int, MeasurementEvent,
                     Iterator[MeasurementEvent]]] = []
    serial = 0
    for stream in streams:
        try:
            event = next(stream)
        except StopIteration:
            continue
        heappush(heap, (event.timestamp, serial, event, stream))
        serial += 1
    while heap:
        _, _, event, stream = heappop(heap)
        yield event
        try:
            following = next(stream)
        except StopIteration:
            continue
        heappush(heap, (following.timestamp, serial, following, stream))
        serial += 1


def _iter_shared_timestamp_file(
        path: Path, specs: list[SensorSpec], config: RawInputConfig,
        report: IngestionReport) -> Iterator[MeasurementEvent]:
    """Read a conventionally time-aligned combined CSV once per pass."""
    options = specs[0].options
    usecols = list(dict.fromkeys(
        column for spec in specs for column in spec.columns.values()))
    last_times: dict[str, float | Fraction | None] = {
        spec.section: None for spec in specs}
    row_offset = 2
    chunks = pd.read_csv(
        path, usecols=usecols, chunksize=config.chunk_size,
        sep=options.delimiter, quotechar=options.quotechar,
        comment=options.comment, encoding=options.encoding,
        decimal=options.decimal, dtype=object)
    for chunk in chunks:
        report.rows_read += len(chunk)
        for row_values in chunk.itertuples(index=False, name=None):
            row = dict(zip(chunk.columns, row_values))
            for spec in specs:
                event, last_time = _validated_event(
                    spec, row, path, row_offset, last_times[spec.section],
                    config, report)
                last_times[spec.section] = last_time
                if event is not None:
                    yield event
            row_offset += 1


def _iter_spec_file(path: Path, spec: SensorSpec, config: RawInputConfig,
                    report: IngestionReport,
                    count_rows: bool) -> Iterator[MeasurementEvent]:
    """Stream and validate one sensor projection from a CSV file."""
    options = spec.options
    usecols = list(dict.fromkeys(spec.columns.values()))
    last_time: float | None = None
    row_offset = 2
    chunks = pd.read_csv(path, usecols=usecols, chunksize=config.chunk_size,
                         sep=options.delimiter, quotechar=options.quotechar,
                         comment=options.comment, encoding=options.encoding,
                         decimal=options.decimal, dtype=object)
    for chunk in chunks:
        if count_rows:
            report.rows_read += len(chunk)
        for row_values in chunk.itertuples(index=False, name=None):
            row = dict(zip(chunk.columns, row_values))
            event, last_time = _validated_event(
                spec, row, path, row_offset, last_time, config, report)
            if event is not None:
                yield event
            row_offset += 1


def _record_rejection_error(report: IngestionReport, message: str) -> None:
    if len(report.errors) < _MAX_REPORTED_ERRORS:
        report.errors.append(message)
    else:
        report.errors_omitted += 1


def _validated_event(
        spec: SensorSpec, row: Mapping[str, Any], path: Path, row_offset: int,
        last_time: float | Fraction | None, config: RawInputConfig,
        report: IngestionReport) -> tuple[
            MeasurementEvent | None, float | Fraction | None]:
    """Parse one projected row and apply stream-local validation/reporting."""
    try:
        event = _event_from_row(spec, row, path, row_offset)
    except (TypeError, ValueError, OverflowError) as error:
        if config.invalid_record_policy == "strict":
            raise ValueError(f"{path}:{row_offset}: {error}") from error
        report.rejected += 1
        _record_rejection_error(report, f"{path}:{row_offset}: {error}")
        if "partial" in str(error).lower():
            report.partial_records += 1
        return None, last_time
    if event is None:
        return None, last_time
    if event.kind == "gnss" and not event.payload.accepted:
        report.quality_filtered += 1
        return None, last_time
    if last_time is not None and event.timestamp <= last_time:
        duplicate = event.timestamp == last_time
        if duplicate:
            report.duplicate_timestamps += 1
        else:
            report.backward_timestamps += 1
        if config.invalid_record_policy == "strict":
            relation = "duplicate" if duplicate else "backward"
            raise ValueError(
                f"{path}:{row_offset}: {relation} timestamp {event.timestamp}")
        report.rejected += 1
        return None, last_time
    return event, event.timestamp


def _event_from_row(spec: SensorSpec, row: Mapping[str, Any], source: Path,
                    row_number: int) -> MeasurementEvent | None:
    if spec.kind == "reference":
        if not _validate_reference_row(spec, row):
            return None
    else:
        required_values = [row[spec.columns[name]] for name in spec.required]
        present = [not _missing(value) for value in required_values]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(f"partial {spec.kind} measurement")
    raw_time = row[spec.columns["timestamp"]]
    if _missing(raw_time):
        raise ValueError(f"missing timestamp for {spec.kind} measurement")
    timestamp, absolute = _timestamp(raw_time, spec)
    payload = _payload(spec, row)
    return MeasurementEvent(timestamp, spec.kind, payload, absolute, source,
                            row_number, spec.frame)


def _payload(spec: SensorSpec, row: Mapping[str, Any]) -> Any:
    get = lambda name: _number(
        row[spec.columns[name]], spec.columns[name], spec.options.decimal)
    if spec.kind == "accelerometer":
        factor = _acceleration_factor(spec.units["acceleration"])
        return tuple(get(axis) * factor * sign for axis, sign in zip("xyz", spec.signs))
    if spec.kind == "gyroscope":
        factor = _angular_factor(spec.units["angular_rate"])
        return tuple(get(axis) * factor * sign for axis, sign in zip("xyz", spec.signs))
    if spec.kind == "altimeter":
        return get("altitude") * _length_factor(spec.units["length"])
    if spec.kind == "gradiometer":
        return get("top_signal"), get("bottom_signal")
    if spec.kind == "quantum_imu":
        af = _acceleration_factor(spec.units["acceleration"])
        rf = _angular_factor(spec.units["angular_rate"])
        return {
            "acceleration": tuple(get(f"acceleration_{axis}") * af * sign
                                  for axis, sign in zip("xyz", spec.signs)),
            "angular_rate": tuple(get(f"angular_rate_{axis}") * rf * sign
                                  for axis, sign in zip("xyz", spec.signs)),
        }
    if spec.kind == "gnss":
        angle = _angle_factor(spec.units.get("coordinate_angle", spec.units["angle"]))
        length = _length_factor(spec.units["length"])
        latitude = get("latitude") * angle
        longitude = get("longitude") * angle
        altitude = get("altitude") * length
        velocity = None
        velocity_fields = ("velocity_x", "velocity_y", "velocity_z")
        velocity_present = [name in spec.columns and not _missing(row[spec.columns[name]])
                            for name in velocity_fields]
        if any(velocity_present) and not all(velocity_present):
            raise ValueError("partial GNSS velocity measurement")
        if all(velocity_present):
            vf = _velocity_factor(spec.units["velocity"])
            velocity = tuple(get(name) * vf for name in velocity_fields)
            velocity_frame = spec.frame or "ned"
            if velocity_frame == "enu":
                velocity = (velocity[1], velocity[0], -velocity[2])
            elif velocity_frame == "ecef":
                converted = vel_ecef2vel_ned(
                    np.asarray(velocity), np.asarray((latitude, longitude, altitude)))
                velocity = tuple(float(value) for value in converted)
            elif velocity_frame != "ned":
                raise ValueError(f"unsupported GNSS velocity frame '{velocity_frame}'")
        optional: dict[str, Any] = {}
        for name in ("fix_type", "horizontal_accuracy", "vertical_accuracy", "speed_accuracy"):
            if name in spec.columns and not _missing(row[spec.columns[name]]):
                optional[name] = get(name)
        for name in ("horizontal_accuracy", "vertical_accuracy"):
            if name in optional:
                optional[name] *= length
        if "speed_accuracy" in optional:
            optional["speed_accuracy"] *= _velocity_factor(
                spec.units["velocity"])
        if "satellite_count" in spec.columns and not _missing(row[spec.columns["satellite_count"]]):
            optional["satellite_count"] = int(get("satellite_count"))
        valid = None
        if "valid" in spec.columns and not _missing(row[spec.columns["valid"]]):
            valid = _boolean(row[spec.columns["valid"]])
        accepted = valid is not False
        q = spec.quality
        accepted &= _minimum(optional.get("fix_type"), q.get("minimumFixType"))
        accepted &= _minimum(optional.get("satellite_count"), q.get("minimumSatelliteCount"))
        accepted &= _maximum(optional.get("horizontal_accuracy"), q.get("maximumHorizontalAccuracy"))
        accepted &= _maximum(optional.get("vertical_accuracy"), q.get("maximumVerticalAccuracy"))
        accepted &= _maximum(optional.get("speed_accuracy"), q.get("maximumSpeedAccuracy"))
        return GnssFix(
            latitude, longitude, altitude, velocity, "ned", valid,
            optional.get("fix_type"), optional.get("satellite_count"),
            optional.get("horizontal_accuracy"), optional.get("vertical_accuracy"),
            optional.get("speed_accuracy"), bool(accepted))
    return _reference_payload(spec, row, get)


_REFERENCE_GROUPS = {
    "position": ("latitude", "longitude", "altitude"),
    "velocity": ("velocity_x", "velocity_y", "velocity_z"),
    "acceleration": ("acceleration_x", "acceleration_y", "acceleration_z"),
    "Euler attitude": ("heading", "pitch", "roll"),
    "quaternion attitude": (
        "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"),
    "angular rate": ("angular_rate_x", "angular_rate_y", "angular_rate_z"),
}


def _validate_reference_row(spec: SensorSpec, row: Mapping[str, Any]) -> bool:
    """Validate complete independently optional state groups in one row."""
    has_group = False
    for group_name, fields in _REFERENCE_GROUPS.items():
        if not all(name in spec.columns for name in fields):
            continue
        present = [not _missing(row[spec.columns[name]]) for name in fields]
        if any(present) and not all(present):
            raise ValueError(f"partial reference {group_name} measurement")
        has_group |= all(present)
    return has_group


def _reference_payload(spec: SensorSpec, row: Mapping[str, Any],
                       get: Callable[[str], float]) -> dict[str, float]:
    """Normalize a reference row while retaining its flat payload contract."""
    result = {}
    for name, column in spec.columns.items():
        if (name == "timestamp" or name.startswith("quaternion_")
                or name.startswith("velocity_") or _missing(row[column])):
            continue
        value = _number(row[column], column)
        if name in {"latitude", "longitude"}:
            value *= _angle_factor(
                spec.units.get("coordinate_angle", spec.units["angle"]))
        elif name in {"heading", "pitch", "roll"}:
            value *= _angle_factor(
                spec.units.get("attitude_angle", spec.units["angle"]))
        elif name == "altitude":
            value *= _length_factor(spec.units["length"])
        elif name.startswith("acceleration_"):
            value *= _acceleration_factor(spec.units["acceleration"])
            value *= spec.signs["xyz".index(name[-1])]
        elif name.startswith("angular_rate_"):
            value *= _angular_factor(spec.units["angular_rate"])
            value *= spec.signs["xyz".index(name[-1])]
        result[name] = value

    quaternion_fields = (
        "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z")
    if all(name in spec.columns and not _missing(row[spec.columns[name]])
           for name in quaternion_fields):
        quaternion = np.asarray([get(name) for name in quaternion_fields])
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("reference quaternion norm must be non-zero")
        quaternion /= norm
        if spec.quaternion_direction == "nedtobody":
            quaternion[1:] *= -1
        for name, value in zip(quaternion_fields, quaternion):
            result[name] = float(value)
        attitude = quaternion_to_euler(quaternion)
        result.update(zip(("heading", "pitch", "roll"), map(float, attitude)))

    velocity_fields = ("velocity_x", "velocity_y", "velocity_z")
    if all(name in spec.columns and not _missing(row[spec.columns[name]])
           for name in velocity_fields):
        factor = _velocity_factor(spec.units["velocity"])
        velocity = np.asarray([
            get(name) * factor * sign
            for name, sign in zip(velocity_fields, spec.signs)
        ])
        frame = spec.frame or "body"
        if frame != "body":
            if frame == "enu":
                velocity = velocity[[1, 0, 2]]
                velocity[2] *= -1
            elif frame == "ecef":
                position_fields = ("latitude", "longitude", "altitude")
                if not all(name in result for name in position_fields):
                    raise ValueError(
                        "reference ECEF velocity requires position in the same row")
                velocity = vel_ecef2vel_ned(
                    velocity, np.asarray([result[name] for name in position_fields]))
            elif frame != "ned":
                raise ValueError(
                    f"unsupported reference velocity frame '{frame}'")
            attitude_fields = ("heading", "pitch", "roll")
            if not all(name in result for name in attitude_fields):
                raise ValueError(
                    "reference non-body velocity requires attitude in the same row")
            attitude = np.radians([result[name] for name in attitude_fields])
            velocity = rotate_3d(*attitude) @ velocity
        result.update(zip(velocity_fields, map(float, velocity)))
    return result


def _timestamp(
        value: Any, spec: SensorSpec) -> tuple[
            float | Fraction, datetime | None]:
    unit = spec.time_unit.replace(" ", "").lower()
    factors = {
        "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
        "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
        "us": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
        "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
    }
    unix_factors = {
        "unix": 1.0, "unixs": 1.0, "unix_s": 1.0,
        "epoch": 1.0, "epochs": 1.0, "epoch_s": 1.0,
        "unixms": 1e-3, "unix_ms": 1e-3,
        "epochms": 1e-3, "epoch_ms": 1e-3,
        "unixus": 1e-6, "unix_us": 1e-6,
        "epochus": 1e-6, "epoch_us": 1e-6,
        "unixns": 1e-9, "unix_ns": 1e-9,
        "epochns": 1e-9, "epoch_ns": 1e-9,
    }
    if unit in factors:
        return (_number(value, spec.columns["timestamp"], spec.options.decimal)
                * factors[unit]), None
    if unit in unix_factors:
        timestamp = _fraction_number(
            value, spec.columns["timestamp"], spec.options.decimal)
        timestamp *= Fraction(str(unix_factors[unit]))
        try:
            absolute = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        except (OSError, OverflowError, ValueError) as error:
            raise ValueError(f"invalid Unix timestamp '{value}'") from error
        return timestamp, absolute
    if unit == "auto":
        try:
            return _number(
                value, spec.columns["timestamp"], spec.options.decimal), None
        except ValueError:
            pass
    elif unit not in {"datetime", "iso8601", "date", "timestamp"}:
        raise ValueError(f"unsupported time unit '{spec.time_unit}'")
    try:
        parsed = pd.to_datetime(value, format=spec.datetime_format, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid datetime '{value}'") from error
    if parsed.tzinfo is None:
        try:
            parsed = parsed.tz_localize(ZoneInfo(spec.timezone))
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"unknown source timezone '{spec.timezone}'") from error
    parsed = parsed.tz_convert("UTC")
    absolute = parsed.to_pydatetime()
    return absolute.timestamp(), absolute


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


def _normalized_numeric_text(value: Any, decimal: str) -> str:
    text = str(value).strip()
    if decimal != ".":
        text = text.replace(decimal, ".")
    return text


def _number(value: Any, field_name: str, decimal: str = ".") -> float:
    if _missing(value):
        raise ValueError(f"missing numeric value for {field_name}")
    try:
        number = float(_normalized_numeric_text(value, decimal))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric value '{value}' for {field_name}") from error
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value for {field_name}")
    return number


def _fraction_number(value: Any, field_name: str,
                     decimal: str = ".") -> Fraction:
    """Parse a finite decimal without discarding sub-float timestamp bits."""
    if _missing(value):
        raise ValueError(f"missing numeric value for {field_name}")
    try:
        number = Fraction(_normalized_numeric_text(value, decimal))
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"invalid numeric value '{value}' for {field_name}") from error
    return number


def _integer(value: Any, field_name: str) -> int:
    number = _number(value, field_name)
    if not number.is_integer():
        raise ValueError(f"{field_name} must be an integer")
    return int(number)


def _boolean(value: Any) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "on", "valid"}:
        return True
    if normalized in {"false", "no", "0", "off", "invalid"}:
        return False
    raise ValueError(f"invalid boolean value '{value}'")


def _minimum(value: float | None, threshold: float | None) -> bool:
    return threshold is None or (value is not None and value >= threshold)


def _maximum(value: float | None, threshold: float | None) -> bool:
    return threshold is None or (value is not None and value <= threshold)


def _normalize_unit(unit: str) -> str:
    return unit.lower().replace(" ", "").replace("²", "2").replace("°", "deg")


def _normalize_direction(direction: str) -> str:
    return (direction.strip().lower().replace("_", "")
            .replace("-", "").replace(" ", "").replace("2", "to"))


def _acceleration_factor(unit: str) -> float:
    normalized = _normalize_unit(unit)
    factors = {"m/s2": 1.0, "m/s^2": 1.0, "ms-2": 1.0, "g": _G, "mg": _G / 1000.0}
    if normalized not in factors:
        raise ValueError(f"unsupported acceleration unit '{unit}'")
    return factors[normalized]


def _angular_factor(unit: str) -> float:
    normalized = _normalize_unit(unit)
    factors = {"deg/s": 1.0, "degree/s": 1.0, "degrees/s": 1.0,
               "rad/s": 180.0 / math.pi, "radian/s": 180.0 / math.pi,
               "radians/s": 180.0 / math.pi}
    if normalized not in factors:
        raise ValueError(f"unsupported angular-rate unit '{unit}'")
    return factors[normalized]


def _angle_factor(unit: str) -> float:
    normalized = _normalize_unit(unit)
    if normalized in {"deg", "degree", "degrees"}:
        return 1.0
    if normalized in {"rad", "radian", "radians"}:
        return 180.0 / math.pi
    raise ValueError(f"unsupported angular unit '{unit}'")


def _length_factor(unit: str) -> float:
    normalized = _normalize_unit(unit)
    factors = {"m": 1.0, "metre": 1.0, "metres": 1.0, "meter": 1.0, "meters": 1.0,
               "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}
    if normalized not in factors:
        raise ValueError(f"unsupported length unit '{unit}'")
    return factors[normalized]


def _velocity_factor(unit: str) -> float:
    normalized = _normalize_unit(unit)
    factors = {"m/s": 1.0, "meter/second": 1.0, "metre/second": 1.0,
               "km/h": 1.0 / 3.6, "kmph": 1.0 / 3.6,
               "knot": 0.5144444444444445, "knots": 0.5144444444444445, "kt": 0.5144444444444445}
    if normalized not in factors:
        raise ValueError(f"unsupported velocity unit '{unit}'")
    return factors[normalized]
