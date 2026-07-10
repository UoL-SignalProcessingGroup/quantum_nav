"""Focused tests for configuration-driven recorded sensor input."""

from configparser import ConfigParser
from pathlib import Path

import math
import pytest

from qnav.input.config_handler import ConfigHandler
from qnav.input.raw import GnssFix, load_raw_dataset, is_raw_mode


def _write(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_combined_csv_is_chunked_repeatable_and_coincident_streams_are_allowed(tmp_path):
    _write(tmp_path / "sensors.csv", """
time,ax,ay,az,gx,gy,gz
0,1,2,3,0.1,0.2,0.3
1000,4,5,6,0.4,0.5,0.6
2000,7,8,9,0.7,0.8,0.9
""")
    config_path = _write(tmp_path / "main.ini", """
[Input]
inputMode = raw
[RawData]
combinedFile = sensors.csv
chunkSize = 1
[RawAccelerometer]
timestampColumn = time
timeUnit = ms
xColumn = ax
yColumn = ay
zColumn = az
units = g
xSign = -1
[RawGyroscope]
timestampColumn = time
timeUnit = ms
xColumn = gx
yColumn = gy
zColumn = gz
units = rad/s
""")

    handler = ConfigHandler(config_path)
    assert is_raw_mode(handler)
    dataset = load_raw_dataset(handler)
    first_pass = list(dataset.iter_events())
    second_pass = list(dataset.iter_events())

    assert [(event.timestamp, event.kind) for event in first_pass] == [
        (0.0, "accelerometer"), (0.0, "gyroscope"),
        (1.0, "accelerometer"), (1.0, "gyroscope"),
        (2.0, "accelerometer"), (2.0, "gyroscope"),
    ]
    assert first_pass == second_pass
    assert first_pass[0].payload == pytest.approx((-9.80665, 19.6133, 29.41995))
    assert first_pass[1].payload == pytest.approx(tuple(v * 180 / math.pi for v in (.1, .2, .3)))
    assert dataset.report.rows_read == 3
    assert dataset.report.accepted == 6

    discovery = dataset.discover()
    assert discovery.earliest_event.timestamp == 0
    assert discovery.latest_event.timestamp == 2
    assert discovery.first_gnss_fix is None


def test_raw_subconfig_and_each_csv_resolve_against_declaring_file(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write(raw_dir / "accel.csv", "t,a,b,c\n0,1,2,3\n2,4,5,6")
    _write(raw_dir / "gyro.csv", "t,p,q,r\n1,7,8,9\n3,10,11,12")
    _write(raw_dir / "mapping.ini", """
[RawData]
chunkSize = 2
[RawAccelerometer]
file = accel.csv
timestampColumn = t
xColumn = a
yColumn = b
zColumn = c
[RawGyroscope]
file = gyro.csv
timestampColumn = t
xColumn = p
yColumn = q
zColumn = r
""")
    main = _write(tmp_path / "main.ini", """
[Input]
inputMode = raw
rawDataConfig = raw/mapping.ini
""")

    dataset = load_raw_dataset(main)
    assert dataset.declaring_base == raw_dir
    assert dataset.mapping_config.has_section("RawAccelerometer")
    assert dataset.config.is_enabled("accelerometer")
    assert dataset.config.spec_for("gyroscope").file == raw_dir / "gyro.csv"
    assert [(event.timestamp, event.kind) for event in dataset] == [
        (0, "accelerometer"), (1, "gyroscope"),
        (2, "accelerometer"), (3, "gyroscope"),
    ]


def test_datetime_gnss_quality_and_velocity_normalization(tmp_path):
    _write(tmp_path / "gnss.csv", """
utc,lat,lon,height,ve,vn,vu,valid,sats,hacc
2026-07-10T10:00:00,53,-3,100,2,1,3,true,3,1
2026-07-10T10:00:01,53.1,-3.1,101,5,4,6,true,8,2
""")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = gnss.csv
[RawGnss]
timestampColumn = utc
timeUnit = datetime
timezone = Europe/London
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = height
velocityXColumn = ve
velocityYColumn = vn
velocityZColumn = vu
velocityFrame = enu
validColumn = valid
satelliteCountColumn = sats
horizontalAccuracyColumn = hacc
minimumSatelliteCount = 5
maximumHorizontalAccuracy = 3
""")

    dataset = load_raw_dataset(config)
    events = list(dataset)
    assert len(events) == 1
    assert events[0].absolute_time.isoformat().startswith("2026-07-10T09:00:01+00:00")
    fix = events[0].payload
    assert isinstance(fix, GnssFix)
    assert fix.velocity == pytest.approx((4, 5, -6))
    assert fix.velocity_frame == "ned"
    assert dataset.report.quality_filtered == 1
    assert dataset.discover().first_gnss_fix == fix


def test_drop_policy_reports_partial_and_backward_records(tmp_path):
    _write(tmp_path / "imu.csv", """
t,x,y,z
0,1,2,3
1,4,,6
-1,7,8,9
2,10,11,12
""")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = imu.csv
invalidRecordPolicy = drop
[RawAccelerometer]
timestampColumn = t
xColumn = x
yColumn = y
zColumn = z
""")

    dataset = load_raw_dataset(config)
    assert [event.timestamp for event in dataset] == [0, 2]
    assert dataset.report.rejected == 2
    assert dataset.report.partial_records == 1
    assert dataset.report.backward_timestamps == 1


def test_strict_policy_rejects_duplicate_timestamps(tmp_path):
    _write(tmp_path / "imu.csv", "t,x,y,z\n0,1,2,3\n0,4,5,6")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = imu.csv
[RawAccelerometer]
timestampColumn = t
xColumn = x
yColumn = y
zColumn = z
""")
    dataset = load_raw_dataset(config)
    with pytest.raises(ValueError, match="duplicate timestamp"):
        list(dataset)


def test_missing_mapped_column_fails_before_iteration(tmp_path):
    _write(tmp_path / "imu.csv", "t,x,y\n0,1,2")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = imu.csv
[RawAccelerometer]
timestampColumn = t
xColumn = x
yColumn = y
zColumn = z
""")
    with pytest.raises(ValueError, match="missing mapped CSV columns: z"):
        load_raw_dataset(config)


def test_initial_state_components_are_exposed(tmp_path):
    _write(tmp_path / "imu.csv", "t,x,y,z\n0,1,2,3")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = imu.csv
[RawInitialState]
latitude = 53
longitude = -3
heading = 90
pitch = 1
roll = -2
[RawAccelerometer]
timestampColumn = t
xColumn = x
yColumn = y
zColumn = z
""")
    initial = load_raw_dataset(config).config.initial_state
    assert initial.position == (53, -3, None)
    assert initial.attitude == (90, 1, -2)


def test_parser_without_input_section_defaults_to_simulation():
    parser = ConfigParser()
    parser.read_dict({"RawData": {"file": "unused.csv"}})
    assert not is_raw_mode(parser)
