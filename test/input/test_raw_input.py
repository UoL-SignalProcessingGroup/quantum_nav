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
imuInputLevel = truth
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
    raw_config = load_raw_dataset(config).config
    initial = raw_config.initial_state
    assert initial.position == (53, -3, None)
    assert initial.attitude == (90, 1, -2)
    assert raw_config.imu_input_level == "truth"


def test_parser_without_input_section_defaults_to_simulation():
    parser = ConfigParser()
    parser.read_dict({"RawData": {"file": "unused.csv"}})
    assert not is_raw_mode(parser)


def test_reference_quaternion_is_normalized_and_converted_to_euler(tmp_path):
    _write(tmp_path / "truth.csv", """
t,qw,qx,qy,qz
0,2,0,0,2
""")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
quaternionWColumn = qw
quaternionXColumn = qx
quaternionYColumn = qy
quaternionZColumn = qz
quaternionDirection = bodyToNed
""")

    dataset = load_raw_dataset(config)
    event = next(iter(dataset))
    payload = event.payload
    assert payload["quaternion_w"] == pytest.approx(math.sqrt(0.5))
    assert payload["quaternion_x"] == pytest.approx(0)
    assert payload["quaternion_y"] == pytest.approx(0)
    assert payload["quaternion_z"] == pytest.approx(math.sqrt(0.5))
    assert payload["heading"] == pytest.approx(90)
    assert payload["pitch"] == pytest.approx(0)
    assert payload["roll"] == pytest.approx(0)
    assert dataset.discover().first_reference_event.payload == payload


def test_reference_ned_to_body_quaternion_is_conjugated(tmp_path):
    _write(tmp_path / "truth.csv", "t,qw,qx,qy,qz\n0,1,0,0,-1")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
quaternionWColumn = qw
quaternionXColumn = qx
quaternionYColumn = qy
quaternionZColumn = qz
quaternionDirection = ned_to_body
""")
    payload = next(iter(load_raw_dataset(config))).payload
    assert payload["quaternion_z"] == pytest.approx(math.sqrt(0.5))
    assert payload["heading"] == pytest.approx(90)


@pytest.mark.parametrize("direction, error", [
    (None, "quaternionDirection is required"),
    ("spacecraftToWorld", "quaternionDirection must be bodyToNed or nedToBody"),
])
def test_reference_quaternion_direction_is_explicit(tmp_path, direction, error):
    _write(tmp_path / "truth.csv", "t,qw,qx,qy,qz\n0,1,0,0,0")
    direction_line = "" if direction is None else f"quaternionDirection = {direction}"
    config = _write(tmp_path / "raw.ini", f"""
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
quaternionWColumn = qw
quaternionXColumn = qx
quaternionYColumn = qy
quaternionZColumn = qz
{direction_line}
""")
    with pytest.raises(ValueError, match=error):
        load_raw_dataset(config)


def test_reference_rejects_incomplete_groups_and_zero_quaternion(tmp_path):
    _write(tmp_path / "truth.csv", "t,qw,qx,qy,qz\n0,0,0,0,0")
    incomplete = _write(tmp_path / "incomplete.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
quaternionWColumn = qw
quaternionXColumn = qx
quaternionYColumn = qy
quaternionDirection = bodyToNed
""")
    with pytest.raises(ValueError, match="quaternion attitude mappings must be complete"):
        load_raw_dataset(incomplete)

    complete = _write(tmp_path / "complete.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
quaternionWColumn = qw
quaternionXColumn = qx
quaternionYColumn = qy
quaternionZColumn = qz
quaternionDirection = bodyToNed
""")
    with pytest.raises(ValueError, match="quaternion norm must be non-zero"):
        list(load_raw_dataset(complete))


def test_reference_groups_are_optional_per_row_but_complete_when_present(tmp_path):
    _write(tmp_path / "truth.csv", """
t,lat,lon,alt,vx,vy,vz
0,0.1,0.2,10,,,
1,,,,1,2,3
2,0.3,,12,,,
""")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = alt
velocityXColumn = vx
velocityYColumn = vy
velocityZColumn = vz
velocityFrame = body
""")
    dataset = load_raw_dataset(config)
    events = dataset.iter_events()
    assert next(events).payload == {"latitude": .1, "longitude": .2, "altitude": 10}
    assert next(events).payload == {
        "velocity_x": 1, "velocity_y": 2, "velocity_z": 3}
    with pytest.raises(ValueError, match="partial reference position measurement"):
        next(events)


def test_reference_units_frames_and_vector_signs_normalize_to_body(tmp_path):
    _write(tmp_path / "truth.csv", """
t,lat,lon,alt,h,p,r,v1,v2,v3,ax,ay,az,wx,wy,wz
0,0.5,-0.25,100,1.5707963267948966,0,0,1,2,3,4,5,6,0.1,0.2,0.3
""")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = alt
headingColumn = h
pitchColumn = p
rollColumn = r
velocityXColumn = v1
velocityYColumn = v2
velocityZColumn = v3
accelerationXColumn = ax
accelerationYColumn = ay
accelerationZColumn = az
angularRateXColumn = wx
angularRateYColumn = wy
angularRateZColumn = wz
coordinateUnits = radians
attitudeUnits = radians
velocityUnits = km/h
accelerationUnits = g
angularRateUnits = rad/s
velocityFrame = enu
xSign = -1
ySign = 1
zSign = -1
""")
    payload = next(iter(load_raw_dataset(config))).payload
    assert payload["latitude"] == pytest.approx(math.degrees(.5))
    assert payload["longitude"] == pytest.approx(math.degrees(-.25))
    assert payload["heading"] == pytest.approx(90)
    # Signed ENU (-E, N, -U) -> NED (N, -E, U), then heading 90 NED->body.
    assert tuple(payload[f"velocity_{axis}"] for axis in "xyz") == pytest.approx(
        (-1 / 3.6, -2 / 3.6, 3 / 3.6))
    assert tuple(payload[f"acceleration_{axis}"] for axis in "xyz") == pytest.approx(
        (-4 * 9.80665, 5 * 9.80665, -6 * 9.80665))
    assert tuple(payload[f"angular_rate_{axis}"] for axis in "xyz") == pytest.approx(
        tuple(value * 180 / math.pi for value in (-.1, .2, -.3)))


def test_reference_angle_units_remains_fallback_for_both_angle_domains(tmp_path):
    _write(tmp_path / "truth.csv", "t,lat,lon,alt,h,p,r\n0,0.1,0.2,3,0.3,0.4,0.5")
    config = _write(tmp_path / "raw.ini", """
[Input]
inputMode = raw
[RawData]
file = truth.csv
[RawReference]
timestampColumn = t
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = alt
headingColumn = h
pitchColumn = p
rollColumn = r
angleUnits = radians
""")
    payload = next(iter(load_raw_dataset(config))).payload
    for name, value in (("latitude", .1), ("longitude", .2),
                        ("heading", .3), ("pitch", .4), ("roll", .5)):
        assert payload[name] == pytest.approx(math.degrees(value))
