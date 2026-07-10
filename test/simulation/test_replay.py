from pathlib import Path

import numpy as np

from qnav.input.config_handler import ConfigHandler
from qnav.output.data import ResultsTable
from qnav.simulation.replay import run_raw_replay


def test_raw_imu_gnss_replay_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "sensors.csv"
    csv_path.write_text(
        "time,ax,ay,az,gx,gy,gz,lat,lon,alt,vn,ve,vd\n"
        "0.0,0,0,-9.80665,0,0,0,52.0,-2.0,100,0,0,0\n"
        "0.1,0,0,-9.80665,0,0,0,52.0,-2.0,100,0,0,0\n"
        "0.2,0,0,-9.80665,0,0,0,52.0,-2.0,100,0,0,0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "replay.ini"
    config_path.write_text(
        f"""
[Input]
inputMode = raw

[RawData]
combinedFile = {csv_path.name}
timeUnit = s

[RawInitialState]
heading = 0
pitch = 0
roll = 0

[RawAccelerometer]
timestampColumn = time
xColumn = ax
yColumn = ay
zColumn = az
accelerationUnits = m/s^2

[RawGyroscope]
timestampColumn = time
xColumn = gx
yColumn = gy
zColumn = gz
angularRateUnits = deg/s

[RawGnss]
timestampColumn = time
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = alt
velocityXColumn = vn
velocityYColumn = ve
velocityZColumn = vd
velocityFrame = ned

[Measurement]
imuMeasurementFreq = 10

[Estimation]
estimatedStateModel = simple
integrationMethod = numerical

[GPS]
gpsFusionMethod = fixedgain
gpsFixedGain = 1.0

[Gravity]
estimatedGravityFunction = somigliana
estimatedGravityCorrectionMap = none

[Geoid]
estimatedGeoidModel = none

[Database]
geoidDatabase = databases/geoid

[Output]
resultsDownSampleRate = 1
""",
        encoding="utf-8",
    )

    results = run_raw_replay(ConfigHandler(config_path))

    assert len(results.time_steps) >= 2
    assert np.all(np.diff(results.time_steps) >= 0)
    assert np.isfinite(results.states).all()
    assert np.allclose(results.states[-1, 1:4], [52.0, -2.0, 100.0])
    assert results.report.accepted == 9

    table = ResultsTable(tmp_path / "estimation.qnr")
    results.write(table)
    assert table.is_data_loaded
    assert "position" in table.get_data()["data"]


def test_truth_imu_uses_sensor_models_and_can_export_measurements(tmp_path: Path):
    csv_path = tmp_path / "truth_imu.csv"
    csv_path.write_text(
        "time,ax,ay,az,gx,gy,gz\n"
        "0.0,0,0,-9.80665,0,0,0\n"
        "0.1,0,0,-9.80665,0,0,0\n"
        "0.2,0,0,-9.80665,0,0,0\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "truth.ini"
    config_path.write_text(
        f"""
[Input]
inputMode = raw

[RawData]
combinedFile = {csv_path.name}
timeUnit = s
imuInputLevel = truth

[RawInitialState]
latitude = 52
longitude = -2
altitude = 100
heading = 0
pitch = 0
roll = 0

[RawAccelerometer]
timestampColumn = time
xColumn = ax
yColumn = ay
zColumn = az
accelerationUnits = m/s^2

[RawGyroscope]
timestampColumn = time
xColumn = gx
yColumn = gy
zColumn = gz
angularRateUnits = deg/s

[Measurement]
imuMeasurementFreq = 10
accelerometerInitialStaticBiasMean = 0
accelerometerBiasDriftRate = 0
accelerometerScaleErrorMean = 0
accelerometerMeasurementError = 0
accelerometerNonOrthogonalityMean = 0
gyroscopeInitialStaticBiasMean = 0
gyroscopeBiasDriftRate = 0
gyroscopeScaleErrorMean = 0
gyroscopeMeasurementError = 0
gyroscopeNonOrthogonalityMean = 0
useGaussianMarkovNoise = no

[Estimation]
estimatedStateModel = simple
integrationMethod = numerical

[Gravity]
estimatedGravityFunction = somigliana
estimatedGravityCorrectionMap = none

[Geoid]
estimatedGeoidModel = none

[Database]
geoidDatabase = databases/geoid

[Random]
initialRandomSeed = 123
errorRandomSeed = 456

[Output]
resultsDownSampleRate = 1
saveProcessedImu = yes
""",
        encoding="utf-8",
    )

    results = run_raw_replay(ConfigHandler(config_path))
    results.write_processed_imu(tmp_path)

    acceleration = np.genfromtxt(
        tmp_path / "processed_accelerometer.csv", delimiter=",", names=True)
    gyroscope = np.genfromtxt(
        tmp_path / "processed_gyroscope.csv", delimiter=",", names=True)
    assert np.allclose(acceleration["z"], -9.80665)
    assert np.allclose(acceleration["x"], 0)
    assert np.allclose(gyroscope["x"], 0)
    assert len(acceleration) == len(gyroscope) == 3

    noisy_text = config_path.read_text(encoding="utf-8").replace(
        "accelerometerMeasurementError = 0",
        "accelerometerMeasurementError = 1000").replace(
        "gyroscopeMeasurementError = 0",
        "gyroscopeMeasurementError = 1000")
    config_path.write_text(noisy_text, encoding="utf-8")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    run_raw_replay(ConfigHandler(config_path)).write_processed_imu(first_dir)
    run_raw_replay(ConfigHandler(config_path)).write_processed_imu(second_dir)
    first = np.genfromtxt(
        first_dir / "processed_accelerometer.csv", delimiter=",", names=True)
    second = np.genfromtxt(
        second_dir / "processed_accelerometer.csv", delimiter=",", names=True)
    for field in ("x", "y", "z"):
        assert np.array_equal(first[field], second[field])
    assert not np.allclose(first["z"], -9.80665)


def _reference_replay_config(tmp_path: Path, initial_state: str = "") -> Path:
    config_path = tmp_path / "reference_replay.ini"
    config_path.write_text(
        f"""
[Input]
inputMode = raw

[RawData]
timeUnit = s
imuInputLevel = truth

{initial_state}

[RawAccelerometer]
file = imu.csv
timestampColumn = time
xColumn = ax
yColumn = ay
zColumn = az
accelerationUnits = m/s2

[RawGyroscope]
file = imu.csv
timestampColumn = time
xColumn = gx
yColumn = gy
zColumn = gz
angularRateUnits = deg/s

[RawReference]
file = truth.csv
timestampColumn = time
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = alt
velocityXColumn = vx
velocityYColumn = vy
velocityZColumn = vz
velocityFrame = body
quaternionWColumn = qw
quaternionXColumn = qx
quaternionYColumn = qy
quaternionZColumn = qz
quaternionDirection = bodyToNed

[Measurement]
imuMeasurementFreq = 10
accelerometerInitialStaticBiasMean = 0
accelerometerBiasDriftRate = 0
accelerometerScaleErrorMean = 0
accelerometerMeasurementError = 0
accelerometerNonOrthogonalityMean = 0
gyroscopeInitialStaticBiasMean = 0
gyroscopeBiasDriftRate = 0
gyroscopeScaleErrorMean = 0
gyroscopeMeasurementError = 0
gyroscopeNonOrthogonalityMean = 0
useGaussianMarkovNoise = no

[Estimation]
estimatedStateModel = simple
integrationMethod = numerical

[Gravity]
estimatedGravityFunction = somigliana
estimatedGravityCorrectionMap = none

[Geoid]
estimatedGeoidModel = none

[Database]
geoidDatabase = databases/geoid

[Random]
initialRandomSeed = 123
errorRandomSeed = 456

[Output]
resultsDownSampleRate = 1
""",
        encoding="utf-8",
    )
    return config_path


def test_reference_initializes_and_exports_on_exact_estimate_times(tmp_path: Path):
    (tmp_path / "imu.csv").write_text(
        "time,ax,ay,az,gx,gy,gz\n"
        "0.0,0,0,-9.80665,0,0,0\n"
        "0.1,0,0,-9.80665,0,0,0\n"
        "0.2,0,0,-9.80665,0,0,0\n",
        encoding="utf-8",
    )
    root_half = np.sqrt(0.5)
    (tmp_path / "truth.csv").write_text(
        "time,lat,lon,alt,vx,vy,vz,qw,qx,qy,qz\n"
        f"0.0,52,-2,100,1,2,3,{root_half},0,0,{root_half}\n"
        f"0.1,52.0001,-2,101,4,5,6,{root_half},0,0,{root_half}\n"
        f"0.2,52.0002,-2,102,7,8,9,{root_half},0,0,{root_half}\n",
        encoding="utf-8",
    )

    results = run_raw_replay(ConfigHandler(_reference_replay_config(tmp_path)))

    np.testing.assert_allclose(results.time_steps, [0, 0.1, 0.2])
    np.testing.assert_allclose(results.states[0, 1:4], [52, -2, 100])
    np.testing.assert_allclose(results.states[0, 4:7], [1, 2, 3])
    np.testing.assert_allclose(results.states[0, 10:13], [90, 0, 0])
    np.testing.assert_allclose(results.reference["position"][:, 0],
                               [52, 52.0001, 52.0002])
    np.testing.assert_allclose(results.reference["velocity"],
                               [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_allclose(results.reference["attitude"],
                               [[90, 0, 0]] * 3, atol=1e-12)

    table = ResultsTable(tmp_path / "ground_truth.qnr")
    assert results.write_reference(table)
    np.testing.assert_array_equal(table.get_data()["time_steps"], results.time_steps)
    assert set(table.get_data()["data"]) == {"position", "velocity", "attitude"}


def test_reference_slerp_has_no_extrapolation_and_preserves_initial_tick(tmp_path: Path):
    (tmp_path / "imu.csv").write_text(
        "time,ax,ay,az,gx,gy,gz\n"
        "0.0,0,0,-9.80665,0,0,0\n"
        "0.1,0,0,-9.80665,0,0,0\n"
        "0.2,0,0,-9.80665,0,0,0\n"
        "0.3,0,0,-9.80665,0,0,0\n",
        encoding="utf-8",
    )
    root_half = np.sqrt(0.5)
    (tmp_path / "truth.csv").write_text(
        "time,lat,lon,alt,vx,vy,vz,qw,qx,qy,qz\n"
        "0.05,52,-2,100,0,0,0,1,0,0,0\n"
        f"0.25,52,-2,100,0,0,0,{-root_half},0,0,{-root_half}\n",
        encoding="utf-8",
    )
    initial = """
[RawInitialState]
latitude = 52
longitude = -2
altitude = 100
velocityX = 0
velocityY = 0
velocityZ = 0
heading = 0
pitch = 0
roll = 0
"""

    results = run_raw_replay(ConfigHandler(
        _reference_replay_config(tmp_path, initial)))

    np.testing.assert_allclose(results.time_steps, [0, 0.1, 0.2, 0.3])
    assert np.isnan(results.reference["attitude"][[0, 3]]).all()
    np.testing.assert_allclose(
        results.reference["attitude"][1:3],
        [[22.5, 0, 0], [67.5, 0, 0]], atol=1e-10)
