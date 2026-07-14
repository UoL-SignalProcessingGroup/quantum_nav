from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from qnav.fusion.ins import NumericalINS
from qnav.input.config_handler import ConfigHandler
from qnav.output.data import ResultsTable
import qnav.simulation.replay as replay_module
from qnav.simulation.replay import run_raw_replay


def _write_imu_replay_config(
        tmp_path: Path, *, input_level: str = "measurement",
        frame: str = "sensor", include_gnss: bool = False) -> Path:
    gnss_section = """
[RawGnss]
timestampColumn = time
latitudeColumn = lat
longitudeColumn = lon
altitudeColumn = alt
""" if include_gnss else ""
    config_path = tmp_path / "imu_replay.ini"
    config_path.write_text(
        f"""
[Input]
inputMode = raw

[RawData]
combinedFile = imu.csv
timeUnit = s
imuInputLevel = {input_level}

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
accelerationUnits = m/s2
frame = {frame}

[RawGyroscope]
timestampColumn = time
xColumn = gx
yColumn = gy
zColumn = gz
angularRateUnits = deg/s
frame = {frame}

{gnss_section}

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

[Vehicle]
sensorAngleX = 90
sensorAngleY = 0
sensorAngleZ = 0
leverArmAngleX = 0
leverArmAngleY = 0
leverArmAngleZ = 0

[Estimation]
estimatedStateModel = simple
integrationMethod = numerical

[GPS]
gpsFusionMethod = fixedgain
gpsFixedGain = 1

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


def test_first_complete_imu_pair_only_starts_the_integration_clock(
        tmp_path: Path):
    (tmp_path / "imu.csv").write_text(
        "time,ax,ay,az,gx,gy,gz,lat,lon,alt\n"
        "0,,,,,,,52,-2,100\n"
        "5,1,0,-9.80665,0,0,0,,,\n"
        "6,1,0,-9.80665,0,0,0,,,\n",
        encoding="utf-8",
    )
    intervals = []

    def capture_interval(self, state, time_step=None):
        intervals.append(time_step)

    with patch.object(NumericalINS, "perform_fusion", capture_interval):
        run_raw_replay(ConfigHandler(_write_imu_replay_config(
            tmp_path, include_gnss=True)))

    assert intervals == [pytest.approx(1.0)]


def test_staggered_truth_imu_initializes_held_pair_without_losing_interval(
        tmp_path: Path):
    (tmp_path / "imu.csv").write_text(
        "time,ax,ay,az,gx,gy,gz\n"
        "0,0,0,-9.80665,,,\n"
        "0.05,,,,0,0,0\n"
        "0.10,0,0,-9.80665,,,\n"
        "0.15,,,,0,0,0\n",
        encoding="utf-8",
    )
    intervals = []

    def capture_interval(self, state, time_step=None):
        intervals.append(time_step)

    with patch.object(NumericalINS, "perform_fusion", capture_interval):
        run_raw_replay(ConfigHandler(_write_imu_replay_config(
            tmp_path, input_level="truth", frame="body")))

    assert intervals == pytest.approx([0.05, 0.05])


def test_quantum_fusion_advances_once_per_complete_staggered_imu_pair(
        tmp_path: Path, monkeypatch):
    (tmp_path / "imu.csv").write_text(
        "time,ax,ay,az,gx,gy,gz,qax,qay,qaz,qgx,qgy,qgz\n"
        "0,0,0,-9.80665,,,,,,,,,\n"
        "0.005,,,,0,0,0,,,,,,\n"
        "0.010,0,0,-9.80665,,,,,,,,,\n"
        "0.015,,,,0,0,0,,,,,,\n"
        "0.020,0,0,-9.80665,,,,,,,,,\n"
        "0.025,,,,0,0,0,,,,,,\n"
        "0.030,,,,,,,0,0,-9.80665,0,0,0\n",
        encoding="utf-8",
    )
    config_path = _write_imu_replay_config(tmp_path)
    with config_path.open("a", encoding="utf-8") as config_file:
        config_file.write(
            """
[RawQuantumImu]
timestampColumn = time
accelerationXColumn = qax
accelerationYColumn = qay
accelerationZColumn = qaz
angularRateXColumn = qgx
angularRateYColumn = qgy
angularRateZColumn = qgz
accelerationUnits = m/s2
angularRateUnits = deg/s
frame = sensor

[Quantum]
quantumImuFrequency = 1
"""
        )
    calls = []
    pending_calls = []

    class QuantumFusion:
        def __init__(self, accelerometer, gyroscope):
            self.accelerometer = accelerometer
            self.gyroscope = gyroscope

        def perform_fusion(self, state):
            calls.append((
                self.accelerometer.timestamp,
                self.gyroscope.timestamp,
            ))
            return False

        def apply_pending_measurement(self, state):
            pending_calls.append(self.accelerometer.timestamp)
            return True

    monkeypatch.setattr(
        replay_module.quantum_conf, "get_quantum_imu_fusion",
        lambda config, sensor, accelerometer, gyroscope: QuantumFusion(
            accelerometer, gyroscope))

    run_raw_replay(ConfigHandler(config_path))

    assert calls == pytest.approx([
        (0.0, 0.005),
        (0.010, 0.015),
        (0.020, 0.025),
    ])
    assert pending_calls == pytest.approx([0.020])


@pytest.mark.parametrize(
    ("input_level", "frame", "acceleration"),
    [
        ("measurement", "body", (1.0, 0.0, -9.80665)),
        ("measurement", "sensor", (0.0, -1.0, -9.80665)),
        ("truth", "body", (1.0, 0.0, -9.80665)),
        ("truth", "sensor", (0.0, -1.0, -9.80665)),
    ],
)
def test_raw_imu_frame_is_honored_with_nonzero_mounting(
        tmp_path: Path, input_level: str, frame: str, acceleration):
    ax, ay, az = acceleration
    (tmp_path / "imu.csv").write_text(
        "time,ax,ay,az,gx,gy,gz\n"
        f"0,{ax},{ay},{az},0,0,0\n"
        f"0.1,{ax},{ay},{az},0,0,0\n",
        encoding="utf-8",
    )

    results = run_raw_replay(ConfigHandler(_write_imu_replay_config(
        tmp_path, input_level=input_level, frame=frame)))

    final_velocity = results.states[-1, 4:7]
    assert final_velocity[0] == pytest.approx(0.1, rel=5e-3)
    assert final_velocity[1] == pytest.approx(0.0, abs=1e-5)


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


def test_reference_aligner_interpolates_sparse_groups_independently():
    aligner = replay_module._ReferenceAligner()
    first_position = np.array([52.0, -2.0, 100.0])
    second_position = np.array([52.002, -2.0, 100.0])
    aligner.push(0.0, {
        "latitude": first_position[0],
        "longitude": first_position[1],
        "altitude": first_position[2],
    })
    aligner.push(1.0, {
        "velocity_x": 1.0,
        "velocity_y": 2.0,
        "velocity_z": 3.0,
    })
    aligner.add_target(1.0)
    aligner.push(2.0, {
        "latitude": second_position[0],
        "longitude": second_position[1],
        "altitude": second_position[2],
    })

    result = aligner.finish()
    expected_position = replay_module.trans.ecef2lla(
        (replay_module.trans.lla2ecef(first_position)
         + replay_module.trans.lla2ecef(second_position)) / 2)
    np.testing.assert_allclose(result["position"][0], expected_position)
    np.testing.assert_allclose(result["velocity"][0], [1, 2, 3])
    assert all(not aligner._pending[field] for field in (
        "acceleration", "attitude", "angle_rates"))


def test_replay_dispatches_altimeter_quantum_imu_and_gradiometer(
        tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "extended_sensors.csv"
    csv_path.write_text(
        "time,ax,ay,az,gx,gy,gz,alt,qax,qay,qaz,qgx,qgy,qgz,top,bottom\n"
        "0,0,0,-9.80665,0,0,0,300,1,2,3,4,5,6,0.6,0.1\n"
        "1,0,0,-9.80665,0,0,0,321,7,8,9,10,11,12,0.9,0.2\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "extended.ini"
    config_path.write_text(
        f"""
[Input]
inputMode = raw
[RawData]
combinedFile = {csv_path.name}
timeUnit = s
[RawInitialState]
latitude = 52
longitude = -2
altitude = 100
velocityX = 0
velocityY = 0
velocityZ = 0
accelerationX = 0
accelerationY = 0
accelerationZ = -9.80665
heading = 0
pitch = 0
roll = 0
angularRateX = 0
angularRateY = 0
angularRateZ = 0
[RawAccelerometer]
timestampColumn = time
xColumn = ax
yColumn = ay
zColumn = az
accelerationUnits = m/s2
frame = sensor
[RawGyroscope]
timestampColumn = time
xColumn = gx
yColumn = gy
zColumn = gz
angularRateUnits = deg/s
frame = sensor
[RawAltimeter]
timestampColumn = time
altitudeColumn = alt
altitudeUnits = m
[RawQuantumImu]
timestampColumn = time
accelerationXColumn = qax
accelerationYColumn = qay
accelerationZColumn = qaz
angularRateXColumn = qgx
angularRateYColumn = qgy
angularRateZColumn = qgz
accelerationUnits = m/s2
angularRateUnits = deg/s
frame = sensor
[RawGravityGradiometer]
timestampColumn = time
topSignalColumn = top
bottomSignalColumn = bottom
[Measurement]
imuMeasurementFreq = 1
[Estimation]
estimatedStateModel = simple
integrationMethod = numerical
[Altimeter]
altimeterFusion = fixedgain
altimeterGainAmount = 1
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

    quantum_measurements = []
    gradient_measurements = []

    class QuantumFusion:
        def __init__(self, sensor):
            self.sensor = sensor

        def apply_pending_measurement(self, state):
            return False

        def perform_fusion(self, state):
            acceleration, angle_rates = self.sensor.last_measurement
            quantum_measurements.append((acceleration, angle_rates))
            state.update_estimates(
                acceleration=acceleration, angle_rates=angle_rates)
            self.sensor.reset()

    class GradientFusion:
        def __init__(self, sensor):
            self.sensor = sensor

        def perform_fusion(self, state):
            top, bottom = self.sensor.last_measurement
            gradient_measurements.append((top, bottom))
            position = state.position
            position[1] = top - bottom
            state.update_estimates(position=position)
            self.sensor.reset()

    monkeypatch.setattr(
        replay_module.quantum_conf, "get_quantum_imu_fusion",
        lambda config, sensor, accelerometer, gyroscope: QuantumFusion(sensor))
    monkeypatch.setattr(
        replay_module.quantum_grav_conf, "get_quantum_grav_fusion",
        lambda config, sensor: GradientFusion(sensor))

    results = run_raw_replay(ConfigHandler(config_path))

    assert results.report.accepted == 10
    assert len(quantum_measurements) == len(gradient_measurements) == 2
    np.testing.assert_allclose(quantum_measurements[-1][0], [7, 8, 9])
    np.testing.assert_allclose(quantum_measurements[-1][1], [10, 11, 12])
    np.testing.assert_allclose(gradient_measurements[-1], [0.9, 0.2])
    np.testing.assert_allclose(results.states[-1, 1:4], [52, 0.7, 321])
    np.testing.assert_allclose(results.states[-1, 7:10], [7, 8, 9])
    np.testing.assert_allclose(results.states[-1, 13:16], [10, 11, 12])
