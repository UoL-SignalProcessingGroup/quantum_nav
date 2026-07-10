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
