
import pytest

from qnav.gps.ephemeris import SatelliteType
from qnav.gps.satellite import SatelliteParams
from qnav.gps import ephemeris
from datetime import datetime
from pathlib import Path


# @pytest.mark.dependency()
def test_get_path(tmp_date: datetime, tmp_path: Path):

    # Validate the path's directory and file name
    result = ephemeris.get_rinex_path(tmp_date, tmp_path)
    assert result.name == tmp_date.strftime("%Y_%j.rnx")
    assert result.parent == tmp_path


def test_valid_download(tmp_date: datetime, tmp_path: Path):
    
    # Firstly, ensure the file does not already exist.
    eph_file = ephemeris.get_rinex_path(tmp_date, tmp_path)
    assert not eph_file.exists()

    # Secondly, obtain the file and ensure its downloaded.
    ephemeris.download_rinex_file(tmp_date, tmp_path)
    assert eph_file.exists()

    # Thirdly, check the file is not empty.
    assert eph_file.stat().st_size > 0


def test_invalid_download(invalid_date: datetime, tmp_path: Path):

    # Ensure FileNotFoundError is raised for out-of-range date
    with pytest.raises(FileNotFoundError):
        ephemeris.download_rinex_file(invalid_date, tmp_path)


def test_read_local_rinex(local_rinex_file):

    # Attempt to read the pre-downloaded rinex file.
    rinex_data = ephemeris.read_rinex_file(local_rinex_file)
    satellites = ephemeris.rinex_to_satellite_params(rinex_data)

    # Ensure each of the returned objects is a satellite instance.
    assert all([s for s in satellites if isinstance(s, SatelliteParams)])


def test_read_remote_rinex(tmp_date: datetime, tmp_path: Path):

    # Download the requested rinex file.
    eph_file = ephemeris.get_rinex_path(tmp_date, tmp_path)
    ephemeris.download_rinex_file(tmp_date, tmp_path)

    # Attempt to read the downloaded rinex file.
    rinex_data = ephemeris.read_rinex_file(eph_file)
    satellites = ephemeris.rinex_to_satellite_params(rinex_data)

    # Ensure each of the returned objects is a satellite instance.
    assert all([s for s in satellites if isinstance(s, SatelliteParams)])


def test_fixed_position_rinex(tmp_path: Path):

    test_date = datetime(2020, 11, 10)
    ephemeris.download_rinex_file(test_date, tmp_path)
    test_file = ephemeris.get_rinex_path(test_date, tmp_path)

    rinex_data = ephemeris.read_rinex_file(test_file)
    ephemeris.rinex_to_satellite_params(rinex_data)


def test_rinex_filtering(local_rinex_file: Path):

    # Attempt to read the pre-downloaded rinex file.
    rinex_data = ephemeris.read_rinex_file(local_rinex_file)

    # Iterate over all valid satellite types:
    for rinex_type in [s for s in SatelliteType]:

        # Filter the satellites by current type
        satellites = ephemeris.rinex_to_satellite_params(
            rinex_data, rinex_type)

        # Ensure all types are valid:
        for sat in satellites:
            assert sat.type == rinex_type