"""
=====================
download_all_files.py
=====================

:summary:
    A simple script used for downloading all resource files.
    This script allows the defining of an area to download all
    database files to automatically. The location to save the
    files to can be defined.

:authors:
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""

from datetime import datetime
from pathlib import Path

from qnav.gravity.geosat import Geosat44
from qnav.gravity.ggm_plus import GGMPlusAcc
from qnav.gravity.ggm_plus import GGMPlusDist
from qnav.gravity.marine import MarineGravity
from qnav.gravity.somigliana import Somigliana
from qnav.gravity.srtm2gravity import SRTM2GravityFull
from qnav.gravity.srtm2gravity import SRTM2GravityRes
from qnav.gps.ephemeris import download_rinex_files


def download_dted(database_dir: Path,
                  lat_range: tuple[int, int],
                  lon_range: tuple[int, int],
                  dted_level: int = 2):
    """
    Download DTED elevation files given by area range and level.
    This function will download Digital Terrain Elevation (DTED) files
    contained within a given latitude and longitude range.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path

    :param lat_range: The latitude range to download for. Any file
        covering an area in this range will be downloaded.
    :type lat_range: tuple[int, int]

    :param lon_range: The longitude range to download for. Any file
        covering an area in this range will be downloaded.
    :type lon_range: tuple[int, int]

    :param dted_level: The DTED level to download (supports 1 and 2).
    :type dted_level: int
    """
    raise NotImplementedError(
        'Unfortunately, no longer supported '
        'due to changes to EarthExplorers API')


def download_geoid(database_dir: Path,
                   model: str = 'egm2008-1'):
    """
    Download PGM geoid raster grid by model name.
    This function will download raster matrices represented as Pixel Gray Map
     (PGM) files to the selected location.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path

    :param model: The model name to download (supports 'egm84-30', 'egm84-15',
        'egm96-15', 'egm96-5', 'egm2008-5', 'egm2008-2_5' or 'egm2008-1').
    :type model: str
    """
    raise NotImplementedError(
        'Unfortunately, not supported due to '
        'hosting site complications')


def download_geosat_gravity(database_dir: Path):
    """
    Download Geosat-44 gravity map model to given location.
    This function will download the geosat-44 gravity map data to the given
    database directory location.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path
    """

    grav_dir = database_dir / 'gravity' / 'geosat'
    grav_model = Somigliana()
    geoid_model = None

    print(f'Downloading geosat-44')
    geosat = Geosat44(grav_model, geoid_model, grav_dir, auto_load=False)
    geosat.download_data()

    print(f'Finished downloading Geosat-44 gravity map.')


def download_marine_gravity(database_dir: Path):
    """
    Download Marine gravity map model to given location.
    This function will download the marine gravity map data to the given
    database directory location.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path
    """

    grav_dir = database_dir / 'gravity' / 'marine'
    grav_model = Somigliana()
    geoid_model = None

    print(f'Downloading Marine Gravity')
    marine = MarineGravity(grav_model, geoid_model, grav_dir, auto_load=False)
    marine.download_data()

    print(f'Finished downloading Marine Gravity Map')


def download_ggm_plus(database_dir: Path,
                      lat_range: tuple[int, int],
                      lon_range: tuple[int, int],
                      model_type: str = 'disturbance'):
    """
    Download GGMPlus data for given area range and type.
    This function will download the GGMPlus data files
    contained within a given latitude and longitude range.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path

    :param lat_range: The latitude range to download for. Any file
        covering an area in this range will be downloaded.
    :type lat_range: tuple[int, int]

    :param lon_range: The longitude range to download for. Any file
        covering an area in this range will be downloaded.
    :type lon_range: tuple[int, int]
    
    :param model_type: The GGMPlus data type to download 
        (supports 'acceleration' or 'disturbance').
    :type model_type: str
    """

    grav_dir = database_dir / 'gravity' / 'ggm_plus'
    grav_model = Somigliana()
    geoid_model = None

    match model_type:

        case 'acceleration':
            ggm_plus = GGMPlusAcc(grav_model, geoid_model, grav_dir)
            model_name = 'GGMPlus Acceleration'

        case 'disturbance':
            ggm_plus = GGMPlusDist(grav_model, geoid_model, grav_dir)
            model_name = 'GGMPlus Disturbance'

        case _:
            raise ValueError(f'Unrecognized model type \'{model_type}\'')

    for i_lat in range(*lat_range):
        for i_lon in range(*lon_range):
            print(f'Downloading {model_name}: \t ({i_lat}, {i_lon})')
            ggm_plus.download_data(i_lat, i_lon)

    print(f'Finished downloading {model_name}.')


def download_srtm2gravity(database_dir: Path,
                          lat_range: tuple[int, int],
                          lon_range: tuple[int, int],
                          model_type: str = 'full'):
    """
    Download srtm2gravity data for given area range and type.
    This function will download the SRTM 2 Gravity data files
    contained within a given latitude and longitude range.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path

    :param lat_range: The latitude range to download for. Any file
        covering an area in this range will be downloaded.
    :type lat_range: tuple[int, int]

    :param lon_range: The longitude range to download for. Any file
        covering an area in this range will be downloaded.
    :type lon_range: tuple[int, int]

    :param model_type: The SRTM2Gravity data type to download
        (supports 'full' or 'residual').
    :type model_type: str
    """

    grav_dir = database_dir / 'gravity' / 'srtm2gravity'
    grav_model = Somigliana()
    geoid_model = None

    match model_type:

        case 'full':
            srtm = SRTM2GravityFull(grav_model, geoid_model, grav_dir)
            model_name = 'Srtm2Gravity Full-Scale'

        case 'residual':
            srtm = SRTM2GravityRes(grav_model, geoid_model, grav_dir)
            model_name = 'Srtm2Gravity Residual'

        case _:
            raise ValueError(f'Unrecognized model type \'{model_type}\'')

    for i_lat in range(*lat_range):
        for i_lon in range(*lon_range):
            print(f'Downloading {model_name}: \t ({i_lat}, {i_lon})')
            srtm.download_data(i_lat, i_lon)

    print(f'Finished downloading {model_name}.')


def download_gps_data(database_dir: Path,
                      start_date: datetime,
                      end_date: datetime):
    """
    Download GPS Rinex files for given date range (inclusive).
    This function will download the GPS satellite Rinex files across a given
    date range, acquiring the date for each day.

    :param database_dir: The database folder location to download to.
    :type database_dir: Path

    :param start_date: The earliest date to download the data from.
    :type start_date: datetime

    :param end_date: The latest date to download the data until (included).
    :type end_date: datetime
    """

    gps_dir = database_dir / 'gps' / 'rinex'
    download_rinex_files(start_date, end_date, gps_dir)
    print(f'Finished downloading GPS files.')


# When script is called directly.
if __name__ == '__main__':

    # Specify the save location
    drive_dir = Path('/') / 'Volumes' / 'T9'
    db_dir = drive_dir / 'quantum nav' / 'source code' / 'databases'

    # Generate missing directories
    if not db_dir.is_dir():
        db_dir.mkdir(parents=True)

    # Specify the download area range (inclusive)
    gravity_lat_range = -60, 60         # Must be ascending order
    gravity_lon_range = -180, 180       # Must be ascending order

    # # Specify the download time range (inclusive)
    gps_start_date = datetime(2019, 1, 1)
    gps_end_date = datetime(2025, 3, 1)

    # Download the geosat-44 gravity map (if missing)
    download_geosat_gravity(db_dir)

    # Download the marine gravity map (if missing)
    download_marine_gravity(db_dir)

    # Download the GGMPlus gravity maps (if missing)
    download_ggm_plus(db_dir, gravity_lat_range, gravity_lon_range, 'acceleration')
    download_ggm_plus(db_dir, gravity_lat_range, gravity_lon_range, 'disturbance')

    # Download the SRTM2Gravity maps (if missing)
    download_srtm2gravity(db_dir, gravity_lat_range, gravity_lon_range, 'full')
    download_srtm2gravity(db_dir, gravity_lat_range, gravity_lon_range, 'residual')

    # Download the GPS RINEX data files (if missing)
    download_gps_data(db_dir, gps_start_date, gps_end_date)
