"""
=========
geosat.py
=========

:summary:
    Geosat-44 reading, interpolation and gravity functions.
    This module contains all the gravity calculation functionalities related
    to the geosat-44 dataset. The Geosat44 database contains records of
    time-offsets, positions, geoid heights, gravity anomalies and
    calculation uncertainties.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""
from typing import Optional

import numpy as np

from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

from qnav.earth.geoid import GeoidModel
from qnav.gravity.map import GravityMap
from qnav.gravity.base import GravityModel
from qnav.util.connections import download_file


class Geosat44(GravityMap):
    """
    Gravity Map implementation for the Geosat-44 dataset.
    Provides corrections for a given base gravity function, using the
    non-gridded geosat data values. Performs linear interpolation by
    producing a Delaunay triangulation mesh with fixed zero value for
    extrapolation.
    """

    # The full URL where required geosat resources are located.
    __DOWNLOAD_URL = 'https://www.ngdc.noaa.gov/mgg/gravity/1999/data/global/geosat44/'

    # The required gravity files.
    __MAP_FILES = 'geo44asc.bin', 'geo44des.bin'

    # The value interpolation instance.
    _interp = None

    def __init__(self, base_model: GravityModel, geoid_model: Optional[GeoidModel],
                  map_dir: Path, auto_download: bool = True, auto_load: bool = True):
        """
        Generates a Geosat44 instance to apply gravity corrections.
        Creates a Geosat44 gravity model using a given base gravity
        function, directory to store geosat data and (optionally)
        the ability to download data on demand.

        :param base_model: The base gravity model to correct.
        :type base_model: GravityModel

        :param geoid_model: The geoid model to use for underlation lookup.
        :type geoid_model: GeoidModel

        :param map_dir: The local path for geosat database files.
        :type map_dir: Path

        :param auto_download: (Optional) Allow automatic downloading of files.
            Required for first time use. By default is set to True.
        :type auto_download: bool

        :param auto_load: (Optional) Automatically load gravity map files.
            Required for normal usage without firstly calling load_data().
            By default is set to True.
        :type auto_load: bool
        """
        super().__init__(base_model, geoid_model, map_dir, auto_download)
        if auto_load:
            self.load_data()

    def __str__(self) -> str:
        """
        Returns an identifiable descriptive name for the gravity model.

        :return: A descriptive string for the gravity model.
        :rtype: str
        """
        return "Geosat44"

    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:
        grav_anom = self.get_map_value(lat, lon)
        g = self._base_model.calc_gravity_z(lat, lon, alt)
        g += self._interp_anomaly(lat, lon, alt, grav_anom)
        return g

    def calc_gravity_z_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        grav_anom = self.get_map_value(lat, lon)
        g = self._base_model.calc_gravity_z_vec(lat, lon, alt)
        g += self._interp_anomaly_vec(lat, lon, alt, grav_anom)
        return g

    def get_map_value(self, lat: float | np.ndarray, lon: float | np.ndarray) -> float | np.ndarray:
        """
        Interpolates the gravity anomaly for requested position(s).
        Looks-up and returns the gravity anomaly value for requested latitude
        and longitude positions. Supports both scalar and array input.

        :param lat: The latitude coordinate(s) in decimal degrees.
        :type lat: float | np.ndarray

        :param lon: The longitude coordinate(s) in decimal degrees.
        :type lon: float | np.ndarray

        :return: The interpolated gravity anomaly value(s) in m/s^2.
        :rtype: float | np.ndarray
        """

        anom_values = self._interp(lat, lon)
        if np.isscalar(lat) and np.isscalar(lon):
            anom_values = float(anom_values)
        return anom_values

    def get_anomaly(self, lat: float, lon: float) -> float:
        return self.get_map_value(lat, lon)

    def get_anomaly_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.get_map_value(lat, lon)

    def get_disturbance(self, lat: float, lon: float) -> float:
        anomaly = self.get_anomaly(lat, lon)
        return self._anomaly_to_disturbance(lat, lon, anomaly)

    def get_disturbance_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        anomaly = self.get_anomaly_vec(lat, lon)
        return self._anomaly_to_disturbance_vec(lat, lon, anomaly)

    def load_data(self):
        """
        Loads the Geosat data and generates internal interpolator for look-up.
        Called on initialisation to load the required data to allow for
        correction interpolation. Due to spacing between geosat records,
        linear interpolation is used for stability.
        """

        # Download missing data if possible
        self.download_data()

        all_data: dict[str, np.ndarray] = {}

        # For each geosat data file to use:
        for file_name in self.__MAP_FILES:
            file_path = self._map_dir / file_name

            # Raise error if file not found:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f'missing Geosat file {file_name}')

            # Append current file to data set:
            data = read_geosat_file(file_path)
            for k, v in data.items():
                current_values = all_data.get(k, [])
                all_data[k] = np.append(current_values, v)

        # Produce a look-up grid for all data
        points = np.column_stack((all_data['latitude'], all_data['longitude']))
        values = all_data['gravity_anomaly']
        tri = Delaunay(points)

        # Generate an interpolation instance for the geosat data.
        self._interp = LinearNDInterpolator(
            tri, values, fill_value=0.0, rescale=False)

    def download_data(self, overwrite: bool = False):
        """
        Downloads the required Geosat-44 files from the source.
        If allowed and files are not present, will download the required
        geosat files to the set directory. Optionally, files can be
        overwritten.

        :param overwrite: Should downloading replace existing files.
            By default this is set to false and only missing files
            are downloaded.
        :type overwrite: bool
        """

        # Prevent download if not allowed:
        if not self._auto_download:
            return

        # Generate required parent directories:
        if not self._map_dir.exists():
            self._map_dir.mkdir(parents=True)

        # For each of the required files:
        for file_name in self.__MAP_FILES:
            local_path = self._map_dir / file_name

            # Download if requested or missing from directory:
            if overwrite or not local_path.is_file():
                download_url = f'{self.__DOWNLOAD_URL}{file_name}'
                download_file(download_url, local_path)


def read_geosat_file(file_path: Path) -> dict[str, np.ndarray]:
    """
    Reads and extracts the content of a given geosat file.
    Parses a given binary (.bin) geosat file and returns its records in a
    Python dictionary. The records include:
    • time_offset:  The time since the beginning of each pass (seconds)
    • latitude:     The latitude position of measurement (degrees)
    • longitude:    The longitude position of measurement (degrees)
    • geoid:        The geoid height from the ellipse (metres)
    • anomaly:      The calculated gravity anomaly (m/s^2)
    • uncertainty:  The measurement uncertainty (m/s^2)

    :param file_path: The path for the binary geosat file to read.
    :type file_path: Path

    :return: A dictionary containing the read data records.
    :rtype: dict[str, np.ndarray]
    """

    # The datatypes in each record of the geosat file.
    dt = [
        ('time', '<i2'),
        ('latitude', '<i4'),
        ('longitude', '<i4'),
        ('geoid', '<i2'),
        ('anomaly', '<i2'),
        ('uncertainty', '<i2')
    ]

    # Read the entirety of the given file
    table_data = np.fromfile(file_path, dtype=dt, count=-1)

    # Convert the table data into a 64-bit float matrix.
    float_data = np.array(table_data.tolist(), dtype=np.float64)

    # Convert from matrix from integer values to floats
    float_data[:, 0] *= 0.489  # Time offset (seconds)
    float_data[:, 1] /= 1e3  # Latitude (degrees)
    float_data[:, 2] /= 1e3  # Longitude (degrees)
    float_data[:, 3] /= 1e2  # Geoid height (metres)
    # float_data[:, 4] /= 1e6  # Gravity anomaly (m/s^-2)
    # float_data[:, 5] /= 1e6  # Gravity uncertainty (m/s^-2)

    float_data[:, 4] /= 1e7  # Gravity anomaly (m/s^-2)
    float_data[:, 5] /= 1e7  # Gravity uncertainty (m/s^-2)

    # Fix the longitude bounds (from 0:360 to -180:180).
    float_data[float_data[:, 2] > 180, 2] -= 360

    # Finally, return the data in the form of a struct.
    return {
        'time_offset': float_data[:, 0],
        'latitude': float_data[:, 1],
        'longitude': float_data[:, 2],
        'geoid_height': float_data[:, 3],
        'gravity_anomaly': float_data[:, 4],
        'gravity_uncertainty': float_data[:, 5]
    }


def create_interp_grid(geosat_data: dict[str, np.ndarray],
                       interp_method: str = 'linear',
                       resolution: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates interpolated grid points for geosat data.
    For given geosat data, a set of grid points are generated using the
    requested interpolation method and resolution. This is provided to
    produced uniformly gridded maps for faster look-up and interpolation.

    :param geosat_data: The contents read from the geosat file(s).
    :type geosat_data: dict[str, np.ndarray]

    :param interp_method: (Optional) The interpolation method to use.
        Supported methods include 'nearest', 'linear' and 'cubic'.
        By default, the 'linear' method is used.
    :type interp_method: str

    :param resolution: (Optional) The requested step-size for the
        interpolation in decimal degrees. By default 0.1 is used.
    :type resolution:

    :return: A tuple containing the produced latitude, longitude and gravity
        anomaly values, respectively.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    """

    # Extract the dataset values and positions.
    lat_points = geosat_data["latitude"]
    lon_points = geosat_data["longitude"]
    anon_values = geosat_data["gravity_anomaly"]

    # Obtain the grid positions to calculate values for.
    query_lon_points = np.arange(-180, 180, resolution)
    query_lat_points = np.arange(-90, 90, resolution)

    # Convert the query points into equal sized grids.
    lon_grid, lat_grid = np.meshgrid(query_lon_points, query_lat_points)

    # Perform grid data interpolation using the selected method.
    anom_grid = griddata((lon_points, lat_points),
                         anon_values, (lon_grid, lat_grid),
                         method=interp_method, fill_value=0.0)

    # Finally, return the produced data.
    return lat_grid, lon_grid, anom_grid

