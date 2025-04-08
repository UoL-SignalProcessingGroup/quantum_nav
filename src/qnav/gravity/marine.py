"""
=========
marine.py
=========

:summary:
    Marine Gravity Map reading, interpolation and gravity functions.
    This module contains all the gravity calculation functionalities related
    to the marine gravity dataset. The Marine Gravity Map database contains
    records of positions and free-air gravity anomalies, uncertainties in
    anomalies and vertical gravity gradients. Data was last updated
    August 2022.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""


import h5py
import numpy as np

from pathlib import Path
from typing import Optional
from scipy.interpolate import RegularGridInterpolator

from qnav.earth.geoid import GeoidModel
from qnav.gravity.base import GravityModel
from qnav.gravity.map import GravityMap
from qnav.util.connections import download_file


class MarineGravity(GravityMap):
    """
    Gravity Map implementation for the Marine Gravity dataset.
    Provides corrections for a given base gravity function, using the
    marine gravity map anomaly values.
    """

    # The full URL where required geosat resources are located.
    __DOWNLOAD_URL = 'https://topex.ucsd.edu/pub/global_grav_1min/'

    # The required gravity files.
    __MAP_FILE = 'curv_32.1.nc'

    # The interpolation method to use.
    _interp_method = 'linear'

    # The value interpolation instance.
    _interp = None

    def __init__(self, base_model: GravityModel,
                 geoid_model: Optional[GeoidModel],
                 map_dir: Path, auto_download: bool = True, auto_load: bool = True):
        """
        Generates a Marine Gravity Map instance to apply gravity corrections.
        Creates a Marine Gravity model using a given base gravity
        function, geoid model and directory to store data and
        (optionally) the ability to download data on demand.

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
        return 'Marine Gravity Map'

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
        anom_values = self._interp((lat, lon))
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
        Loads the Marine Gravity data and generates interpolator for look-up.
        Called on initialisation to load the required data to allow for
        correction interpolation.
        """

        # Download missing data if possible
        self.download_data()

        # For each geosat data file to use:
        file_path = self._map_dir / self.__MAP_FILE

        if not file_path.is_file():
            raise FileNotFoundError(
                f'missing Marine Gravity file {self.__MAP_FILE}')

        data = read_marine_file(file_path)
        self._interp = RegularGridInterpolator(
            (data['latitude'], data['longitude']), data['anomaly'],
            method=self._interp_method, fill_value=0.0, bounds_error=False)


    def download_data(self, overwrite: bool = False):
        """
        Downloads the required marine gravity files from the source.
        If allowed and files are not present, will download the required
        geosat files to the set directory. Optionally, files can be
        overwritten.

        :param overwrite: Should downloading replace existing files.
            By default this is set to false and only missing files
            are downloaded.
        :type overwrite: bool
        """

        # Prevent download if not allowed:
        if self._auto_download:

            # Generate required parent directories:
            if not self._map_dir.exists():
                self._map_dir.mkdir(parents=True)

            # Download if requested or missing from directory:
            local_path = self._map_dir / self.__MAP_FILE
            if overwrite or not local_path.is_file():
                download_url = f'{self.__DOWNLOAD_URL}{self.__MAP_FILE}'
                download_file(download_url, local_path)


def read_marine_file(file_path: Path) -> dict[str, np.ndarray]:
    """
    This function extracts the data from a Marine gravity dataset. Marine
    gravity files are encoded in HDF5 scientific data format. Once
    successfully read, the gravity anomaly matrix, along with the latitude
    and longitude positions will be returned. The gravity values are in
    metres/seconds^2.

    :param file_path: The location of the Marine gravity dataset to read.
    :type file_path: Path

    :return: A dictionary containing the marine gravity map data, consisting
        of gravity anomaly, latitude and longitude values.
    :rtype: dict[str, np.ndarray]
    """

    # Attempt to open the given HDF5 file:
    with h5py.File(file_path, "r") as grav_file:

        # Extract the anomaly matrix, latitude and longitude values
        lat_points = grav_file['lat'][()].astype(np.float64)
        lon_points = grav_file['lon'][()].astype(np.float64)
        grav_points = grav_file['z'][()].astype(np.float64) / 1e5

    # Return the extracted data as numpy arrays
    return {
        'anomaly': grav_points,
        'latitude': lat_points,
        'longitude': lon_points
    }


if __name__ == '__main__':

    from qnav.gravity.somigliana import Somigliana

    project_dir = Path(__file__).parent.parent.parent
    grav_dir = project_dir / 'databases' / 'gravity' / 'marine'
    som = Somigliana()

    marine = MarineGravity(som, None, grav_dir)
    test_g_acc = marine.calc_gravity_z(52, -3, 1000)

