"""
===========
ggm_plus.py
===========

:summary:
    Global Gravity Map Plus (GGMPlus) gravity map implementations.
    This module provides support for the GGMPlus gravity models. Specifically,
    two implementations for the GGPlus gravity acceleration and gravity
    disturbance corrections.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

import math
import numpy as np

from pathlib import Path
from typing import Optional

from numpy import ndarray
from requests import HTTPError
from scipy.interpolate import RegularGridInterpolator

from qnav.earth.geoid import GeoidModel
from qnav.gravity.map import GravityMap
from qnav.gravity.base import GravityModel
from qnav.util.connections import download_file

# The size to expect for GGM Plus tiles:
_TILE_SIZE = [2500, 2500]


class GGMPlusAcc(GravityMap):
    """
    Global Gravity Map Plus for Acceleration Correction.
    An implementation of GGMPlus, using the provided base gravity
    acceleration values. This overwrites the base values zero-altitude
    gravity estimation, but uses its altitude extrapolation.
    """

    # The current minimum latitude of the region of loaded data.
    _min_lat: float = float("nan")

    # The current minimum latitude of the region of loaded data.
    _max_lat: float = float("nan")

    # The current minimum latitude of the region of loaded data.
    _min_lon: float = float("nan")

    # The current minimum latitude of the region of loaded data.
    _max_lon: float = float("nan")

    # The interpolation method to use (must handle NaN values).
    _interp_method: str = 'linear'

    # The gravity correction interpolator for value look-up.
    _interp: RegularGridInterpolator

    def __init__(self, base_model: GravityModel,
                 geoid_model: Optional[GeoidModel], map_dir: Path,
                 auto_download: bool = True, margin_size: int = 1) -> None:

        # Call parent constructor and all record requested margin size
        super().__init__(base_model, geoid_model, map_dir, auto_download)
        if margin_size < 1: raise ValueError("margin_size must be >= 1")
        self._margin_size = margin_size

    def __str__(self) -> str:
        return 'GGMPlus (Acceleration)'

    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:

        # TODO: Check implementation
        base_acc = self.get_map_value(lat, lon)
        grav_0 = self._base_model.calc_gravity_z(lat, lon, 0)
        grav_h = self._base_model.calc_gravity_z(lat, lon, alt)

        if math.isnan(base_acc):
            return grav_h

        return base_acc + (grav_h - grav_0)

    def calc_gravity_z_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:

        # TODO: Check implementation
        base_acc = self.get_map_values(lat, lon)
        z = np.zeros_like(alt)

        grav_0 = self._base_model.calc_gravity_z_vec(lat, lon, z)
        grav_h = self._base_model.calc_gravity_z_vec(lat, lon, alt)

        to_return = base_acc + (grav_h - grav_0)
        nan_indices = np.isnan(base_acc)
        to_return[nan_indices] = grav_h[nan_indices]
        return to_return

    def get_anomaly(self, lat: float, lon: float) -> float:
        disturbance = self.get_disturbance(lat, lon)
        return self._disturbance_to_anomaly(lat, lon, disturbance)

    def get_anomaly_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        disturbance = self.get_disturbance_vec(lat, lon)
        return self._disturbance_to_anomaly_vec(lat, lon, disturbance)

    def get_disturbance(self, lat: float, lon: float) -> float:
        g_lambda = self.base_model.calc_gravity_z(lat, lon, 0)
        g_model = self.get_map_value(lat, lon)
        if math.isnan(g_model):
            return 0
        return g_model - g_lambda

    def get_disturbance_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        # return self.get_map_values(lat, lon)
        alt = np.zeros_like(lat)
        g_lambda = self.base_model.calc_gravity_z_vec(lat, lon, alt)
        g_model = self.get_map_values(lat, lon)
        to_return =  g_model - g_lambda
        return np.nan_to_num(to_return)

    def get_map_value(self, lat: float, lon: float) -> float:
        """
        Interpolates the gravity acceleration for requested position(.
        Looks-up and returns the gravity anomaly value for requested latitude
        and longitude position.

        :param lat: The latitude coordinate in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate in decimal degrees.
        :type lon: float

        :return: The interpolated gravity map value in m/s^2.
        :rtype: float
        """

        # If required, load the missing data:
        if not self._is_point_loaded(lat, lon):
            self._load_data(lat, lon)

        # Result the interpolated result:
        return float(self._interp((lat, lon)))


    def get_map_values(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Looks-up the correction values using the map.
        This will dynamically load data to cover the range, keeping the
        maximum number of loaded tiles under the margin value, to conserve
        memory constraints.

        :param lat: The latitude coordinate in decimal degrees.
        :type lat: np.ndarray

        :param lon: The longitude coordinate in decimal degrees.
        :type lon: np.ndarray

        :return: The corresponding values interpolated from the map.
        :rtype: np.ndarray
        """

        # Validate shape of input:
        if lat.shape != lon.shape:
            ValueError("lat, lon and alt must have same shape")

        # Record collected results and
        has_result = np.zeros(lat.shape, dtype=bool)
        results = np.zeros(lat.shape, dtype=float)

        # Until completion
        while True:

            # Get values for current loaded region:
            i = self._are_points_loaded(lat, lon)
            if np.any(i):
                results[i] = self._interp((lat[i], lon[i]))
                has_result[i] = True

            # Stop if all results are collected
            if np.all(has_result):
                return results

            # Otherwise, load next region and repeat
            j = np.argwhere(~has_result)[0]
            self._load_data(lat[*j], lon[*j])

    def download_data(self, origin_lat: int, origin_lon: int, overwrite: bool = False):
        """
        Downloads the required marine gravity files from the source.
        If allowed and files are not present, will download the required
        geosat files to the set directory. Optionally, files can be
        overwritten.

        :param origin_lat: Origin latitude of tile to download.
        :type origin_lat: int

        :param origin_lon: Origin longitude of tile to download.
        :type origin_lon: int

        :param overwrite: Should downloading replace existing files.
            By default this is set to false and only missing files
            are downloaded.
        :type overwrite: bool
        """

        local_path = self._get_local_path(origin_lat, origin_lon)

        if not local_path.parent.exists():
            local_path.parent.mkdir(parents=True)

        if overwrite or not local_path.is_file():
            download_url = self._get_download_url(
                origin_lat, origin_lon)

            try:
                download_file(download_url, local_path)
            except HTTPError:
                pass  # TODO: Improve error handling

    def _is_point_loaded(self, lat: float, lon: float) -> bool:
        lat_check = self._min_lat <= lat <= self._max_lat
        lon_check = self._min_lon <= lon <= self._max_lon
        return lat_check and lon_check

    def _are_points_loaded(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        lat_check = (lat >= self._min_lat) & (lat <= self._max_lat)
        lon_check = (lon >= self._min_lon) & (lon <= self._max_lon)
        return lat_check & lon_check

    def _load_data(self, lat: float, lon: float):

        # Unpack properties
        tile_size = self._tile_size
        tile_margin = self._margin_size

        # Get the origins of the tiles to acquire
        tile_range = np.arange(-tile_margin, tile_margin + 1) * tile_size
        [centre_lat, centre_lon] = self._get_origin(lat, lon)
        lat_origins = tile_range + centre_lat
        lon_origins = tile_range + centre_lon

        # Obtain the position grid
        centre_offset = self._step_size / 2
        self._min_lat = float(np.min(lat_origins)) + centre_offset
        self._min_lon = float(np.min(lon_origins)) + centre_offset
        self._max_lat = float(np.max(lat_origins)) - centre_offset + tile_size
        self._max_lon = float(np.max(lon_origins)) - centre_offset + tile_size

        # Generate grids for each latitude and longitude coordinate
        lat_ticks = np.arange(self._min_lat, self._max_lat + 1e-10, self._step_size)
        lon_ticks = np.arange(self._min_lon, self._max_lon + 1e-10, self._step_size)
        all_tiles = []

        for tile_lat in lat_origins:
            lon_tiles = []

            for tile_lon in lon_origins:
                tile_data = self._get_tile(tile_lat, tile_lon)
                lon_tiles.append(tile_data)

            # Append list to full tiles
            all_tiles.append(lon_tiles)

        # Merge all tiles into single array
        tile_data = np.block(all_tiles)

        # Generate an interpolator for collected data:
        self._interp = RegularGridInterpolator(
            (lat_ticks, lon_ticks), tile_data,
            method=self._interp_method,
            fill_value=self._fill_value,
            bounds_error=False)

    def _get_tile(self, origin_lat: int, origin_lon: int) -> np.ndarray:

        local_path = self._get_local_path(origin_lat, origin_lon)
        if not local_path.is_file() and self._auto_download:
            self.download_data(origin_lat, origin_lon)

        if not local_path.is_file():
            return get_empty_ggm_tile()

        return read_ggm_file(local_path)

    def _get_origin(self, lat: float | int, lon: float | int) -> tuple[int, int]:

        ts = self._tile_size
        lat_origin = ts * (lat // ts)
        lon_origin = ts * (lon // ts)
        return int(lat_origin), int(lon_origin)

    def _get_tile_file(self, lat: float, lon: float) -> str:

        # Get origin coordinates for the relevant file
        lat_origin, lon_origin = self._get_origin(lat, lon)

        # Get the direction characters
        lat_char = ('N', 'S')[lat_origin < 0]
        lon_char = ('E', 'W')[lon_origin < 0]

        # Remove sign of coordinates
        lat_origin = abs(lat_origin)
        lon_origin = abs(lon_origin)

        # Format the file name with origin coordinates
        file_name = (f'{lat_char}{lat_origin:02d}'
                     f'{lon_char}{lon_origin:03d}'
                     f'{self._file_suffix}')

        # Finally, return teh full file path
        return f'{file_name}.{self._file_ext}'

    def _get_local_path(self, lat: float, lon: float) -> Path:
        tile_name = self._get_tile_file(lat, lon)
        return self._map_dir / 'acc' / tile_name

    def _get_download_url(self, lat: float, lon: float) -> str:
        tile_name = self._get_tile_file(lat, lon)
        return f'{self._download_url}{tile_name}'

    @property
    def _file_suffix(self) -> str:
        return ''

    @property
    def _file_ext(self) -> str:
        """
        The file extension of the GGMPlus gravity acceleration data.

        :return: The file extension of data files used.
        :rtype: str
        """
        return 'ga'

    @property
    def _download_url(self) -> str:
        """
        The base URL used for downloading GGMPlus acceleration data.

        :return: The URL where GGMPlus gravity acceleration data can be
            downloaded from.
        :rtype: str
        """
        return 'https://ddfe.blazejbucha.com/models/GGMplus/data/ga/'

    @property
    def _tile_size(self) -> float:
        """
        Returns the dimensions of tiles in decimal degrees.
        This is the length of each square tile for latitude and longitude.

        :return: The size of each time in decimal degrees.
        :rtype: float
        """
        return 5

    @property
    def _step_size(self) -> float:
        """
        Returns the step size between cell in decimal degrees.
        This is the distance between records in latitude and longitude
        directions.

        :return: The step size distance between gravity map grid cells.
        :rtype: float
        """
        return 0.002

    @property
    def _fill_value(self) -> float:
        """
        Returns the fill value to use for missing value.
        This will replace 'NaN' values for interpolation and extrapolation.

        :return: Returns NaN.
        :rtype: float
        """
        return float('nan')


class GGMPlusDist(GGMPlusAcc):
    """
    Global Gravity Map Plus for Gravity Disturbance Correction.
    An implementation of GGMPlus, using the acquired gravity disturbance
    values values.
    """

    def __str__(self) -> str:
        return 'GGMPlus (Disturbance)'

    def get_map_value(self, lat: float, lon: float) -> float:
        value = super().get_map_value(lat, lon)
        if math.isnan(value):
            return 0
        return value

    def get_map_values(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        values = super().get_map_values(lat, lon)
        return np.nan_to_num(values)

    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:

        # TODO: Check implementation
        grav_dist = self.get_map_value(lat, lon)
        grav_0 = self._base_model.calc_gravity_z(lat, lon, 0)
        grav_h = self._base_model.calc_gravity_z(lat, lon, alt)
        return grav_0 + grav_dist + (grav_h - grav_0)


    def calc_gravity_z_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:

        # TODO: Check implementation
        grav_dist = self.get_map_values(lat, lon)
        z = np.zeros_like(alt)

        grav_0 = self._base_model.calc_gravity_z_vec(lat, lon, z)
        grav_h = self._base_model.calc_gravity_z_vec(lat, lon, alt)
        return grav_0 + grav_dist + (grav_h - grav_0)

    def _get_tile(self, origin_lat: int, origin_lon: int) -> np.ndarray:
        to_return = super()._get_tile(origin_lat, origin_lon)
        return np.nan_to_num(to_return)

    def _get_local_path(self, lat: float, lon: float) -> Path:
        tile_name = self._get_tile_file(lat, lon)
        return self._map_dir / 'dist' / tile_name

    def get_disturbance(self, lat: float, lon: float) -> float:
        return self.get_map_value(lat, lon)

    def get_disturbance_vec(self, lat: np.ndarray, lon: np.ndarray) -> ndarray:
        return self.get_map_values(lat, lon)

    @property
    def _file_ext(self) -> str:
        """
        The file extension of the GGMPlus gravity acceleration data.

        :return: The file extension of data files used.
        :rtype: str
        """
        return 'dg'

    @property
    def _download_url(self) -> str:
        """
        The base URL used for downloading GGMPlus acceleration data.

        :return: The URL where GGMPlus gravity acceleration data can be
            downloaded from.
        :rtype: str
        """
        return 'https://ddfe.blazejbucha.com/models/GGMplus/data/dg/'


def read_ggm_file(file_path: Path) -> np.ndarray:
    """
    Reads the data from a given GGM_plus file. Once read, the returned
    2500-by-2500 matrix data will be in :math:`metres/second^2`. No latitude
    or longitude coordinates are contained within any GGM_plus file.

    :param file_path: The location of the GGM_plus file to read.
    :type file_path: Path or str

    :return: The gravitational acceleration matrix read from the file,
        measured in :math:`metres/second^2`
    :rtype: NumPy Matrix (2500-by-2500 elements)
    """

    # Get the expected format of the data:
    if file_path.suffix in ['.ga', '.ha']:
        data_type = np.dtype('>i4')

    elif file_path.suffix in ['.eta', '.xi', '.dg']:
        data_type = np.dtype('>i2')

    else:
        raise ValueError(f'Unrecognised file extension: {file_path.suffix}')

    # Read the contents from the selected file
    data = np.fromfile(file_path, dtype=data_type)

    # Convert from integer to float values
    data = data.astype(np.float64)

    # Correct NaN values.
    nan_value = np.iinfo(data_type).min
    data[data <= nan_value] = np.nan

    # Reshape the data and convert from mGals to m/s^2
    data = np.reshape(data, _TILE_SIZE) / 1e6

    # Finally, return the read data
    return data.T

def get_empty_ggm_tile() -> np.ndarray:
    """
    Produces an empty tile of dimensions of GGM Plus data file.
    Used when data is unavailable and cannot be downloaded.

    :return: Empty GGM Plus tile data for missing files.
    :rtype: NumPy Matrix (2500-by-2500 elements)
    """
    return np.zeros(_TILE_SIZE) + np.nan


if __name__ == "__main__":

    from qnav.gravity.somigliana import Somigliana

    project_dir = Path(__file__).parent.parent.parent
    grav_dir = project_dir / 'databases' / 'gravity' / 'ggm_plus'
    som = Somigliana()

    ggm_acc = GGMPlusAcc(som, None, grav_dir)
    # test_g_acc = ggm_acc.calc_gravity_z(52, -3, 1000)

    # ggm_dist = GGMPlusDist(som, None, grav_dir)
    # test_g_dist = ggm_dist.calc_gravity_z(52, -3, 1000)

    # test_lat = np.array([54.0232638, 51.41854768, 53.67120925, 53.20008043, 51.04160022,
    #                      50.90639821, 52.90216386, 52.09873456, 53.12637224, 51.85739463])
    #
    # test_lon = np.array([-3.23375375, -2.36566192, -3.61861449, -2.79436671, -1.46919802,
    #                      -1.34033806, -3.4074206 , -1.86157922, -1.36622413, -2.38345550])
    #
    # test_alt = np.array([1383.08959147, 487.31443131, 6816.4005876 , 8499.10171285, 5806.21782301,
    #                      7367.69684912, 930.88664461, 2299.53236648, 9965.8192528 , 458.22239443])
    #
    # ggm_acc = GGMPlusAcc(som, None, grav_dir)
    # test_g_acc= ggm_acc.calc_gravity_z_vec(test_lat, test_lon, test_alt)

    error_pos = (50.22312480923596, -2.627512029578551, 3415.40078975074)
    test_dist = ggm_acc.get_disturbance(error_pos[0], error_pos[1])
    print(test_dist)
