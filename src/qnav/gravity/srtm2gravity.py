"""
===============
srtm2gravity.py
===============

:summary:
    SRTM2Gravity gravity map implementations.
    This module provides support for the SRTM2Gravity gravity models.
    Specifically, the full-scale and residual gravity maps.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

import numpy as np

from pathlib import Path

from qnav.gravity.ggm_plus import GGMPlusDist

# The size to expect for SRTM2Gravity tiles:
_TILE_SIZE = [1200, 1200]

class SRTM2GravityFull(GGMPlusDist):
    """
    Implementation of SRTM2Gravity Full-Scale gravity.
    Provides a gravity correction map that uses the SRTM2Gravity Full-Scale
    dataset to provide gravity disturbance values.
    """

    def __str__(self) -> str:
        return 'SRTM2Gravity (Full-Scale)'

    def _get_tile(self, origin_lat: int, origin_lon: int) -> np.ndarray:

        local_path = self._get_local_path(origin_lat, origin_lon)
        if not local_path.is_file() and self._auto_download:
            self.download_data(origin_lat, origin_lon)

        if not local_path.is_file():
            return get_empty_srtm_tile()

        data = read_srtm_file(local_path)
        data = np.nan_to_num(data)
        return data

    def _get_sub_dir(self, lat: float, lon: float) -> str:

        # Required
        lat = float(lat)
        lon = float(lon)

        # Get the direction characters
        lat_char = ('N', 'S')[lat < 0]
        lon_char = ('E', 'W')[lon < 0]

        sub_size = 30
        lat_origin = abs(int(sub_size * (lat // sub_size)))
        lon_origin = abs(int(sub_size * (lon // sub_size)))

        # Format the file name with origin coordinates
        return f'{lat_char}{lat_origin:02d}{lon_char}{lon_origin:03d}'

    def _get_local_path(self, lat: float, lon: float) -> Path:
        tile_name = self._get_tile_file(lat, lon)
        tile_sub_dir = self._get_sub_dir(lat, lon)
        return self._map_dir / 'full_scale' / tile_sub_dir / tile_name

    def _get_download_url(self, lat: float, lon: float) -> str:
        tile_name = self._get_tile_file(lat, lon)
        tile_sub_dir = self._get_sub_dir(lat, lon)
        return f'{self._download_url}/{tile_sub_dir}/{tile_name}'

    # # TODO: REMOVE AFTER
    # def _are_points_loaded(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    #     lat_check = (lat >= (self._min_lon + 1)) & (lat <= (self._max_lat - 1))
    #     lon_check = (lon >= (self._min_lon) & (lon <= self._max_lon)
    #     return lat_check & lon_check

    @property
    def _file_suffix(self) -> str:
        return '_full'

    @property
    def _file_ext(self) -> str:
        """
        The file extension of the GGMPlus gravity acceleration data.

        :return: The file extension of data files used.
        :rtype: str
        """
        return 'bin'

    @property
    def _download_url(self) -> str:
        """
        The base URL used for downloading GGMPlus acceleration data.

        :return: The URL where GGMPlus gravity acceleration data can be
            downloaded from.
        :rtype: str
        """
        return 'https://ddfe.blazejbucha.com/models/SRTM2gravity2018/data/FullScaleGravity/'

    @property
    def _tile_size(self) -> float:
        """
        Returns the dimensions of tiles in decimal degrees.
        This is the length of each square tile for latitude and longitude.

        :return: The size of each time in decimal degrees.
        :rtype: float
        """
        return 1

    @property
    def _step_size(self) -> float:
        """
        Returns the step size between cell in decimal degrees.
        This is the distance between records in latitude and longitude
        directions.

        :return: The step size distance between gravity map grid cells.
        :rtype: float
        """
        return 3 / 3600


class SRTM2GravityRes(SRTM2GravityFull):

    def __str__(self) -> str:
        return 'SRTM2Gravity (Residual)'

    def _get_local_path(self, lat: float, lon: float) -> Path:
        tile_name = self._get_tile_file(lat, lon)
        tile_sub_dir = self._get_sub_dir(lat, lon)
        return self._map_dir / 'residual' / tile_sub_dir / tile_name

    @property
    def _file_suffix(self) -> str:
        return '_res'

    @property
    def _download_url(self) -> str:
        """
        The base URL used for downloading GGMPlus acceleration data.

        :return: The URL where GGMPlus gravity acceleration data can be
            downloaded from.
        :rtype: str
        """
        return 'https://ddfe.blazejbucha.com/models/SRTM2gravity2018/data/ResidualGravity/'


def read_srtm_file(file_path: Path) -> np.ndarray:
    """
    Reads the data from a given SRTM2Gravity file. Once read, the returned
    1200-by-1200 matrix data will be in :math:`metres/second^2`. No latitude
    or longitude coordinates are contained within any SRTM2Gravity file.

    :param file_path: The location of the GGM_plus file to read.
    :type file_path: Path or str

    :return: The gravity disturbance matrix read from the file,
        measured in :math:`metres/second^2`
    :rtype: NumPy Matrix (1200-by-1200 elements)
    """

    # The data format of the file.
    data_type = '>i4'  # (32-bit signed-integers big-endian)

    # Read the srtm2gravity data from the provided file.
    return_data = np.fromfile(file_path, dtype=data_type)

    # Reformat the data to mGals and restructure the read vector into a
    # 1200-by-1200 matrix (rows latitude, columns longitude).
    return_data = return_data.astype(np.float64) * 1e-7
    return_data = np.reshape(return_data, _TILE_SIZE).T

    # Finally, return the data.
    return return_data


def get_empty_srtm_tile() -> np.ndarray:
    """
    Produces an empty tile of dimensions of GGM Plus data file.
    Used when data is unavailable and cannot be downloaded.

    :return: Empty GGM Plus tile data for missing files.
    :rtype: NumPy Matrix (1200-by-1200 elements)
    """
    return np.zeros(_TILE_SIZE) + np.nan


if __name__ == "__main__":

    from qnav.gravity.somigliana import Somigliana
    # from qnav.earth.geoid import GeoidPGM

    project_dir = Path(__file__).parent.parent.parent
    grav_dir = project_dir / 'databases' / 'gravity' / 'srtm2gravity'
    som = Somigliana()

    # geoid_dir = project_dir / 'databases' / 'geoid' / 'egm2008-1.pgm'
    # test_geoid = GeoidPGM(geoid_dir)
    #
    # srtm_full = SRTM2GravityFull(som, test_geoid, grav_dir)
    #
    # # g0 = srtm_full.calc_gravity_z(52, -3, 1000)
    # # g1 = srtm_full.calc_gravity_xyz(52, -3, 1000)
    # g2 = srtm_full.calc_vertical_grad(52, -3, 1000)

    srtm_full = SRTM2GravityFull(som, None, grav_dir)
    # test_g_acc = srtm_full.calc_gravity_z(52, -3, 1000)

    srtm_res = SRTM2GravityRes(som, None, grav_dir)
    # test_g_dist = srtm_res.calc_gravity_z(52, -3, 1000)

    for i_lat in range(50, 59):
        for i_lon in range(-12, 2):

            print(f'Downloading ({i_lat}, {i_lon})')
            srtm_full.download_data(i_lat, i_lon)
            srtm_res.download_data(i_lat, i_lon)

