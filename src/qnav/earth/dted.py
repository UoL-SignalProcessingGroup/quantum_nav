"""
=======
dted.py
=======

:summary:
    A module for managing Digital Terrain Elevation Data (DTED).
    This is responsible for the reading and dynamic loading of DTED files for
    extracting and interpolating elevation from a locally stored Digital
    Elevation Map (DEM). This allows the extraction of the terrain height
    above the WSG84 ellipsoid, excluding man-made features. This is namely
    used for waypoint/trajectory generation of man-made features.

:authors:
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of this module.
"""

import qnav.util.transformations as trans
import numpy.typing as npt
import numpy as np
import math

from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


class DtedTile:
    """
    Holds and handles the read content for a single DTED file.
    Manages the data contained in a DTED tile (1-degree by 1-degree region).
    This allows local interpolation and other relevant calculations. Instances
    of this are used by :class:`~dted.DtedHandler` for dynamic loading and
    management of terrain information.
    """

    __slots__ = \
        '__origin_lat', \
        '__origin_lon', \
        '__centre_lat', \
        '__centre_lon', \
        '__dted_version', \
        '__lat_ticks', \
        '__lon_ticks', \
        '__dted_elevation', \
        '__dted_interp'

    def __init__(self, origin_lat: int, origin_lon: int,
                 dted_dir: Path,
                 dted_version: int = 2,
                 interp_method: str = 'linear',
                 is_empty_on_error: bool = True):
        """
        Generates DTED tile holding corresponding DTED information.
        On creation this will attempt to load the corresponding DTED
        information covered by the region. The methods supplied by this class
        can then be used to extract the relevant data. By default, if the
        corresponding DTED cannot be read, substitute information will be
        used that assumes the terrain to be flat.

        :param origin_lat: The integer latitude value (south-most coordinate).
        :type origin_lat: int

        :param origin_lon: The integer longitude value (west-most coordinate).
        :type origin_lon: int

        :param dted_dir: The local database_dir where DTED file are stored.
        :type dted_dir: Path

        :param dted_version: The version of the DTED data to use (default=2).
        :type dted_version: int

        :param interp_method: The interpolation method to use (default='linear').
            Supported values are: 'linear', 'nearest', 'slinear', 'cubic',
            'quintic' and 'pchip'.
        :type interp_method: str

        :param is_empty_on_error: Should empty/flat terrain be assumed if the
            corresponding DTED file cannot be read (default=True).
        :type is_empty_on_error: bool
        """

        # Get the full path for the DTED file:
        dted_file = get_dted_file_path(
            dted_dir, origin_lat, origin_lon, dted_version)

        # The value to use for points outside the data range.
        fill_value = 0

        # Attempt to read the file contents:
        if dted_file.is_file():
            dted_dict = read_dted_file(dted_file)
            fill_value = np.nan

        elif is_empty_on_error:
            dted_dict = get_template_dted_data(
                origin_lat, origin_lon,
                2, 2)
        else:
            raise FileNotFoundError(dted_file)

        # Extract the read contents:
        self.__origin_lat = dted_dict['lat_origin']
        self.__origin_lon = dted_dict['lon_origin']
        self.__lat_ticks = dted_dict['lat_ticks']
        self.__lon_ticks = dted_dict['lon_ticks']
        self.__dted_elevation = dted_dict['elevation']
        self.__centre_lat = self.__origin_lat + 0.5
        self.__centre_lon = self.__origin_lon + 0.5
        self.__dted_version = dted_version

        # Create an interpolator for the DTED data
        self.__dted_interp = RegularGridInterpolator(
            (self.__lat_ticks, self.__lon_ticks),
            self.__dted_elevation,
            method=interp_method,
            fill_value=fill_value)

    def get_elevation(self, lat: float, lon: float, method: str = None) -> float:
        """
        Interpolates the terrain elevation for the requested position.
        Given a latitude and longitude, the interpolated terrain elevation
        will be return (in metres). Optionally, an overriding interpolation
        method to use can be provided.

        :param lat: The latitude coordinate of the reference position.
        :type lat: float

        :param lon: The longitude coordinate of the reference position.
        :type lon: float

        :param method: An alternative interpolation method to use. Supported
            values are: 'linear', 'nearest', 'slinear', 'cubic',
                'quintic' and 'pchip'.
        :type method: str

        :return: The interpolated terrain elevation (in metres)
        :rtype: float
        """
        xi = (lat, lon)
        yi = self.__dted_interp(xi, method=method)
        return float(np.array(yi))

    def get_elevation_vec(self, lat_points: npt.ArrayLike, lon_points: npt.ArrayLike, method: str = None) -> np.ndarray:
        """
        Interpolates the terrain elevation for the requested positions.
        Given latitude and longitude coordinates, the interpolated terrain
        elevations will be return (in metres). Optionally, an overriding
        interpolation method to use can be provided.

        :param lat_points: The latitude coordinates of the reference positions.
        :type lat_points: npt.ArrayLike

        :param lon_points: The longitude coordinates of the reference positions.
        :type lon_points: npt.ArrayLike

        :param method: An alternative interpolation method to use. Supported
            values are: 'linear', 'nearest', 'slinear', 'cubic',
                'quintic' and 'pchip'.
        :type method: str

        :return: The interpolated terrain elevations (in metres)
        :rtype: npt.ArrayLike
        """
        return self.__dted_interp(
            (lat_points, lon_points), method=method)

    def is_position_covered(self, lat: float, lon: float) -> bool:
        """
        A method for checking if a requested point is covered by the tile.
        Performs a check to see if the position is encompassed by the tile,
        allowing its terrain heights to be interpolated.

        :param lat: The latitude coordinate of the reference position.
        :type lat: float

        :param lon: The longitude coordinate of the reference position.
        :type lon: float

        :return: True if the point is covered by the tile.
        :rtype: bool
        """
        return self.__origin_lat <= lat <= (self.__origin_lat + 1) \
            and self.__origin_lat <= lon <= (self.__origin_lon + 1)

    def is_position_covered_vec(self, lat: npt.ArrayLike, lon: npt.ArrayLike) -> np.ndarray:
        """
        A vectorised method for checking if points are covered by the tile.
        Performs a check to see which positions are encompassed by the tile,
        supporting their terrain heights to be interpolated.

        :param lat: The latitude coordinates of the reference positions.
        :type lat: npt.ArrayLike

        :param lon: The longitude coordinates of the reference positions.
        :type lon: npt.ArrayLike

        :return: A boolean array indicating which points are covered by the
            tile. True is used for points inside the tile.
        :rtype: np.ndarray.
        """
        lat_check = lat >= self.__origin_lat & lat <= (self.__origin_lat + 1)
        lon_check = lon >= self.__origin_lon & lon <= (self.__origin_lon + 1)
        return lat_check & lon_check

    def distance_to_centre(self, lat: float, lon: float) -> float:
        """
        Returns approximation to distance from centre of tile (in metres).
        Calculates the approximate to distance from centre of tile for the
        given position. Namely used to identify the most distant tile.

        :param lat: The latitude coordinate of the reference position.
        :type lat: float

        :param lon: The longitude coordinate of the reference position.
        :type lon: float

        :return: The distance from the centre of the tile (in metres).
        :rtype: float
        """
        centre_lat, centre_lon = self.__centre_lat, self.__centre_lon
        return trans.haversine(lat, lon, 0, centre_lat, centre_lon, 0)


class DtedHandler:
    """
    Manages the dynamic loading and handling of regions of DTED data.
    This class is use dynamically load DTED data for extracting elevation.
    This automatically determines the required DTED files and loads into
    memory. Constraints on the maximum amount of data to load can be set to
    conserve resources.
    """

    # All properties of the class.
    __slots__ = \
        '__max_tiles', \
        '__interp_method', \
        '__dted_tiles', \
        '__dted_version', \
        '__dted_dir'

    def __init__(self, dted_dir: Path, dted_version: int = 2,
                 max_tiles: int = None, method: str = None):
        """
        Generates a DTED handler instance for dynamically managing DTED data.
        Given a path to a local DTED dataset and the DTED version to use, a
        DtedHandler instance will be returned allowing data to be dynamically
        queried.

        :param dted_dir: The location of the local DTED database.
        :type dted_dir: Path

        :param dted_version: The DTED data version to use (default=2).
        :type dted_version: int

        :param max_tiles: An optional maximum number of DTED tiles/file to
            hold in memory at a time (default=None).
        :type max_tiles: int

        :param method: The interpolation method to use. Supported methods:
            'linear', 'nearest', 'slinear', 'cubic', 'quintic' and 'pchip'.
        :type method: str
        """
        self.__dted_tiles = {}
        self.__dted_dir = dted_dir
        self.__dted_version = dted_version
        self.__max_tiles = max_tiles
        self.__interp_method = method

    def get_elevation(self, lat: float, lon: float, method: str = None) -> float:
        """
        Interpolates the terrain elevation for the requested position.
        Performed interpolation of the terrain elevation for the given
        latitude and longitude location. Optionally an overriding
        interpolation method can be used.

        :param lat: The latitude coordinates of the reference positions.
        :type lat: float

        :param lon: The longitude coordinates of the reference positions.
        :type lon: float

        :param method: The overriding interpolation method to use. Supported
            methods are: 'linear', 'nearest', 'slinear', 'cubic',
            'quintic' and 'pchip'.
        :type method: str

        :return: The interpolated terrain elevation (in metres).
        :rtype: float
        """
        dted_tile = self.__get_dted_tile(lat, lon)
        return dted_tile.get_elevation(lat, lon, method)

    def get_elevation_vec(self, lat_points: np.ndarray, lon_points: np.ndarray, method: str = None) -> np.ndarray:
        """
        Interpolates the terrain elevation at given positions.
        Performed vectorised interpolation of the terrain elevation for the
        given latitude and longitude locations. Optionally an overriding
         interpolation method can be used.

        :param lat_points: The latitude coordinates of the reference positions.
        :type lat_points: np.ndarray

        :param lon_points: The longitude coordinates of the reference positions.
        :type lon_points: np.ndarray

        :param method: The overriding interpolation method to use. Supported
            methods are: 'linear', 'nearest', 'slinear', 'cubic',
            'quintic' and 'pchip'.
        :type method: str

        :return: The interpolated terrain elevation (in metres).
        :rtype: np.ndarray
        """

        # Pre-allocate memory for returning
        to_return = np.full(lat_points.shape, np.nan)

        # Obtain the origin positions for each coordinate
        lat_origins = np.floor(lat_points)
        lon_origins = np.floor(lon_points)

        # Obtain all unique origin positions:
        origins = np.column_stack([lat_origins, lon_origins])
        u_coords, u_indices, rev_indices = np.unique(
            origins, axis=0, return_index=True, return_inverse=True)

        # For each unique origin position and tile:
        for coords, index in zip(u_coords, u_indices):

            # Get the required DTED tile.
            dted_tile = self.__get_dted_tile(*coords)
            i = index == rev_indices

            # Extract the required elevations.
            to_return[i] = dted_tile.get_elevation_vec(
                lat_points[i], lon_points[i], method)

        # Return the results.
        return to_return

    def set_max_tiles(self, value: int):
        """
        A setter method for the maximum number of tiles.
        Can use the value None to remove limit on number of tiles to hold.

        :param value: The maximum number of DTED tiles to hold in the cache.
        :type value: int
        """
        if value is not None and value < 1:
            raise ValueError(f"Max tiles limit must be positive integer")
        self.__max_tiles = value

    def clear_cache(self):
        """
        Clears all cached DTED from memory.
        Simply removes all DTED tiles from the cache.
        """
        self.__dted_tiles.clear()

    def __get_dted_tile(self, lat: float, lon: float, to_remove: int = 1) -> DtedTile:
        """
        Return the corresponding DTED tile from cache or disk.
        Attempts to fetch the corresponding DTED tile from the cache. If not
        present, the DTED tile will be generated and added to the cache
        before being returned. If set, any size limits of the cache will be
        maintained.

        :param lat: The latitude position of the corresponding DTED tile.
        :type lat: float

        :param lon: The longitude position of the corresponding DTED tile.
        :type lon: float

        :param to_remove: The number of cache entries to remove if size is exceeded.
        :type to_remove: int

        :return: The corresponding DTED Tile covering the given position.
        :rtype: DtedTile
        """

        # Get the look-up key for the DTED tile
        lat_origin = math.floor(lat)
        lon_origin = math.floor(lon)
        tile_key = (lat_origin, lon_origin)

        # If the tile is not already in the cache:
        if tile_key not in self.__dted_tiles:

            # Ensure there is space for it:
            if (self.__max_tiles is not None and
                    len(self.__dted_tiles) >= self.__max_tiles):
                new_cache_size = max(self.__max_tiles - to_remove, 0)
                self.__reduce_tile_cache(lat, lon, new_cache_size)

            # Add the required DTED tile to the cache:
            self.__dted_tiles[tile_key] = DtedTile(
                lat_origin, lon_origin,
                self.__dted_dir,
                self.__dted_version,
                self.__interp_method)

        # Return the corresponding DTED tile.
        return self.__dted_tiles[tile_key]

    def __reduce_tile_cache(self, lat: float, lon: float, new_size: int):
        """
        Reduces size of the cache to the requested size.
        Removes the most distant DTED tiles from the given position so that
        the cache if of requested size with little probability of reloading
        previosuly cleared tiles.

        :param lat: The latitude coordinate of the current position.
        :type lat: float

        :param lon: The longitude coordinate of the current position.
        :type lon: float

        :param new_size: The new requested size of the tile cache.
        :type new_size: int
        """

        # Only continue if required:
        if new_size >= len(self.__dted_tiles):
            return

        # Get a list of (key, distance) tuples
        key_dist = [(key, tile.distance_to_centre(lat, lon))
                    for (key, tile) in self.__dted_tiles.items()]

        # Ensure the list is sorted by distance in ascending order.
        key_dist.sort(key=lambda a: a[2], reverse=False)

        # Remove the most distant entries in the dictionary:
        for to_remove, _ in key_dist[new_size:]:
            self.__dted_tiles.pop(to_remove)


def get_dted_file_path(dted_dir: Path, origin_lat: int, origin_lon: int, dted_version: int) -> Path:
    """
    Give a latitude, longitude and optional DTED version number,
    this function generates the file path of the DTED file that would
    contain information for the given position.

    :param dted_dir: The parent database_dir of the DTED files.
    :type dted_dir: Path

    :param origin_lat: The integer latitude value (south-most coordinate).
    :type origin_lat: int

    :param origin_lon: The integer longitude value (west-most coordinate).
    :type origin_lon: int

    :param dted_version: The version of the DTED data.
    :type dted_version: int

    :return the location of a corresponding DTED file.
    :rtype: Path
    """

    # Calculate the letters/ characters within the local path.
    lat_char = ("s", "n")[int(origin_lat) >= 0]
    lon_char = ("w", "e")[int(origin_lon) >= 0]

    # Round down the given latitude and longitude.
    origin_lat = math.floor(abs(origin_lat))
    origin_lon = math.floor(abs(origin_lon))

    # Format strings containing the  database_dir and data_file.
    lon_str = "{}{:03d}".format(lon_char, origin_lon)
    lat_str = "{}{:02d}.dt{:d}".format(lat_char, origin_lat, dted_version)

    # Return the completed path.
    return dted_dir / lon_str / lat_str


def read_dted_file(file_dir: Path) -> dict:
    """
    Extracts elevation information from a provided DTED file (in metres).
    When provided with the location of a DTED file (of any level), this
    function will return an elevation matrix and latitude-longitude spacing
    of each point. This function reads DTED files of format version 1.1.

    :param file_dir: The location of the DTED file to be read.
    :type file_dir: Path

    :return: A dictionary containing the contents of the DTED file.
        This contains the extracted origin coordinates, step sizes,
        coordinate ticks and elevation matrix.
    :rtype: dict
    """

    uhl_len = 80  # The "User Header Label" length in bytes.
    dsi_len = 648  # The "Data Set Identification Record" length in bytes.
    acc_len = 2700  # The "Accuracy Description Record" length in bytes.

    data_rec_head_len = 8  # The header length for each DTED record.
    data_rec_check_len = 4  # The checksum length appended to each DTED record.
    data_rec_start_pos = uhl_len + dsi_len + acc_len  # The total length of previous sections.

    # Attempt to open the given file and read in binary:
    with open(file_dir, "rb") as dted:

        # Skip the first 4 bytes.
        dted.seek(4, 0)

        # Obtain the origin longitude (make negative if west).
        lon_origin = trans.deg2dec(int(dted.read(3)), int(dted.read(2)), int(dted.read(2)))
        if dted.read(1) == b'W':
            lon_origin = -lon_origin

        # Obtain the origin latitude (make negative if west).
        lat_origin = trans.deg2dec(int(dted.read(3)), int(dted.read(2)), int(dted.read(2)))
        if dted.read(1) == b'S':
            lat_origin = -lat_origin

        # Obtain the latitude and longitude separation intervals (in seconds).
        lon_step = trans.deg2dec(0, 0, float(dted.read(4)) / 10)
        lat_step = trans.deg2dec(0, 0, float(dted.read(4)) / 10)
        dted.seek(47, 0)

        # Obtain the number of latitude and longitude lines (store as a string).
        num_lon_lines = int(dted.read(4).decode("utf-8"))
        num_lat_lines = int(dted.read(4).decode("utf-8"))

        # Initialise the return matrix and move the seek position to the first record.
        dted_matrix = np.zeros([num_lat_lines, num_lon_lines], dtype=np.int16)
        dted.seek(data_rec_start_pos, 0)

        # For each (longitude) record within the DTED file:
        for i in range(0, num_lon_lines):
            # Skip the header information.
            dted.seek(data_rec_head_len, 1)

            # Set the read datatype as 2-byte signed-integers.
            dt = np.dtype('>i2')

            # Read in the elevation points for the current longitude.
            dted_matrix[:, i] = np.fromfile(dted, dtype=dt, count=num_lat_lines)

            # Skip the check-sum.
            dted.seek(data_rec_check_len, 1)

        # Calculate the latitude-longitude separation spacing using the origin locations and known number of lines.
        lon_ticks = np.linspace(lon_origin, lon_origin + num_lon_lines * lon_step, num_lon_lines)
        lat_ticks = np.linspace(lat_origin, lat_origin + num_lat_lines * lat_step, num_lat_lines)

        # Replace values of -32,767 (or less) with the given (or default) nan value.
        # Within DTED files the smallest integer value is used as a placeholder for nan.
        dted_matrix[dted_matrix < 0] = -(dted_matrix[dted_matrix < 0] + 32768)
        dted_matrix = dted_matrix.astype(float)
        dted_matrix[dted_matrix == -32767] = np.nan

        # Check the data contains NaN values:
        # has_nan = np.any(np.isnan(dted_matrix))

        # Finally, return the collected data:
        return {
            'lat_origin': lon_origin,
            'lon_origin': lat_origin,
            'lat_step': lat_step,
            'lon_step': lon_step,
            'lat_ticks': lat_ticks,
            'lon_ticks': lon_ticks,
            'elevation': dted_matrix
        }


def get_template_dted_data(origin_lat: int, origin_lon: int,
                           num_lat_points: int = 2,
                           num_lon_points: int = 2) -> dict:
    """
    Generates a template DTED data dictionary as if read from a file.
    Simulates the output of :func:`~dted.read_dted_file` by generating
    identical output structure with flat elevation. This is to be used as
    substitution for missing/unavailable DTED files.

    :param origin_lat: The integer latitude value (south-most coordinate).
    :type origin_lat: int

    :param origin_lon: The integer longitude value (west-most coordinate).
    :type origin_lon: int

    :param num_lat_points: The number of latitude points (rows) to generate.
    :type num_lat_points: int

    :param num_lon_points: The number of longitude points (columns) to generate.
    :type num_lon_points: int

    :return: A dictionary containing contents of a DTED file.
        This contains the imitated origin coordinates, step sizes,
        coordinate ticks and elevation matrix that would be read.
    :rtype: dict
    """

    assert num_lat_points >= 2, 'Number of lat points must be greater than 1!'
    assert num_lon_points >= 2, 'Number of lon points must be greater than 1!'

    lat_ticks = np.linspace(origin_lat, origin_lat + 1, num_lat_points)
    lon_ticks = np.linspace(origin_lon, origin_lon + 1, num_lon_points)
    dted_matrix = np.zeros((num_lat_points, num_lon_points))

    lat_step = 1 / (num_lat_points - 1)
    lon_step = 1 / (num_lon_points - 1)

    # Finally, return the collected data:
    return {
        'lat_origin': origin_lat,
        'lon_origin': origin_lon,
        'lat_step': lat_step,
        'lon_step': lon_step,
        'lat_ticks': lat_ticks,
        'lon_ticks': lon_ticks,
        'elevation': dted_matrix
    }
