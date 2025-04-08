"""
========
geoid.py
========

:summary:
    A module for interpolating geoid heights from presented geoid PGM files.
    This module provides classes and functions used to extract geoid
    information from given PGM encoded geoid files. The most common use for
    this module is for interpolating the geoid height at requested positions.

:authors:
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of this module.
"""

import numpy as np
import re

from abc import ABC
from abc import abstractmethod
from scipy import interpolate
from pathlib import Path

from qnav.util.transformations import ned2lla
from qnav.util.transformations import ned2lla_vec


# The default offset value used for encoding heights inside a PGM file.
_PGM_OFFSET = -108

# The default scale value used for encoding heights inside a PGM file.
_PGM_SCALE = 0.003


class GeoidModel(ABC):
    """
    Abstract base class for geoid representations.
    Defines the method that implemented geoid representations must supply.
    """

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a description of the geoid model instance.
        Named used for summarising the model and its configuration. Mostly
        used when reporting or reviewing simulations.

        :return: A string describing the model instance.
        :rtype: str
        """
        pass

    @abstractmethod
    def get_height(self, lat: float, lon: float) -> float:
        """
        Returns geoid height for single latitude and longitude position.

        :param lat: A latitude coordinate in decimal degrees.
        :type lat: float

        :param lon: A longitude coordinate in decimal degrees.
        :type lon: float

        :return: The geoid height for the corresponding location (in metres).
        :rtype: float
        """
        pass

    @abstractmethod
    def get_height_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Returns geoid height for given latitude and longitude positions.

        :param lat: The latitude coordinates in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinates in decimal degrees.
        :type lon: float

        :return: The geoid height for the corresponding locations (in metres).
        :rtype: float
        """
        pass

    @abstractmethod
    def get_dov(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Returns the Deflection of the Vertical (DoV) for requested position.
        Calculates and returns the DoV gradients for the north and east
        components (in metres) for requested location.

        :param lat: A latitude coordinate in decimal degrees.
        :type lat: float

        :param lon: A longitude coordinate in decimal degrees.
        :type lon: float

        :return: Tuple containing the north and east components respectively.
        :rtype: tuple[float, float]
        """
        pass

    @abstractmethod
    def get_dov_vec(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the Deflection of the Vertical (DoV) for requested positions.
        Calculates and returns the DoV gradients for the north and east
        components (in metres) for requested location.

        :param lat: The latitude coordinates in decimal degrees.
        :type lat: np.ndarray

        :param lon: The longitude coordinates in decimal degrees.
        :type lon: np.ndarray

        :return: Tuple containing the north and east components respectively.
        :rtype: tuple[float, float]
        """
        pass


class GeoidPGM(GeoidModel):
    """
    A class for handling and managing reading the Geoid height information
    from a selected PGM (Pixel Gray Map) file and providing functionality for
    extracting required information from it. Methods for reading or
    interpolating the Geoid height at given coordinates, as well as
    visualising the loaded data are all contained within this class.
    """

    # The pixel gray map file path.
    _pgm_file: Path

    # The matrix containing the read geoid heights.
    _geoid_matrix = None

    # The interpolation of the above matrix.
    _geoid_matrix_interp = None

    def __init__(self, pgm_file: Path):
        """
        Reads and returns geoid interpolate for given geoid PGM file.
        On succession, produces a look-up class for interpolating the geoid
        height at requested positions, using the provided geoid PGM data.

        :param pgm_file: The name of the PGM file to be used.
        :type pgm_file: Path
        """

        # Attempt to read the given geoid file.
        self._geoid_matrix = read_pgm_data(pgm_file)
        self._pgm_file = pgm_file

        # Knowing the shape of the above matrix, calculate the latitude and
        # longitude coordinates for each position within the matrix.
        geoid_size = np.shape(self._geoid_matrix)
        lat_ticks = np.linspace(-90, 90, geoid_size[0])
        lon_ticks = np.linspace(-180, 180, geoid_size[1])

        self._geoid_matrix_interp = interpolate.RegularGridInterpolator(
            (lat_ticks, lon_ticks), self._geoid_matrix, 'linear')

        # With the original matrix known, produce an interpolated version.
        # self._geoid_matrix_interp = interpolate.RectBivariateSpline(
        #     lat_ticks, lon_ticks, self._geoid_matrix)

    def __str__(self) -> str:
        """
        Returns a describing name for the geoid model.
        Simply extracts and returns the file name of the PGM file used.

        :return: An identifying name of the geoid model.
        :rtype: str
        """
        return self._pgm_file.with_suffix('').name

    def get_height(self, lat: float, lon: float) -> float:
        """
        Interpolates geoid height for given latitude and longitude.
        For a single coordinate point, interpolates and returns the geoid
        elevation height in metres.

        :param lat: A latitude coordinate in decimal degrees.
        :type lat: float

        :param lon: A longitude coordinate in decimal degrees.
        :type lon: float

        :return: The geoid height for the corresponding location (in metres).
        :rtype: float
        """
        # return float(self._geoid_matrix_interp(lat, lon, grid=False))
        return self._geoid_matrix_interp((lat, lon))

    def get_height_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Interpolates geoid height for given latitude and longitude points.
        For a single coordinate point, interpolates and returns the geoid
        elevation height in metres.

        :param lat: The latitude coordinates in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinates in decimal degrees.
        :type lon: float

        :return: The geoid height for the corresponding locations (in metres).
        :rtype: float
        """
        if lat.shape != lon.shape:
            raise ValueError('lat and lon array must be of same shape')
        # return self._geoid_matrix_interp(lat, lon, grid=False)
        return self._geoid_matrix_interp((lat, lon))

    def get_dov(self, lat: float, lon: float, step_size: float = 100) -> tuple[float, float]:
        """
        Returns the Deflection of the Vertical (DoV) for requested position.
        Calculates and returns the DoV gradients for the north and east
        components (in metres) for requested location.

        :param lat: A latitude coordinate in decimal degrees.
        :type lat: float

        :param lon: A longitude coordinate in decimal degrees.
        :type lon: float

        :param step_size: The step size for calculating the gradient.
        :type step_size: float

        :return: Tuple containing the north and east components respectively.
        :rtype: tuple[float, float]
        """

        # Get the offsets for north and east
        north_offset = np.array([step_size, 0, 0])
        east_offset = np.array([0, step_size, 0])

        # Get the three position coordinates
        centre_lla = np.array([lat, lon, 0.0])
        north_lla = ned2lla(north_offset, centre_lla)
        east_lla = ned2lla(east_offset, centre_lla)

        # Get the geoid heights for all positions.
        centre_height = self.get_height(lat, lon)
        north_height = self.get_height(float(north_lla[0]), float(north_lla[1]))
        east_height = self.get_height(float(east_lla[0]), float(east_lla[1]))

        # Get and return the two DoV gradients
        north_dov = ((north_height - centre_height) / step_size)
        east_dov = ((east_height - centre_height) / step_size)
        return north_dov, east_dov

    def get_dov_vec(self, lat: np.ndarray, lon: np.ndarray, step_size: float = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the Deflection of the Vertical (DoV) for requested position.
        Calculates and returns the DoV gradients for the north and east
        components (in metres) for requested location.

        :param lat: The latitude coordinates in decimal degrees.
        :type lat: np.ndarray

        :param lon: The longitude coordinates in decimal degrees.
        :type lon: np.ndarray

        :param step_size: The step size for calculating the gradient.
        :type step_size: float

        :return: Tuple containing the north and east components respectively.
        :rtype: tuple[float, float]
        """

        if lat.shape != lon.shape:
            raise ValueError('lat and lon array must be of same shape')

        # Get the offsets for north and east
        north_offset = np.array([step_size, 0, 0])
        east_offset = np.array([0, step_size, 0])

        # Format the centre positions
        centre_lla = np.zeros([lat.size, 3])
        centre_lla[:, 0] = lat.ravel()
        centre_lla[:, 1] = lon.ravel()

        # Get the north and east position coordinates
        north_lla = ned2lla_vec(north_offset, centre_lla)
        east_lla = ned2lla_vec(east_offset, centre_lla)

        # Get the geoid heights for all positions.
        centre_heights = self.get_height_vec(centre_lla[:, 0], centre_lla[:, 1])
        north_heights = self.get_height_vec(north_lla[:, 0], north_lla[:, 1])
        east_heights = self.get_height_vec(east_lla[:, 0], east_lla[:, 1])

        # Get and return the two DoV gradients
        north_dov = ((north_heights - centre_heights) / step_size)
        east_dov = ((east_heights - centre_heights) / step_size)
        north_dov = np.reshape(north_dov, lat.size)
        east_dov = np.reshape(east_dov, lat.size)
        return north_dov, east_dov

    @property
    def resolution(self) -> tuple[int, int]:
        """
        The north and east resolutions of the grid data.
        Can be used for determining the accuracy of the model.

        :return: The north and east steps of the grid data.
        :rtype: tuple[int, int]
        """
        return self._geoid_matrix.shape


class GeoidDummy(GeoidModel):

    def __str__(self) -> str:
        return 'No Geoid'

    def get_height(self, lat: float, lon: float) -> float:
        return 0

    def get_height_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return np.zeros(lat.shape)

    def get_dov(self, lat: float, lon: float) -> tuple[float, float]:
        return 0, 0

    def get_dov_vec(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(lat.shape), np.zeros(lat.shape)


def read_pgm_data(pgm_file: Path,
                  pgm_offset: float = _PGM_OFFSET,
                  pgm_scale: float = _PGM_SCALE) -> np.ndarray:
    """
    Reads and returns the numerical contents of given PGM file.
    For a given PGM file, this function will read it and extract the Geoid
    height information (relative to the Earth's Ellipsoid) from it. The
    returned matrix's origin point's coordinates will be [90, -180] with the
    last point's coordinates being [-90, 180].

    :param pgm_file: The location of the PGM file containing Geoid data.
    :type pgm_file: str or Path

    :param pgm_offset: (Optional) The offset values for the values within the
        PGM file as PGM values normally start from 0 (default = -108).
    :type pgm_offset: float

    :param pgm_scale: (Optional) The scale for the values within the PGM file
        as PGM values can only be integers (default = 0.003).
    :type pgm_scale: float

    :return: A matrix containing the Geoid height (relative to the Earth's
        Ellipsoid) for various locations along the earth.
    :rtype: np.ndarray
    """
    # The unique string used to identify PGM file format.
    magic_string = b'P5\n'

    # The string used to represent the starting symbol for a comment.
    comment_string = b"#"

    # Attempt to open the given file and read in binary:
    with open(pgm_file, "rb") as pgm:

        # If the file is not a valid PGM file, raise an error.
        if pgm.readline() != magic_string:
            raise ValueError("File given is not a valid PGM file!")

        # Skip (the header) to the next line that is not a comment.
        current_line = pgm.readline()
        while current_line.startswith(comment_string):
            current_line = pgm.readline()

        # Read the number of columns and rows from the next line.
        # These values are strings separated by a space.
        dims = current_line.decode().split()
        num_columns = int(dims[0])
        num_rows = int(dims[1])

        # Read the string stating the maximum integer value within the data.
        max_value = int(pgm.readline().decode())

        # Knowing the maximum integer value, we can calculate the number of
        # expected bytes each value will take (either 1 or 2 bytes).
        # data_type = ('>u2', '>u1')[max_value < 256]
        data_type = '>u1' if max_value < 256 else '>u2'

        # Read and record all the remaining values within the file.
        # These values are the actual data contained within the PGM file.
        pgm_matrix = np.fromfile(pgm, dtype=data_type, count=-1, offset=0)

        # Reshape the resulting matrix to the expected dimensions.
        # (Convert the 1x(NxM) matrix to an MxN matrix).
        pgm_matrix = np.resize(pgm_matrix, [num_rows, num_columns])

        # Multiply the resulting matrix by the given scale and add the offset.
        return_matrix = (pgm_matrix * pgm_scale) + pgm_offset

        # While not required, shift the origin of the produced matrix from
        # latitude=90 and longitude=0 to latitude=90 and longitude=-180.
        return_matrix = np.roll(return_matrix, round(num_columns / 2), 1)

        # Finally, return the resulting matrix flipped within the up/down
        # direction. The increasing order of axis is required for some
        # interpolation methods (including cubic).
        return np.flipud(return_matrix)


def read_pgm_header(pgm_file: Path) -> dict:
    """
    Returns values present in the header of a given PGM file.
    PGM files can contain comments which describe information such as the
    offset and scale values for the Geoid data they contain. For scenarios
    where the offset and scale are contained within these comments, this
    function will find and extract their values.

    :param pgm_file: The location of the PGM file containing Geoid data.
    :type pgm_file: Path

    :return: A dictionary containing the found offset and scale values
        (or None/Null if the values could not be found within the comments).
    :rtype: dict
    """

    offset = None  # The offset value to be found and returned.
    scale = None  # The scale value to be found and returned.

    # The string used to represent the starting symbol for a comment.
    comment_string = b"# "

    # The regular expression to match and find for the offset value.
    offset_regex = 'offset (.[0-9-.]+)'

    # The regular expression to match and find for the scale value.
    scale_regex = 'scale (.[0-9-.]+)'

    # Attempt to open the given file and read in binary:
    with open(pgm_file, "rb") as pgm:

        # For each line within the given file:
        for line in pgm:

            # If the current line starts with the comment string:
            if line.startswith(comment_string):

                # Get the current line of the file and convert to lower case.
                current_line = line.decode().lower()

                # Search the current line for the given search strings.
                offset_search = re.search(offset_regex, current_line)
                scale_search = re.search(scale_regex, current_line)

                # If a match for the offset regular expression was found,
                # use the first occurrence for the return value.
                if offset_search:
                    offset = offset_search.group(1)

                # If a match for the scale regular expression was found,
                # use the first occurrence for the return value.
                if scale_search:
                    scale = scale_search.group(1)

    # Finally, return the values for offset and scale.
    return {"offset": offset, "scale": scale}



# if __name__ == "__main__":
#
#     project_dir = Path(__file__).parent.parent.parent
#     test_pgm = project_dir / 'databases' / 'geoid' / 'egm2008-5.pgm'
#     test_geoid = GeoidPGM(test_pgm)
#
#     [n_dov, e_dov] = test_geoid.get_dov(52, -3)
#
#     test_lats = np.random.rand(10) * 180 - 90
#     test_lons = np.random.rand(10) * 360 - 180
#     [n_dovs, e_dovs] = test_geoid.get_dov_vec(test_lats, test_lons)

