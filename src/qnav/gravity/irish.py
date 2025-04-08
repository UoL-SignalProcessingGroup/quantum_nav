

import re
import numpy as np

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from typing import Optional
from pathlib import Path

from qnav.earth.geoid import GeoidModel
from qnav.gravity.base import GravityModel
from qnav.gravity.map import GravityMap
from qnav.earth.utm import utm2lla


class IrishSeaGravity(GravityMap):

    # The raster file containing the gravity data.
    _DATA_FILE = 'zone3_freeair_500_all'

    # The ERS file containing the meta data.
    _ERS_FILE = 'zone3_freeair_500_all.ers'

    _anom_interp = []

    def __init__(self, base_model: GravityModel,
                 geoid_model: Optional[GeoidModel],
                 map_dir: Optional[Path]):

        super().__init__(base_model, geoid_model, map_dir)
        self.load_data()

    def __str__(self) -> str:
        return 'IrishSeaGravity'

    def load_data(self):
        """
        Loads the Marine Gravity data and generates interpolator for look-up.
        Called on initialisation to load the required data to allow for
        correction interpolation.
        """

        # Specify the files to load
        data_file = self._map_dir / self._DATA_FILE
        meta_file = self._map_dir / self._ERS_FILE

        # Check files exist to produce clean errors:
        for file in (data_file, meta_file):
            if not file.is_file():
                raise FileNotFoundError(f'missing irish sea file {file}')

        # Read the corresponding meta data file
        meta_data = read_irish_ers_file(meta_file)
        nan_value = meta_data['null_value']

        # Read and reshape the gravity data file
        grav_data = read_irish_gravity_file(data_file, nan_value)
        grid_size = (int(meta_data['num_rows']), int(meta_data['num_cols']))
        grav_data = np.reshape(grav_data, grid_size)

        # Obtain the east and north grid points
        east_grid, north_grid = get_easting_northing_grid(meta_data)

        # Remove NaN values
        is_valid = ~np.isnan(grav_data)
        grav_points = grav_data[is_valid]
        # east_points = east_grid[is_valid]
        # north_points = north_grid[is_valid]

        # TODO: Improve performance of correct conversion
        # # Convert from UTM to LLA
        # lla_points = np.zeros([north_points.size, 2])
        # for i, (x, y) in enumerate(zip(east_points, north_points)):
        #     lla_points[i, :] = utm2lla(x, y, meta_data['zone'])

        # TODO: Remove temporary approx conversion
        # ------------------------------------------------------ #

        # Get the corner points of the grid
        nw_point = (float(east_grid[0, 0]), float(north_grid[0, 0]))
        se_point = (float(east_grid[-1, -1]), float(north_grid[-1, -1]))

        # Convert corner points to latitude and longitude
        nw_ll = utm2lla(nw_point[0], nw_point[1], meta_data['zone'])
        se_ll = utm2lla(se_point[0], se_point[1], meta_data['zone'])

        # Generate a grid of latitude and longitude points
        grid_size = north_grid.shape
        lat_grid = np.linspace(nw_ll[0], nw_ll[1], grid_size[0])
        lon_grid = np.linspace(se_ll[0], se_ll[1], grid_size[1])
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

        # Remove
        lat_points = lat_grid[is_valid]
        lon_points = lon_grid[is_valid]
        lla_points = np.column_stack((lat_points, lon_points))

        # ------------------------------------------------------ #

        # Generate an interpolation instance for the geosat data.
        lla_tri = Delaunay(lla_points)
        self._anom_interp = LinearNDInterpolator(
            lla_tri, grav_points, fill_value=0.0, rescale=False)


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
        anom_values = self._anom_interp(lat, lon)
        if np.isscalar(lat) and np.isscalar(lon):
            anom_values = float(anom_values)
        return anom_values

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


def read_irish_gravity_file(data_file: Path, null_value: float = -99999) -> np.ndarray:
    """
    Extracts the data from a given Irish Sea Gravity Map data file, using
    values read from its corresponding header file. On succession, the read
    gravitational anomaly matrix will be returned (measured in m/s^2). No axis
    information is contained within data files.

    :param data_file: The location of the Irish Sea Gravity Map data file.
    :type data_file: str or Path

    :param null_value: Data values less than or equal to this value will be
        converted to NaN values (default=-99999).
    :type null_value: float

    :return: The extracted gravitational anomaly matrix (measured in m/s^2).
    :rtype: 2D NumPy Array
    """

    # Read the data from the file and convert to floats
    data = np.fromfile(data_file, '<f4').astype(np.float64)
    # data = data.reshape([num_rows, num_cols])

    # Insert NaN values and convert from mGals to m/s^2.
    data[data <= null_value] = np.nan
    data /= 1e5

    # Finally, return the extracted matrix.
    return data

def read_irish_ers_file(header_file) -> dict[str, any]:
    """
    This function will read a given Irish Sea Gravity Map header file (a file
    normally with the extension '.ers') and extract the relevant information
    from it.

    :param header_file: The location of the header corresponding file.
    :type header_file: str or Path

    :return: A dictionary holding the extracted northings and eastings
        origin coordinates, data dimensions, and cell spacing information.
    """

    # The results to return
    results = {}

    # The search patterns to find within the file
    to_match = [
        ('Eastings\\s*=\\s*(.[0-9-.]+)', 'eastings'),
        ('Northings\\s*=\\s*(.[0-9-.]+)', 'northings'),
        ('NrOfCellsPerLine\\s*=\\s*(.[0-9-.]+)', 'num_cols'),
        ('NrOfLines\\s*=\\s*(.[0-9-.]+)', 'num_rows'),
        ('Xdimension\\s*=\\s*(.[0-9-.]+)', 'x_dim'),
        ('Ydimension\\s*=\\s*(.[0-9-.]+)', 'y_dim'),
        ('Projection\\s*=\\s*(.[\\w\\"]+)', 'zone'),
        ('NullCellValue\\s*=\\s*(.[0-9-.]+)', 'null_value'),
    ]

    # Using the contents of the given file:
    with open(str(header_file), "r") as file:
        for line in file:

            # For each of the search patterns:
            for regex, name in to_match:

                # Search for the current pattern.
                finder = re.search(regex, line)

                if finder:
                    results[name] = _str2num(finder.group(1))

    # Finally, return the found values as their required format.
    return results

def get_easting_northing_grid(ers_data: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:

    x_points = np.arange(ers_data['num_cols'], dtype=np.float64)
    x_points *= ers_data['x_dim']
    x_points += ers_data['eastings']

    y_points = -np.arange(ers_data['num_rows'], dtype=np.float64)
    y_points *= ers_data['y_dim']
    y_points += ers_data['northings']

    x_grid, y_grid = np.meshgrid(x_points, y_points)
    return x_grid, y_grid


def _str2num(str_value):
    """
    A simple function that converts a given string to an integer, subject to
    the numerical values contained within the string. The function supports
    both positive and negative values. The first match/ instance

    :param str_value: The string to extract the integer value from.
    :type str_value: str

    :return: The first whole integer found within the given string.
    :rtype: int
    """

    # If a None value is given, simply return None.
    if str_value is None:
        return None

    # Find all integer values within the given string.
    # matches = re.findall(r"-?\d+", str_value)
    matches = re.findall(r'[-+]?\d*\.\d+|\d+', str_value)

    # Raise value error if no match is found:
    if len(matches) == 0:
        raise ValueError(
            f"The string '{str_value}' could not be converted to number!")

    # Otherwise, return teh first occurrence.
    return float(matches[0])


if __name__ == "__main__":

    from qnav.gravity.somigliana import Somigliana

    project_dir = Path(__file__).parent.parent.parent
    grav_dir = project_dir / 'databases' / 'gravity' / 'irish_sea'

    som = Somigliana()
    irish = IrishSeaGravity(som, None, grav_dir)
    test_g = irish.calc_gravity_z(52, -3, 1000)
