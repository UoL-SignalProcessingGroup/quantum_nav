
import pytest
import random
import numpy as np

from pathlib import Path
from qnav.earth.geoid import GeoidModel, GeoidDummy
from qnav.earth.geoid import GeoidPGM


# The minimum supported latitude coordinate
_MIN_LAT = -90.0

# The maximum supported latitude coordinate
_MAX_LAT = 90.0

# The minimum supported longitude coordinate
_MIN_LON = -180.0

# The maximum supported longitude coordinate
_MAX_LON = 180.0

# The highest geoid height correction.
_MAX_HEIGHT = 90.0

# The lowest geoid height correction.
_MIN_HEIGHT = -110

# The maximum DOV.
_MAX_DOV = 0.1

@pytest.fixture
def random_position() -> tuple[float, float]:
    """
    Returns a random position on the Earth surface.
    """
    lat = random.uniform(_MIN_LAT, _MAX_LAT)
    lon = random.uniform(_MIN_LON, _MAX_LON)
    return lat, lon

@pytest.fixture
def random_positions(request) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns random positions on the Earth surface.
    The amount is set via parameters in-directly
    """
    num_points = request.param
    lat = np.random.uniform(_MIN_LAT, _MAX_LAT, num_points)
    lon = np.random.uniform(_MIN_LON, _MAX_LON, num_points)
    return lat, lon

@pytest.fixture(scope='module')
def geoid_dir(project_dir) -> Path:
    """
    Returns the path for the geoid database.

    :param project_dir: Fixture for the project root directory.
    :type project_dir: :class:`pathlib.Path`

    :return: Path where geoid database files are located.
    :rtype: :class:`pathlib.Path`
    """
    return project_dir / 'databases' / 'geoid'

@pytest.fixture(scope='module')
def known_examples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns tuple containing known locations and their geoid heights.

    :return: Tuple contain latitude, longitude and height values.
    :rtype: tuple[list[float], list[float], list[float]]
    """
    points = np.array([
        [ 35.2123524,    68.834067, -27.41],   # Afghanistan
        [ 52.3649088,     4.902883,  43.20],   # Amsterdam
        [-76.0182814,    21.542722,  11.24],   # Antarctica
        [ 78.15780272, -141.446294,  -4.5],    # Arctic Ocean
        [ -9.68719415,  -41.537450, -14.86],   # Brazil
        [ 53.0697269,   -97.891645, -33.32],   # Canada Lake
        [ 51.131163,     -1.704214,  47.68],   # Dstl Porton Down
        [ 67.48231784,  -44.6424197, 44.68],   # Greenland
        [ 64.821810,    -18.802072,  67.90],   # Iceland
        [ 35.73396830,  138.002083,  43.33],   # Japan
        [ 53.401640,     -2.963275,  52.88],   # Liverpool University
        [-16.9425386,    47.8420728, -8.61],   # Madagascar
        [ 11.349565,    142.199480,  34.38],   # Mariana Trench
        [ 27.988194,     86.924374, -28.34],   # Mount Everest
        [ 33.226975,    -41.475063,  19.41],   # North Atlantic Ocean
        [ -7.0698638,   146.252274,  78.02],   # Papua New Guinea
        [-10.853377,    -75.652081,  29.45],   # Peru
        [-28.8616772,   -14.646566,  13.79],   # South Atlantic Ocean
        [ 46.5208957,     8.012781,  52.85],   # Switzerland
        [-25.34511,     131.034356,   2.25],   # Uluru
    ])
    return points[:, 0], points[:, 1], points[:, 2]


def get_geoid_pgm_files() -> list[str]:
    """
    Generate list containing Geoid PGM files for testing.

    :return: List of PGM file names with file extensions.
    :rtype: list[str]
    """
    return ['egm84-30.pgm', 'egm84-15.pgm',
            'egm96-15.pgm', 'egm96-5.pgm',
            'egm2008-5.pgm', 'egm2008-2_5.pgm', 'egm2008-1.pgm']


class TestPGMGeoid:
    """
    Executes tests cases for local
    """

    @pytest.fixture(params=get_geoid_pgm_files(), scope='class')
    def geoid(self, request, geoid_dir: Path) -> GeoidPGM:
        """
        Fixture for the current geoid instance.
        """
        geoid_file = request.param
        geoid_path = geoid_dir / geoid_file
        return GeoidPGM(geoid_path)

    @staticmethod
    def test_height(geoid: GeoidPGM, random_position):
        """
        Ensures geoid heights are within the sensible range.
        """
        h = geoid.get_height(*random_position)
        assert _MIN_HEIGHT < h < _MAX_HEIGHT

    @staticmethod
    @pytest.mark.parametrize('random_positions', (1, 10, 10000), indirect=True)
    def test_height_vec(geoid: GeoidPGM, random_positions):
        """
        Also ensure that geoid heights are still within the sensible range.
        """
        h_vec = geoid.get_height_vec(*random_positions)
        is_valid = (_MIN_HEIGHT < h_vec) & (h_vec < _MAX_HEIGHT)
        assert np.all(is_valid)

    @staticmethod
    @pytest.mark.parametrize('random_positions', (1, 10, 10000), indirect=True)
    def test_height_vec_compare(geoid: GeoidPGM, random_positions):
        """
        Ensures vectorised height interpolation produces same results as scalar.
        """
        h_vec = geoid.get_height_vec(*random_positions)
        for i, (lat, lon) in enumerate(zip(*random_positions)):
            h = geoid.get_height(lat, lon)
            np.testing.assert_almost_equal(h, h_vec[i], decimal=10)

    @staticmethod
    def test_dov(geoid: GeoidPGM, random_position):
        """
        Ensures calculated dov are within the sensible range.
        """
        dov_north, dov_east = geoid.get_dov(*random_position)
        assert abs(dov_north) < _MAX_DOV
        assert abs(dov_east) < _MAX_DOV

    @staticmethod
    @pytest.mark.parametrize('random_positions', (1, 10, 10000), indirect=True)
    def test_dov_vec(geoid: GeoidPGM, random_positions):
        """
        Ensures vectorised dov produces same results as scalar.
        """
        dov_vec = geoid.get_dov_vec(*random_positions)
        north_dov_vec, east_dov_vec = dov_vec
        assert np.all(np.abs(north_dov_vec) < _MAX_DOV)
        assert np.all(np.abs(east_dov_vec) < _MAX_DOV)

    @staticmethod
    @pytest.mark.parametrize('random_positions', (1, 10, 10000), indirect=True)
    def test_dov_vec_compare(geoid: GeoidPGM, random_positions):
        """
        Ensures vectorised dov calculation produces same results as scalar.
        """
        dov_vec = geoid.get_dov_vec(*random_positions)
        north_dov_vec, east_dov_vec = dov_vec

        for i, (lat, lon) in enumerate(zip(*random_positions)):
            north_dov, east_dov = geoid.get_dov(lat, lon)
            np.testing.assert_almost_equal(north_dov, north_dov_vec[i], decimal=10)
            np.testing.assert_almost_equal(east_dov, east_dov_vec[i], decimal=10)

    @staticmethod
    def test_known_cases(geoid: GeoidPGM, known_examples):
        """
        Compares the interpolated geoid heights against known cases.
        """

        # Obtain the average resolution
        # north_res, east_res = geoid.resolution
        # tol = ((90 / north_res) + (180 / east_res)) * 25
        # print(tol)

        for lat, lon, height in zip(*known_examples):
            h = geoid.get_height(lat, lon)
            assert abs(h - height) <= 8.0


class TestDummyGeoid:
    """
    Executes tests for a dummy/placeholder geoid model.
    """

    @pytest.fixture(scope='class')
    def geoid(self, request, geoid_dir: Path) -> GeoidModel:
        """
        Fixture for the current geoid instance.
        """
        return GeoidDummy()

    @staticmethod
    def test_height(geoid, random_position):
        """
        Ensures geoid heights are within the sensible range.
        """
        h = geoid.get_height(*random_position)
        assert h == 0.0

    @staticmethod
    @pytest.mark.parametrize('random_positions', (1, 10, 10000), indirect=True)
    def test_height_vec(geoid, random_positions):
        """
        Also ensure that geoid heights are still within the sensible range.
        """
        h_vec = geoid.get_height_vec(*random_positions)
        assert np.all(h_vec == 0.0)

    @staticmethod
    def test_dov(geoid, random_position):
        """
        Ensures calculated dov are within the sensible range.
        """
        dov_north, dov_east = geoid.get_dov(*random_position)
        assert dov_north == 0.0
        assert dov_east == 0.0

    @staticmethod
    @pytest.mark.parametrize('random_positions', (1, 10, 10000), indirect=True)
    def test_dov_vec(geoid, random_positions):
        """
        Ensures vectorised dov produces same results as scalar.
        """
        dov_vec = geoid.get_dov_vec(*random_positions)
        north_dov_vec, east_dov_vec = dov_vec
        assert np.all(north_dov_vec == 0.0)
        assert np.all(east_dov_vec == 0.0)
