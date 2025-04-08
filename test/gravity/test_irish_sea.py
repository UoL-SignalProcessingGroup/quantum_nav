
import pytest

from gravity.map_tests import MapTests
from qnav.gravity.base import GravityModel
from qnav.gravity.irish import IrishSeaGravity


@pytest.fixture
def valid_lla_range() -> dict[str, float]:
    """
    Defines the range for valid LLA coordinates for geosat44.

    :return: A dictionary with the range of supported LLA positions.
    :rtype: dict[str, float]
    """
    return {
        'min_lat': 52.5861, 'max_lat': 56.2463,
        'min_lon': -19.7746, 'max_lon': -12.0735,
        'min_alt': 0.0, 'max_alt': 10000.0
    }


@pytest.fixture
def invalid_lla_range() -> dict[str, float]:
    """
    Defines the range for invalid LLA coordinates for geosat44.

    :return: A dictionary with the range of unsupported LLA positions.
    :rtype: dict[str, float]
    """
    return {
        'min_lat': 15.0, 'max_lat': 40.0,
        'min_lon': -50.0, 'max_lon': -25.0,
        'min_alt': 0.0, 'max_alt': 10000.0
    }


class TestIrishSea(MapTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        ggm_plus_dir = database_dir / 'irish_sea'
        return IrishSeaGravity(gravity_func, geoid_model, ggm_plus_dir)

