
import pytest

from gravity.map_tests import MapTests, DownloadTests
from qnav.gravity.base import GravityModel
from qnav.gravity.geosat import Geosat44


@pytest.fixture
def valid_lla_range() -> dict[str, float]:
    """
    Defines the range for valid LLA coordinates for geosat44.

    :return: A dictionary with the range of supported LLA positions.
    :rtype: dict[str, float]
    """
    return {
        'min_lat': 10.0, 'max_lat': 45.0,
        'min_lon': -55.0, 'max_lon': -20.0,
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
        'min_lat': 72.1, 'max_lat': 90.0,
        'min_lon': -180.0, 'max_lon': 180.0,
        'min_alt': 0.0, 'max_alt': 10000.0
    }


class TestGeosat(MapTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        allow_downloads = False
        geosat_dir = database_dir / 'geosat'
        return Geosat44(gravity_func, geoid_model, geosat_dir, allow_downloads)


@pytest.mark.slow
class TestGeosatDownload(DownloadTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, tmp_gravity_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        allow_downloads = True
        return Geosat44(gravity_func, geoid_model, tmp_gravity_dir, allow_downloads)

