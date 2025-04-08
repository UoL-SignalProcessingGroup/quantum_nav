
import pytest

from gravity.map_tests import MapTests, DownloadTests
from qnav.gravity.base import GravityModel
from qnav.gravity.srtm2gravity import SRTM2GravityFull
from qnav.gravity.srtm2gravity import SRTM2GravityRes


@pytest.fixture
def valid_lla_range() -> dict[str, float]:
    """
    Defines the range for valid LLA coordinates for geosat44.

    :return: A dictionary with the range of supported LLA positions.
    :rtype: dict[str, float]
    """
    return {
        'min_lat': 50.0, 'max_lat': 55.0,
        'min_lon': -4.0, 'max_lon': -1.0,
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


class TestSRTM2GravityFull(MapTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        allow_downloads = False
        srtm_dir = database_dir / 'srtm2gravity'
        return SRTM2GravityFull(gravity_func, geoid_model, srtm_dir, allow_downloads)


class TestSRTM2GravityRes(MapTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        allow_downloads = False
        srtm_dir = database_dir / 'srtm2gravity'
        return SRTM2GravityRes(gravity_func, geoid_model, srtm_dir, allow_downloads)


@pytest.mark.slow
class TestSRTM2GravityFullDownload(DownloadTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, tmp_gravity_dir) -> GravityModel:
        allow_downloads = True
        return SRTM2GravityFull(gravity_func, geoid_model, tmp_gravity_dir, allow_downloads)


@pytest.mark.slow
class TestSRTM2GravityResDownload(DownloadTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, tmp_gravity_dir) -> GravityModel:
        allow_downloads = True
        return SRTM2GravityRes(gravity_func, geoid_model, tmp_gravity_dir, allow_downloads)
