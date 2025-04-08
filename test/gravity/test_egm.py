
import pytest

from gravity.map_tests import MapTests
from qnav.gravity.base import GravityModel
from qnav.gravity.egm import GeoidCorrection


class TestEgmGravity(MapTests):

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        return GeoidCorrection(gravity_func, geoid_model)
