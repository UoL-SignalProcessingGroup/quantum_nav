
import pytest

from gravity.model_tests import ModelTests
from qnav.gravity.base import GravityModel
from qnav.gravity.nima import WGS84Gravity


class TestWGS84Gravity(ModelTests):
    """
    Performs tests using NIMA's WGS84 gravity model.
    """

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir)-> GravityModel:
        return WGS84Gravity()
