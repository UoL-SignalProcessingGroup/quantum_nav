
import pytest

from gravity.model_tests import ModelTests
from qnav.gravity.base import GravityModel
from qnav.gravity.simple import FixedValue


class TestFixedValue(ModelTests):
    """
    Performs tests using simple fixed value gravity model.
    Some tests have been changed to accommodate the unusual behaviours
    produced by such a simplified model (i.e. gravity being constant).
    """

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        return FixedValue()

    @staticmethod
    def test_gravity_height(model: GravityModel, random_lla):
        lat, lon, _ = random_lla
        alt = 1000.0

        g_0 = model.calc_gravity_z(lat, lon, 0.0)
        g_1 = model.calc_gravity_z(lat, lon, alt)
        assert g_0 == g_1, "Gravity changed with height"

    @staticmethod
    def test_gravity_gradient(model: GravityModel, random_lla):
        g_grad = model.calc_vertical_grad(*random_lla)
        assert g_grad == 0.0, 'Gravity gradient should not change'


