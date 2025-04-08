
import pytest

from gravity.model_tests import ModelTests
from qnav.gravity.base import GravityModel
from qnav.gravity.somigliana import Somigliana


class TestSomigliana(ModelTests):
    """
    Performs tests using Somigliana gravity model.
    """

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        return Somigliana()
