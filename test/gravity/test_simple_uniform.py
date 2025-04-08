
import pytest

from gravity.test_fixed_value import TestFixedValue
from qnav.gravity.base import GravityModel
from qnav.gravity.simple import SimpleUniform


class TestSimpleUniform(TestFixedValue):
    """
    Performs tests using Simple Uniform gravity model.
    Inherits changes for test performed by fixed values.
    """

    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        return SimpleUniform()
