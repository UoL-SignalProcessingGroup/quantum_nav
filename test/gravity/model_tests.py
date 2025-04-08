
import pytest
import numpy as np

from abc import ABC
from abc import abstractmethod
from typing import Optional

from qnav.earth.geoid import GeoidModel
from qnav.gravity.base import GravityModel

# The decimal point tolerance for comparing vectorised results.
# Used to overlook small negligible differences, likely caused
# by difference in order of operations etc.
_DP_TOL: int = 12

def assert_gravity_range(g: float):
    """
    Checks that given gravity acceleration is within expected range.
    Raises an assertion error if acceleration is lower than 9.7 or
    greater than 9.9.

    :param g: The vertical gravity acceleration.
    :type g: float
    """
    error_msg = 'Gravity acceleration is outside of expected range.'
    # assert 9.78032 <= g <= 9.83218, error_msg
    assert 9.7 <= g <= 9.9, error_msg


class ModelTests(ABC):
    """
    Abstract base class for conducting shared gravity model tests.
    This allows the tests to be defined once and used for each model.
    """

    @pytest.fixture(scope='class')
    def gravity_func(self, request) -> Optional[GravityModel]:
        """
        Returns the base gravity function to use for calculations.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        return None

    @pytest.fixture(scope='class')
    def geoid_model(self, request, project_dir) -> Optional[GeoidModel]:
        """
        Returns the optional geoid model for geoid height correction.

        :return: Initialised geoid model instance.
        :rtype: GeoidModel
        """
        return None

    # def model(self, gravity_func, geoid_model, tmp_gravity_dir) -> GravityModel:

    @abstractmethod
    @pytest.fixture(scope='class')
    def model(self, gravity_func, geoid_model, database_dir) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        pass

    @staticmethod
    def test_gravity_range(model: GravityModel, random_lla: tuple):
        """
        Asserts vertical gravity acceleration is within expected range.
        """
        lat, lon, alt = random_lla
        alt = 0
        g = model.calc_gravity_z(lat, lon, alt)
        assert_gravity_range(g)

    @staticmethod
    def test_gravity_height(model: GravityModel, random_lla: tuple):
        lat, lon, _ = random_lla
        alt = 1000.0

        g_0 = model.calc_gravity_z(lat, lon, 0.0)
        g_1 = model.calc_gravity_z(lat, lon, alt)
        assert g_0 > g_1, "Gravity not decreasing with height"

    @staticmethod
    def test_gravity_vector(model: GravityModel, random_lla: tuple):
        """
        Assets gravity vector has appropriate values.
        """
        g_vec = model.calc_gravity_xyz(*random_lla)
        assert g_vec.shape == (3,), 'Gravity vector is incorrect shape.'
        g_x, g_y, g_z = g_vec
        assert_gravity_range(g_z)

    @staticmethod
    def test_gravity_gradient(model: GravityModel, random_lla: tuple):
        g_grad = model.calc_vertical_grad(*random_lla)
        approx_grad = -3.08e-6
        margin = -approx_grad * 0.005
        lower_bound = approx_grad - margin
        upper_bound = approx_grad + margin
        assert lower_bound <= g_grad <= upper_bound, \
            'Gravity gradient exceeds expected bounds'

    @staticmethod
    def test_gradient_height(model: GravityModel, random_lla: tuple):
        lat, lon, _ = random_lla
        alt = 100.0
        g_grad_0 = model.calc_vertical_grad(lat, lon, 0.0)
        g_grad_1 = model.calc_vertical_grad(lat, lon, alt)
        assert g_grad_1 >= g_grad_0, "Gradient not decreasing with height"

    @staticmethod
    @pytest.mark.parametrize('random_lla_points', (1, 10, 10000), indirect=True)
    def test_gravity_vec(model: GravityModel, random_lla_points: tuple):

        # Calculate the values vectorised.
        lats, lons, alts = random_lla_points
        g_vec = model.calc_gravity_z_vec(lats, lons, alts)

        # Calculate the values as scalars:
        for i, (lat, lon, alt) in enumerate(zip(*random_lla_points)):
            g = model.calc_gravity_z(lat, lon, alt)
            np.testing.assert_almost_equal(g, g_vec[i], _DP_TOL)

    @staticmethod
    @pytest.mark.parametrize('random_lla_points', (1, 10, 10000), indirect=True)
    def test_vector_vec(model: GravityModel, random_lla_points: tuple):

        # Calculate the values vectorised.
        lats, lons, alts = random_lla_points
        g_xyz_vec = model.calc_gravity_xyz_vec(lats, lons, alts)

        # Calculate the values as scalars:
        for i, (lat, lon, alt) in enumerate(zip(*random_lla_points)):
            g_xyz = model.calc_gravity_xyz(lat, lon, alt)
            np.testing.assert_almost_equal(g_xyz, g_xyz_vec[i], _DP_TOL)

    @staticmethod
    @pytest.mark.parametrize('random_lla_points', (1, 10, 10000), indirect=True)
    def test_gradient_vec(model: GravityModel, random_lla_points: tuple):

        # Calculate the values vectorised.
        lats, lons, alts = random_lla_points
        dgz_dz_vec = model.calc_vertical_grad_vec(lats, lons, alts)

        # Calculate the values as scalars:
        for i, (lat, lon, alt) in enumerate(zip(*random_lla_points)):
            dgz_dz = model.calc_vertical_grad(lat, lon, alt)
            np.testing.assert_almost_equal(dgz_dz, dgz_dz_vec[i], _DP_TOL)
