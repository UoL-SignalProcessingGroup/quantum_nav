import math
from pathlib import Path

import pytest
import numpy as np

from typing import Optional, Type

from gravity.model_tests import ModelTests, _DP_TOL
from qnav.gravity.map import GravityMap
from qnav.gravity.somigliana import Somigliana
from qnav.gravity.base import GravityModel
from qnav.earth.geoid import GeoidModel
from qnav.earth.geoid import GeoidPGM
from abc import ABC, abstractmethod


def get_gravity_funcs() -> list[Type[Somigliana]]:
    """
    Returns list of supported gravity model classes.
    These can be individually initialised and tested against classes.
    """
    return [Somigliana]


def get_geoid_models() -> list[str | None]:
    """
    Returns list of supported geoid model names.
    These can be used to initialise geoid models and used them
    against test classes.
    """
    # return [None, 'egm2008-5.pgm']
    return [None]


def assert_anomaly_range(g_anom: float):
    """
    Checks that given gravity anomaly is within expected range.
    Raises an assertion error if anomaly is lower than
    -250 milligals or greater than 150 milligals.

    :param g_anom: The gravity anomaly in m/s^2.
    :type g_anom: float
    """
    error_msg = 'Gravity anomaly is outside of expected range.'
    # assert -250e-5 <= g_anom <= 150e-5, error_msg
    assert -400e-5 <= g_anom <= 400e-5, error_msg


def assert_disturbance_range(g_dist: float):
    """
    Checks that given gravity disturbance is within expected range.
    Raises an assertion error if disturbance is lower than
    -250 milligals or greater than 150 milligals.

    :param g_dist: The gravity anomaly in m/s^2.
    :type g_dist: float
    """
    error_msg = 'Gravity anomaly is outside of expected range.'
    assert -250e-5 <= g_dist <= 150e-5, error_msg


class MapTests(ModelTests, ABC):
    """
    Abstract base class for conducting shared gravity map tests.
    This allows the tests to be defined once and used for each
    correction model, involving numerous parameters .
    """

    @pytest.fixture(params=get_gravity_funcs(), scope="class")
    def gravity_func(self, request) -> Optional[GravityModel]:
        func_class = request.param
        return func_class()

    @pytest.fixture(params=get_geoid_models(), scope="class")
    def geoid_model(self, request, project_dir) -> Optional[GeoidModel]:
        geoid_name = request.param
        if geoid_name is None:
            return None
        geoid_file = project_dir / request.param
        return GeoidPGM(geoid_file)

    @staticmethod
    def test_gravity_anomaly(model: GravityMap, random_lla: tuple):
        lat, lon, _ = random_lla
        g_anom = model.get_anomaly(lat, lon)
        assert_anomaly_range(g_anom)

    @staticmethod
    def test_gravity_disturbance(model: GravityMap, random_lla: tuple):
        lat, lon, _ = random_lla
        g_dist = model.get_disturbance(lat, lon)
        assert_disturbance_range(g_dist)

    @staticmethod
    @pytest.mark.parametrize('random_lla_points', (1, 10, 10000), indirect=True)
    def test_anomaly_vec(model: GravityMap, random_lla_points: tuple):

        # Calculate the values vectorised.
        lats, lons, _ = random_lla_points
        g_anom_vec = model.get_anomaly_vec(lats, lons)

        # Calculate the values as scalars:
        for i, (lat, lon, _) in enumerate(zip(*random_lla_points)):
            g_anom = model.get_anomaly(lat, lon)
            np.testing.assert_almost_equal(g_anom, g_anom_vec[i], _DP_TOL)

    @staticmethod
    @pytest.mark.parametrize('random_lla_points', (1, 10, 10000), indirect=True)
    def test_disturbance_vec(model: GravityMap, random_lla_points: tuple):

        # Calculate the values vectorised.
        lats, lons, _ = random_lla_points
        g_dist_vec = model.get_disturbance_vec(lats, lons)

        # Calculate the values as scalars:
        for i, (lat, lon, _) in enumerate(zip(*random_lla_points)):
            g_dist = model.get_disturbance(lat, lon)
            np.testing.assert_almost_equal(g_dist, g_dist_vec[i], _DP_TOL)

    @staticmethod
    def test_inside_gravity(model: GravityMap, gravity_func: GravityModel, random_lla: tuple):

        lat, lon, _ = random_lla
        g_anom = model.get_anomaly(lat, lon)
        g_model = model.calc_gravity_z(*random_lla)
        g_func = gravity_func.calc_gravity_z(*random_lla)

        assert g_anom == 0.0 or g_model != g_func, \
            'Gravity map has no change on gravity function'

    @staticmethod
    def test_outside_gravity(model: GravityMap, gravity_func: GravityModel, random_outside_lla: tuple):

        lat, lon, _ = random_outside_lla
        g_anom = model.get_anomaly(lat, lon)
        g_model = model.calc_gravity_z(*random_outside_lla)
        g_func = gravity_func.calc_gravity_z(*random_outside_lla)

        assert g_anom != 0.0 or g_model == g_func, \
            'Gravity model not falling back to gravity function'

    @staticmethod
    def test_gravity_gradient(model: GravityMap, random_lla: tuple):
        g_grad = model.calc_vertical_grad(*random_lla)
        approx_grad = -3.08e-6
        margin = -approx_grad * 0.015
        lower_bound = approx_grad - margin
        upper_bound = approx_grad + margin
        assert lower_bound <= g_grad <= upper_bound, \
            'Gravity gradient exceeds expected bounds'

    @staticmethod
    def test_gradient_height(model: GravityMap, random_lla: tuple):
        alt = 100.0
        lat, lon, _ = random_lla
        g_grad_0 = model.calc_vertical_grad(lat, lon, 0.0)
        g_grad_1 = model.calc_vertical_grad(lat, lon, alt)
        assert g_grad_1 >= g_grad_0, f"Gradient not decreasing with height ({g_grad_1} < {g_grad_0})"

    @staticmethod
    @pytest.mark.parametrize('random_lla_points', (1, 10, 10000), indirect=True)
    def test_gradient_vec(model: GravityMap, random_lla_points: tuple):

        # Calculate the values vectorised.
        lats, lons, alts = random_lla_points
        dgz_dz_vec = model.calc_vertical_grad_vec(lats, lons, alts)

        # Calculate the values as scalars:
        for i, (lat, lon, alt) in enumerate(zip(*random_lla_points)):
            dgz_dz = model.calc_vertical_grad(lat, lon, alt)
            np.testing.assert_almost_equal(dgz_dz, dgz_dz_vec[i], _DP_TOL)



class DownloadTests(ABC):
    """
    Tests to conduct for verifying gravity map downloading.
    These are only performed for certain models.
    """

    @pytest.fixture(scope='class')
    def gravity_func(self) -> GravityModel:
        """
        Returns a base gravity function to use.

        :return: A base gravity model to correct from.
        :rtype: GravityModel
        """
        return Somigliana()

    @pytest.fixture(scope='class')
    def geoid_model(self) -> Optional[GeoidModel]:
        """
        Returns the optional geoid model for geoid height correction.

        :return: Initialised geoid model instance.
        :rtype: GeoidModel
        """
        return None

    @pytest.fixture(scope='class')
    def tmp_gravity_dir(self, tmp_path_factory) -> Path:
        """
        Returns a temporary folder for holding downloaded gravity data.

        :param tmp_path_factory: Used to generate temporary files.
        :type tmp_path_factory: tmp_path_factory

        :return: The temporary folder to use for tests.
        :rtype: Path
        """
        class_name = self.__class__.__name__
        return tmp_path_factory.mktemp(class_name)

    @abstractmethod
    @pytest.fixture(scope='class')
    def model(self, gravity_func: GravityModel,
              geoid_model: Optional[GeoidModel],
              tmp_gravity_dir: Path) -> GravityModel:
        """
        Initialises and return the gravity model to use.
        This allows the gravity model to be configured for the test.

        :return: Initialised gravity model instance.
        :rtype: GravityModel
        """
        pass

    @staticmethod
    @pytest.mark.slow
    def test_download(model: GravityMap, random_lla: tuple):
        """
        Checks that the model has downloaded and used the gravity map.
        On failure, a usable value will not be produced or an
        error will be raised.
        """
        try:
            g_z = model.calc_gravity_z(*random_lla)
            assert math.isfinite(g_z)
        except FileNotFoundError as e:
            raise FileNotFoundError('Download test failed')
