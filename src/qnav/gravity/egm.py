"""
======
egm.py
======

:summary:
    A module for supplying Earth Gravitational Models (EGMs).
    This provides implementations for EGM gravity models, using a standard
    'base' gravity function and a geoid model for height and surface
    deflection correction.

:authors:
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of this module.
"""


import numpy as np

from qnav.earth.geoid import GeoidModel
from qnav.gravity.map import GravityMap
from qnav.gravity.base import GravityModel
from qnav.util.transformations import radius, radius_vec


class GeoidCorrection(GravityMap):
    """
    Gravity Map using Geoid Model for applying acceleration corrections.
    Using a given geoid model to applying corrections to the gravity
    accelerations, using the height offsets and deflections of vertical.
    """

    def __init__(self, base_model: GravityModel, geoid_model: GeoidModel):
        """
        Creates a gravity map with given base function and geoid model.

        :param base_model: The base gravity model to use for initial gravity
            values. The values produced by this will be corrected.
        :type base_model: GravityModel

        :param geoid_model: The geoid model to use for applying corrections.
        :type geoid_model: GeoidModel
        """
        super().__init__(base_model, geoid_model, None)
        self._map_name = str(geoid_model).upper()

    def __str__(self) -> str:
        """
        Produces a string representation of the gravity map.
        Defines the map name followed by 'Gravity'.

        :return: A description of the gravity map.
        :rtype: str
        """
        return self._map_name

    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:
        """
        Calculates and returns the gravitational acceleration along the Z/Down
        axis (in m/s^2) for the given location and altitude offset.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :param alt: The altitude offset for the position of interest.
            The elevation from the WGS-84 reference ellipsoid in metres.
        :type alt: float

        :return: The downwards gravitational acceleration for each location.
        :rtype: float
        """
        return self._base_model.calc_gravity_z(lat, lon, alt)

    def calc_gravity_z_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        """
        In a vectorised batch, calculates and returns the gravitational
        acceleration along the Z/Down axis (in m/s^2) for the given location
        and altitude offset. Ideal when calculating gravity for a large
        number of positions.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: np.array

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: np.array

        :param alt: The altitude offset for the position of interest.
            The elevation from the WGS-84 reference ellipsoid in metres.
        :type alt: np.array

        :return: The downwards gravitational acceleration for each location
            (in m/s^2).
        :rtype: np.array (N-elements)
        """
        return self._base_model.calc_gravity_z_vec(lat, lon, alt)

    def get_anomaly(self, lat: float, lon: float) -> float:
        r = radius(lat)
        geoid_height = self._geoid_model.get_height(lat, lon)
        gamma = self._base_model.calc_gravity_z(lat, lon, 0)
        d_gamma_d_h = self._base_model.calc_vertical_grad(lat, lon, 0)
        return -geoid_height * d_gamma_d_h - 2 * ((gamma * geoid_height) / r)

    def get_anomaly_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        r = radius_vec(lat)
        alt = np.zeros_like(lat)
        geoid_height = self._geoid_model.get_height_vec(lat, lon)
        gamma = self._base_model.calc_gravity_z_vec(lat, lon, alt)
        d_gamma_d_h = self._base_model.calc_vertical_grad_vec(lat, lon, alt)
        return -geoid_height * d_gamma_d_h - 2 * ((gamma * geoid_height) / r)

    def get_disturbance(self, lat: float, lon: float) -> float:
        anomaly = self.get_anomaly(lat, lon)
        return self._anomaly_to_disturbance(lat, lon, anomaly)

    def get_disturbance_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        anomaly = self.get_anomaly_vec(lat, lon)
        return self._anomaly_to_disturbance_vec(lat, lon, anomaly)








if __name__ == '__main__':

    from pathlib import Path
    from qnav.earth.geoid import GeoidPGM
    from qnav.gravity.somigliana import Somigliana

    project_dir = Path(__file__).parent.parent.parent
    test_pgm = project_dir / 'databases' / 'geoid' / 'egm2008-5.pgm'
    test_geoid = GeoidPGM(test_pgm)

    som = Somigliana()
    test_egm = GeoidCorrection(som, test_geoid)

    test_g = test_egm.calc_gravity_z(52, -3, 1000)
    test_xyz = test_egm.calc_gravity_xyz(52, -3, 1000)
