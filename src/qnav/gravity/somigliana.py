"""
=============
somigliana.py
=============

:summary:
    Implementation of the Somigliana using WGS-84 parameters.
    This class contains gravity functions related to the Somigliana
    formula (with height correction).

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""


import math
import numpy as np
import qnav.util.constants as const

from qnav.gravity.base import GravityModel
from math import sqrt
from math import sin


class Somigliana(GravityModel):
    """
    Gravity function implementation of the Somigliana formula.
    This allows the calculation of the vertical gravity, gravitational
    acceleration and gravity gradients using the Somigliana formula
    (with altitude interpolation).
    """

    # No additional class properties.
    __slots__ = ()

    # First Somigliana height dependency constants K1
    __K1 = 2 * (1 + const.F + const.M) / const.POLAR_AXIS_A

    # Second Somigliana height dependency constants K2
    __K2 = 4 * (const.F / const.POLAR_AXIS_A)

    # Third Somigliana height dependency constants K3
    __K3 = 3 / (const.POLAR_AXIS_A ** 2)

    # Somigliana constant
    __P = ((const.POLAR_AXIS_B * const.G_P -
            const.POLAR_AXIS_A * const.G_E)
           / (const.POLAR_AXIS_A * const.G_E))

    def __str__(self) -> str:
        """
        Returns the name of the gravity map. Used for describing the model.

        :return: The identifying name of the gravity model.
        :rtype: str
        """
        return "Somigliana"

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

        p = self.__P
        k1 = self.__K1
        k2 = self.__K2
        k3 = self.__K3

        ge = const.G_E
        ecc_sq = const.ECC * const.ECC
        s_lat_2 = sin(math.radians(lat)) ** 2

        gz = ge * (1.0 + p * s_lat_2) / sqrt(1.0 - ecc_sq * s_lat_2)
        gz *= (1 - (k1 - k2 * s_lat_2) * alt + k3 * alt * alt)
        return gz

    def calc_gravity_xyz(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Calculates and returns the gravitational acceleration for the
        X/North, Y/East and Z/Down axes (in m/s^2) for the given location
        and altitude offset.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :param alt: The altitude offset for the position of interest.
            The elevation from the WGS-84 reference ellipsoid in metres.
        :type alt: float

        :return: The local gravity vector for the XYZ components (in m/s^2).
        :rtype: numpy.ndarray (3-elements)
        """
        gx = -8.08e-09 * abs(alt) * sin(2.0 * math.radians(lat))
        gz = self.calc_gravity_z(lat, lon, alt)
        return np.array([gx, 0, gz])

    # def calc_gravity_grad(self, lat: float, lon: float, alt: float, step_size: float = 10) -> np.ndarray:
    #     """
    #     Calculates and returns the gravitational gradient tensor of the
    #     X/North, Y/East and Z/Down components for the given location
    #     and altitude offset. These are flattened into a 1D array.
    #
    #     :param lat: The latitude coordinate for the position of interest.
    #         The latitude position value given in decimal degrees.
    #     :type lat: float
    #
    #     :param lon: The longitude coordinate for the position of interest.
    #         The longitude position value given in decimal degrees.
    #     :type lon: float
    #
    #     :param alt: The altitude offset for the position of interest.
    #         The elevation from the WGS-84 reference ellipsoid in metres.
    #     :type alt: float
    #
    #     :param step_size: The step size used in the X and Y directions for
    #         calculating the change in gravity (in metres). Default is 10.
    #     :type alt: step_size
    #
    #     :return: The gravity gradients of the XYZ components. Specifically,
    #         an array containing the
    #     :rtype: numpy.ndarray (9-elements)
    #     """
    #     raise NotImplementedError("Awaiting confirmation on gravity equations")

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

        p = self.__P
        k1 = self.__K1
        k2 = self.__K2
        k3 = self.__K3

        ge = const.G_E
        ecc_sq = const.ECC * const.ECC
        s_lat_2 = np.sin(np.radians(lat)) ** 2

        gz = ge * (1.0 + p * s_lat_2) / np.sqrt(1.0 - ecc_sq * s_lat_2)
        gz *= (1 - (k1 - k2 * s_lat_2) * alt + k3 * alt * alt)
        return gz

    def calc_gravity_xyz_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        """
        In a vectorised batch, calculates and returns the gravitational
        acceleration for the X/North, Y/East and Z/Down axes (in m/s^2)
        for the given location and altitude offset. Ideal when calculating
        gravity for a large number of positions.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: np.array

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: np.array

        :param alt: The altitude offset for the position of interest.
            The elevation from the WGS-84 reference ellipsoid in metres.
        :type alt: np.array

        :return: The local gravity vector for the XYZ components (in m/s^2).
        :rtype: numpy.ndarray (N-by-3 elements)
        """

        # Simple wrapper for methods that do not support vectorisation.
        lat_rad = np.radians(lat)
        gx = -8.08e-09 * np.abs(alt) * np.sin(2.0 * lat_rad)
        gz = self.calc_gravity_z_vec(lat, lon, alt)

        g_xyz = np.zeros((len(lat), 3))
        g_xyz[:, 0] = gx
        g_xyz[:, 2] = gz

        return g_xyz


if __name__ == '__main__':

    som = Somigliana()
    g0 = som.calc_gravity_z(52, -3, 0)
    g1 = som.calc_gravity_z(52, -3, 100)
    gzz = som.calc_vertical_grad(52, -3, 0)

    # steps = np.arange(-180, 180, 0.1)
    #
    # for i in steps:
    #     print(som.calc_gravity_xyz(0, i, 1000.0))
    #
    # tmp = np.zeros_like(steps)
    # test = som.calc_gravity_xyz_vec(tmp, steps, tmp + 1000)

