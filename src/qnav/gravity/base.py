"""
=======
base.py
=======

:summary:
    Gravity model template class interfaces that supported gravity models
    must extend from. The purpose of this is to define the functions, input
    arguments and outputs that gravity models must incorporate to be used
    in the toolbox. This also simplifies the ability to add new gravity
    models future.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""
import math
from abc import abstractmethod
from abc import ABC
from typing import Self

import qnav.util.constants as const
import numpy as np


class GravityModel(ABC):
    """
    The abstract class that gravity models are to follow.
    This defines the interface that all gravity models in the project must
    follow, outlining the necessary functions they must supply for
    calculating various gravity components at requested positions. For a
    gravity model to be used by the toolbox, it must correctly extend from
    this abstract base class.

    In short, each gravity model must supply the following methods:
        • __str__ - The descriptive name for the gravity model.
        • calc_gravity_z - Calculate downwards gravitational acceleration.
        • calc_gravity_xyz - Calculate gravitational acceleration vector.
        • calc_gravity_grad - Calculate gravitational gradient tensor.

    In addition to this, there are also optional methods they can supply:
        • calc_gravity_z_vec - Vectorised downwards gravitational acceleration.
        • calc_gravity_xyz_vec - Vectorised gravitational acceleration vector.
        • calc_gravity_grad_vec - Vectorised gravitational gradient tensor.

    These are used to efficiently calculate gravity for a large range of
    positions. To enforce compatibility, if not implemented, calling these
    methods will execute the scalar corresponding methods in a loop.
    """

    __slots__ = ()

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns an identifiable descriptive name for the gravity model.

        :return: A descriptive string for the gravity model.
        :rtype: str
        """
        pass

    @abstractmethod
    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:
        """
        Calculates and returns the total gravitational acceleration.
        Computes the gravity along the Z/Down axis (in m/s^2) for
        the given location and altitude offset.

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
        pass

    def calc_gravity_xyz(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        Calculates and returns the gravitational acceleration vector.
        Computes the gravity acceleration X/North, Y/East and Z/Down
        components (in m/s^2) for the given location and altitude offset.

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
        # g_z = self.calc_gravity_z(lat, lon, alt)
        # return apply_coriolis_terms(lat, g_z, alt)
        g_z = self.calc_gravity_z(lat, lon, alt)
        return np.array([0.0, 0.0, g_z])

    def calc_vertical_grad(self, lat: float, lon: float,
                           alt: float, step: float = 100) -> float:
        """
        Calculates the vertical gradient of gravity.
        Computes the vertical change in gravity (∂g/∂r) of the Z/Down
        component (in m/s^2). This is primary used for gravity map-matching
        related calculations.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :param alt: The altitude offset for the position of interest.
            The elevation from the WGS-84 reference ellipsoid in metres.
        :type alt: float

        :param step: The step size (in metres) for calculating the gradient.
            To be used when the gradient cannot be computed directly.
        :type step: float

        :return: The vertical gravity gradient (in m/s^2).
        :rtype: float
        """
        half_step = step / 2
        gz_0 = self.calc_gravity_z(lat, lon, alt - half_step)
        gz_1 = self.calc_gravity_z(lat, lon, alt + half_step)
        return (gz_1 - gz_0) / step


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
    #
    #     # Calculate the gravity for the current location.
    #     ref_lla = np.array((lat, lon, alt))
    #     g_ref = self.calc_gravity_xyz(lat, lon, alt)
    #
    #     # Calculate the gravity for an offset in the X direction.
    #     ned_x = np.array((step_size, 0, 0))
    #     lla_x = trans.ned2lla(ned_x, ref_lla)
    #     g_dx = self.calc_gravity_xyz(*lla_x)
    #
    #     # Calculate the gravity for an offset in the X direction.
    #     ned_x = np.array((0, step_size, 0))
    #     lla_y = trans.ned2lla(ned_x, ref_lla)
    #     g_dy = self.calc_gravity_xyz(*lla_y)
    #
    #     # Obtain the gradient in the X and Y directions.
    #     dh_dx = (g_dx - g_ref) / step_size
    #     dh_dy = (g_dy - g_ref) / step_size
    #
    #     # Obtain the radius of the earth for the given latitude.
    #     r_e = trans.radius(lat)
    #
    #     # Define Angular velocity for Earth's rotation (in local NED axes).
    #     omega_e = trans.rotation_rate(lat)
    #     radii_offset = np.array([0, 0, -r_e - alt])
    #     rotation_array = np.array([0, 0, -1])
    #
    #     g = self.calc_gravity_xyz(lat, lon, alt)
    #     dg_norm_dh = - g * (2 * r_e ** 2 / ((r_e + alt) ** 3)) \
    #                  - 2 * np.cross(omega_e, radii_offset) \
    #                  - np.cross(omega_e, np.cross(omega_e, rotation_array))
    #
    #     # Obtain the gradient for the Z directions.
    #     dh_dz = np.array([
    #         dh_dx[2],
    #         dh_dy[2],
    #         -(dh_dx[0] + dh_dy[1]) - dg_norm_dh[2]
    #     ])
    #
    #     # Finally format and return the gravity gradients.
    #     return np.concatenate([dh_dx, dh_dy, dh_dz])

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

        # Simple wrapper for methods that do not support vectorisation.
        g_z = [self.calc_gravity_z(*pos) for pos in zip(lat, lon, alt)]
        return np.array(g_z)

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
        g_xyz = [self.calc_gravity_xyz(*pos) for pos in zip(lat, lon, alt)]
        return np.stack(g_xyz)

    def calc_vertical_grad_vec(self, lat: np.ndarray, lon: np.ndarray,
                               alt: np.ndarray, step: float = 10) -> np.ndarray:
        """
        In a vectorised batch, calculates the vertical gradient of gravity.
        Computes the vertical change in gravity (∂g/∂r) of the Z/Down
        component (in m/s^2). This is primary used for gravity map-matching
        related calculations.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: np.ndarray

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: np.ndarray

        :param alt: The altitude offset for the position of interest.
            The elevation from the WGS-84 reference ellipsoid in metres.
        :type alt: np.ndarray

        :param step: The step size (in metres) for calculating the gradient.
            To be used when the gradient cannot be computed directly.
        :type step: float

        :return: The vertical gravity gradient (in m/s^2).
        :rtype: float
        """
        half_step = step / 2
        gz_0 = self.calc_gravity_z_vec(lat, lon, alt - half_step)
        gz_1 = self.calc_gravity_z_vec(lat, lon, alt + half_step)
        return (gz_1 - gz_0) / step

    # def calc_gravity_grad_vec(self, lat: np.ndarray, lon: np.ndarray,
    #                           alt: np.ndarray, step_size: float = 10) -> np.ndarray:
    #     """
    #     In a vectorised batch, calculates and returns the gravitational
    #     gradient tensor of the X/North, Y/East and Z/Down components
    #     for the given location and altitude offset. These are flattened
    #     into a 1D array. Ideal when calculating gravity for a large number
    #     of positions.
    #
    #     :param lat: The latitude coordinate for the position of interest.
    #         The latitude position value given in decimal degrees.
    #     :type lat: np.array
    #
    #     :param lon: The longitude coordinate for the position of interest.
    #         The longitude position value given in decimal degrees.
    #     :type lon: np.array
    #
    #     :param alt: The altitude offset for the position of interest.
    #         The elevation from the WGS-84 reference ellipsoid in metres.
    #     :type alt: np.array
    #
    #     :param step_size: The step size used in the X and Y directions for
    #         calculating the change in gravity (in metres). Default is 10.
    #     :type alt: step_size
    #
    #     :return: The local gravity vector for the XYZ components (in m/s^2).
    #     :rtype: numpy.ndarray (N-by-3-by-3 elements)
    #     """
    #
    #     # Simple wrapper for methods that do not support vectorisation.
    #     g_grad = [self.calc_gravity_grad(*pos) for pos in zip(lat, lon, alt)]
    #     return np.stack(g_grad)

    @property
    def base_model(self) -> Self | None:
        """
        Returns the base gravity model the model extends from.
        Allows access to the base function/model the instance is
        using (if any).

        :return: Either the base gravity model or None.
        :rtype: GravityModel | None
        """
        return None


def calculate_curvature(lat: float | np.ndarray) -> float | np.ndarray:
    """
    Calculates the mean curvature of the ellipsoid J at given position.
    Using equations from Physical Geodesy 2006, this function calculates
    J for given positions.

    :param lat: The latitude of the point(s) of interest (in decimal degrees).
    :type lat: float | np.ndarray

    :return: The mean curvature of the ellipsoid.
    :rtype: float | np.ndarray
    """

    e_prime = const.ECC_PRIME
    e_prime_sq = e_prime * e_prime

    a = const.POLAR_AXIS_A
    b = const.POLAR_AXIS_B
    c = (a * a) / b

    # Calculate variables M and N using equation (2–149)
    if np.isscalar(lat):
        lat_rad = math.radians(lat)
        tmp = (1 + e_prime_sq * math.cos(lat_rad * lat_rad))
        m = c / (math.pow(tmp, 3 / 2))
        n = c / (math.pow(tmp, 1 / 2))
    else:
        lat_rad = np.radians(lat)
        tmp = (1 + e_prime_sq * np.cos(lat_rad * lat_rad))
        m = c / (np.pow(tmp, 3 / 2))
        n = c / (np.pow(tmp, 1 / 2))

    # Calculate and return J using equation (2–148)
    return (1 / m + 1 / n) * 0.5






