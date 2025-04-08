"""
=======
nima.py
=======

:summary:
    National Imagery and Mapping Agency (NIMA) gravity functions.
    This module contains all the gravity calculation methods published by
    NIMA, specifically those stated in “Department of Defense World
    Geodetic System 1984: Its Definition, and Relationship with Local
    Geodetic Systems, TR8350.2, Third Ed.” Department of Defense,
    Washington, DC: 1997.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

import math
import numpy as np
import qnav.util.constants as const

from qnav.gravity.base import GravityModel


class WGS84Gravity(GravityModel):
    """
    Implementation of the WGS84 representation of Earth gravity.
    This model uses the equations published by NIMA. This is known to be
    a relatively accurate close approximation model for heights up to
    20,000 meters above the WGS84 reference Ellipsoid.
    """

    def __str__(self) -> str:
        """
        Returns an identifiable descriptive name for the gravity model.

        :return: A descriptive string for the gravity model.
        :rtype: str
        """
        return "WGS84 Gravity Model"

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

        # [Equation 4.24] Compute the magnitude of the total normal gravity vector.
        gamma_phi, _, gamma_h = self.calc_gravity_xyz(lat, lon, alt)
        return math.sqrt((gamma_h * gamma_h) + (gamma_phi * gamma_phi))

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

        # Convert to radians.
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        # Commonly used sin and cos for geodetic latitude.
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin2lat = sin_lat * sin_lat

        # Commonly used sin and cos for geodetic longitude.
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        # Get Geocentric latitude (psi)
        tan_lat = sin_lat / cos_lat
        psi = math.atan(tan_lat * (1 - const.F) * (1 - const.F))
        sin_psi = math.sin(psi)
        cos_psi = math.cos(psi)

        # [Equation 4-20]
        alpha = lat_rad - psi
        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)

        # [Equation 4-15] Radius of curvature in the prime vertical (N).
        n = const.POLAR_AXIS_A / math.sqrt(1 - const.ECC_SQ * sin2lat)

        # [Equation 4-14] The rectangular XYZ coordinates from known geodetic coordinates.
        b_over_a = const.POLAR_AXIS_B / const.POLAR_AXIS_A
        x = (n + alt) * cos_lat * cos_lon
        y = (n + alt) * cos_lat * sin_lon
        z = (b_over_a * b_over_a * n + alt) * sin_lat
        # x, y, z = trans.lla2ecef(np.array([lat, lon, alt]))

        # [Equation 4-7]
        a2 = const.POLAR_AXIS_A * const.POLAR_AXIS_A
        b2 = const.POLAR_AXIS_B * const.POLAR_AXIS_B
        e2 = a2 - b2
        e = math.sqrt(e2)

        # [Equation 4-8] Calculate the first ellipsoidal coordinate (u).
        tmp = (x * x) + (y * y) + (z * z) - e2
        u2 = 0.5 * tmp * (1 + math.sqrt(1 + 4 * e2 * (z * z) / (tmp * tmp)))
        u = math.sqrt(u2)

        # [Equation 4-9] Calculate the second ellipsoidal coordinate (beta).
        u2_e2 = u2 + e2
        sqrt_u2_e2 = math.sqrt(u2_e2)
        beta = math.atan(z * sqrt_u2_e2 / (u * math.sqrt((x * x) + (y * y))))

        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)
        sin2beta = sin_beta * sin_beta
        cos2beta = cos_beta * cos_beta

        # [Equation 4-10]
        w = math.sqrt((u2 + e2 * sin2beta) / (u2 + e2))

        # [Equation 4-11]
        q = 0.5 * ((1 + 3 * u2 / e2) * math.atan(e / u) - 3 * u / e)

        # [Equation 4-12]
        qo = 0.5 * ((1 + 3 * b2 / e2) * math.atan(e / const.POLAR_AXIS_B) - 3 * const.POLAR_AXIS_B / e)

        # [Equation 4-13]
        q_prime = 3 * ((1 + u2 / e2) * (1 - (u / e) * math.atan(e / u))) - 1

        # Calculate the applied Centrifugal force for the ellipsoidal components.
        omega_e2 = const.OMEGA_E * const.OMEGA_E
        cf_u = u * cos2beta * omega_e2 / w
        cf_beta = sqrt_u2_e2 * cos_beta * sin_beta * omega_e2 / w

        # [Equation 4-5] Calculate the first ellipsoidal component of the normal gravity vector.
        gamma_u = -(const.GM / u2_e2 + omega_e2 * a2 * e * q_prime *
                    (0.5 * sin2beta - 1.0 / 6.0) / (u2_e2 * qo)) / w + cf_u

        # [Equation 4-6] Calculate the second ellipsoidal component of the normal gravity vector.
        gamma_beta = omega_e2 * a2 * q * sin_beta * cos_beta / (sqrt_u2_e2 * w * qo) - cf_beta

        # [Equation 4-17] (Optimised: R2 x R1 x Gamma_E)
        tmp = u / (w * sqrt_u2_e2)
        gamma_r = ((cos_beta * cos_psi * tmp + sin_beta * sin_psi / w) * gamma_u
                   + (sin_psi * cos_beta * tmp - sin_beta * cos_psi / w) * gamma_beta)
        gamma_psi = ((sin_beta * cos_psi / w - sin_psi * cos_beta * tmp) * gamma_u
                     + (cos_beta * cos_psi * tmp + sin_beta * sin_psi / w) * gamma_beta)

        # [Equation 4-16] Get "normal" gravity (Down-Direction)
        gamma_h = (-gamma_r) * cos_alpha - gamma_psi * sin_alpha

        # [Equation 4-23] Get "tangent" gravity (North-Direction)
        gamma_phi = (-gamma_r) * sin_alpha + gamma_psi * cos_alpha

        # Finally, format and return the output
        return np.array([gamma_phi, 0, gamma_h])


if __name__ == '__main__':
    wgs = WGS84Gravity()
    g_vec = wgs.calc_gravity_xyz(90, 0, 1000)
    print(g_vec)
