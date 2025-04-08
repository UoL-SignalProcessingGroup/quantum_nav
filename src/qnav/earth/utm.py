

import numpy as np
import math

import qnav.util.constants as const

def utm2lla(x: float, y: float, zone: int) -> tuple[float, float]:
    """
    This function coverts from Universal Transverse Mercator (UTM) to
    Latitude-Longitude (LLA) coordinates. This function takes the X (eastern)
    and Y (northern) offset from a given UTM integer zone (1 to 60) and
    return a tuple containing the corresponding latitude-longitude
    coordinates, respectively.

    :param x: The X/Eastern offset from the given UTM zone (in metres).
    :type x: float

    :param y: The Y/Northern offset from the given UTM zone (in metres).
    :type y: float

    :param zone: The integer UTM zone (in range 1 t 60).
    :type zone: int

    :return: The given UTM coordinate converted to Latitude and Longitude
        coordinates (contained within a tuple).
    :rtype: tuple
    """

    if not 1 <= zone <= 60:
        raise ValueError("Zone must be in range 1 to 60")

    # d0 = 180 / math.pi  # Conversion rad to deg
    max_iterations = 100  # maximum iteration for latitude computation
    eps = 1e-11  # minimum residue for latitude computation

    utm_scale_factor = 0.9996  # UTM scale factor
    utm_false_east = 500000  # UTM false East (m)
    utm_false_north = 1e7 * (zone < 0)  # UTM false North (m)
    origin_latitude = 0  # UTM origin latitude (rad)
    origin_longitude = math.radians(6 * abs(zone) - 183)  # TM origin longitude (rad)

    # Ellipsoid eccentricity
    e1 = const.ECC
    n = utm_scale_factor * const.POLAR_AXIS_A

    # Compute parameters for Mercator Transverse projection
    c = __utm_coefficient(e1, 0)
    ys = utm_false_north - n * (
            c[0] * origin_latitude + c[1] * np.sin(2 * origin_latitude) +
            c[2] * np.sin(4 * origin_latitude) + c[3] * np.sin(6 * origin_latitude) +
            c[4] * np.sin(8 * origin_latitude)
    )

    c = __utm_coefficient(e1, 1)
    zt = complex((y - ys) / n / c[0], (x - utm_false_east) / n / c[0])
    z = zt - c[1] * np.sin(2 * zt) - c[2] * np.sin(4 * zt) - \
        c[3] * np.sin(6 * zt) - c[4] * np.sin(8 * zt)

    lon_rad = origin_longitude + np.arctan(np.sinh(np.imag(z)) / np.cos(np.real(z)))
    lat_rad = np.arcsin(np.sin(np.real(z)) / np.cosh(np.imag(z)))
    l = np.log(np.tan(np.pi / 4 + lat_rad / 2))

    # Calculate latitude from the isometric latitude
    lat_rad = 2 * np.arctan(np.exp(l)) - np.pi / 2
    origin_latitude = np.nan
    n = 0

    while (np.any(np.isnan(origin_latitude)) or np.abs(lat_rad - origin_latitude) > eps) and n < max_iterations:
        origin_latitude = lat_rad
        es = e1 * np.sin(origin_latitude)
        lat_rad = 2 * np.arctan(((1 + es) / (1 - es)) ** (e1 / 2) * np.exp(l)) - np.pi / 2
        n += 1

    # Convert the latitude and longitude to radians.
    lat = math.degrees(lat_rad)
    lon = math.degrees(lon_rad)

    # Finally, return the calculated coordinates.
    return lat, lon


def __utm_coefficient(e: float, m: float) -> np.ndarray:
    """
    This function calculate the projection coefficients for UTM to
    latitude-longitude conversion calculation.

    :param e: The first ellipsoid eccentricity
    :type e: float

    :param m: The mercator (0 for transverse mercator, 1 for transverse
        mercator reverse coefficients and 2 for merdian arc)
    :type m: float

    :return: 5 coefficients.
    :rtype: NumPy array
    """

    if m == 0:
        c0 = np.array([
            [-175/16384, 0, -5/256, 0, -3/64, 0, -1/4, 0, 1],
            [-105/4096, 0, -45/1024, 0, -3/32, 0, -3/8, 0, 0],
            [525/16384, 0, 45/1024, 0, 15/256, 0, 0, 0, 0],
            [-175/12288, 0, -35/3072, 0, 0, 0, 0, 0, 0],
            [315/131072, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    elif m == 1:
        c0 = np.array([
            [-175/16384, 0, -5/256, 0, -3/64, 0, -1/4, 0, 1],
            [1/61440, 0, 7/2048, 0, 1/48, 0, 1/8, 0, 0],
            [559/368640, 0, 3/1280, 0, 1/768, 0, 0, 0, 0],
            [283/430080, 0, 17/30720, 0, 0, 0, 0, 0, 0],
            [4397/41287680, 0, 0, 0, 0, 0,  0, 0, 0]
        ])

    else:
        c0 = np.array([
            [-175/16384, 0, -5/256, 0, -3/64, 0, -1/4, 0, 1],
            [-901/184320, 0, -9/1024, 0, -1/96, 0, 1/8, 0, 0],
            [-311/737280, 0, 17/5120, 0, 13/768, 0, 0, 0, 0],
            [899/430080, 0, 61/15360, 0, 0, 0, 0, 0, 0],
            [49561/41287680, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    c = np.zeros(c0.shape[0])
    for i in range(c0.shape[0]):
        c[i] = np.polyval(c0[i, :], e)

    return c


if __name__ == '__main__':
    [test_lat, test_lon] = utm2lla(500000.00, 5761038.21, 30)