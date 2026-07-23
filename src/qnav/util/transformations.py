"""
==================
transformations.py
==================

:summary:
    This module includes all general transformation calculations that
    are required for the Navigation Simulation Study. The output from the
    presented functions has been compared with other implementations such as
    Python's NavPy and MATLAB's mapping toolbox.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of this module.
"""

import qnav.util.constants as constants
import numpy.typing as npt
import datetime as dt
import numpy as np
import math

# from qnav.util.constants import OMEGA_E

# The average radius of the Earth (m).
_R_E = constants.R_E

# The constant WGS84 flattening value used.
_F = constants.F

# Semi-major axis (Equator radius) in metres.
_A = constants.POLAR_AXIS_A

# Semi-minor axis (Pole radius) in metres.
_B = constants.POLAR_AXIS_B  # B = A * (1 - F)

# The first eccentricity value.
_ECC = constants.ECC

# The second eccentricity value.
_ECC_PRIME = constants.ECC_PRIME

# Approximate rotation rate of the Earth (m/s**2).
_OMEGA_E = constants.OMEGA_E

# A very small value used to prevent rounding errors.
_SMALL_VALUE = 1E-9




def deg2rad(theta_deg: float) -> float:
    """
    A wrapper function for converting from degrees to radians.

    :param theta_deg: An input value in degrees.
    :type theta_deg: float

    :return: The converted input value in radians.
    :rtype: float
    """

    return math.radians(theta_deg)  # return theta_deg * np.pi / 180


def rad2deg(theta_rad: float) -> float:
    """
    A wrapper function for converting from radians to degrees.

    :param theta_rad: An input value in radians.
    :type theta_rad: float

    :return: The converted input value in degrees.
    :rtype: float
    """

    return math.degrees(theta_rad)  # return theta_rad * 180 / np.pi


def deg2dec(degrees: float, minutes: float, seconds: float) -> float:
    """
    Converts a degrees, minutes, seconds coordinate to decimal format.

    :param degrees: The degrees of the given coordinate.
    :type degrees: float

    :param minutes: The minutes of the given coordinate.
    :type minutes: float

    :param seconds: The seconds of the given coordinate.
    :type seconds: float

    :return: The decimal equivalent of the given coordinate.
    :rtype: float
    """
    return degrees + (minutes / 60) + (seconds / 3600)


def dec2deg(theta_deg: float) -> tuple[float, float, float]:
    """
    Converts a decimal coordinate to degrees, minutes, seconds format.

    :param theta_deg: The decimal format of the given coordinate.
    :type theta_deg: float or NumPy Array

    :return: The degrees, minutes and seconds of the given coordinate.
    :rtype: float or NumPy Array
    """
    degrees = math.floor(theta_deg)
    minutes = math.floor(60 * (theta_deg - degrees))
    seconds = 3600 * (theta_deg - degrees) - 60 * minutes
    return degrees, minutes, seconds


def radius(lat: float) -> float:
    """
    This function calculates the radius (in metres) from the centre of the
    Earth for a given latitude position, using WGS84 parameters.

    :param lat: The latitude of the points of interest.
    :type lat: float

    :return: The calculated radius in metres for the given position.
    :rtype: float
    """

    lat_rad = math.radians(lat)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    return math.sqrt(
        (((_A ** 2) * cos_lat) ** 2 + (_B ** 2 * sin_lat) ** 2) /
        ((_A * cos_lat) ** 2 + (_B * sin_lat) ** 2)
    )


def radius_vec(lat: np.ndarray) -> np.ndarray:
    """
    The vectorised version of the
    :func:`~qnav.util.transformations.radius` function. This
    calculates the radius (in metres) from the centre of the Earth for a large
    series of given latitude positions, using WGS84 parameters.

    :param lat: The latitude of the points of interest.
    :type lat: numpy.ndarray

    :return: The calculated radius in metres for the given position.
    :rtype: numpy.ndarray
    """

    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)

    return np.sqrt(
        (((_A ** 2) * cos_lat) ** 2 + (_B ** 2 * sin_lat) ** 2) /
        ((_A * cos_lat) ** 2 + (_B * sin_lat) ** 2)
    )


def rotation_rate(lat: float) -> np.ndarray:
    """
    Returns the rotation rate of earth at given latitude position.
    Calculates the rotation rate vector of the earth the given latitude.

    :param lat: The latitude position (in decimal degrees).
    :type lat: float

    :return: The rotation vector in degrees/second.
    :rtype: np.ndarray
    """
    lat_rad = math.radians(lat)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    return _OMEGA_E * np.array([cos_lat, 0, -sin_lat])


def rotation_rate_vec(lat: np.ndarray) -> np.ndarray:
    """
    Returns the rotation rate of earth at given latitude position.
    Calculates the rotation rate vector of the earth the given latitude.

    :param lat: The latitude position (in decimal degrees).
    :type lat: np.ndarray

    :return: The rotation vector in degrees/second.
    :rtype: np.ndarray
    """
    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    return _OMEGA_E * np.array([cos_lat, 0, -sin_lat])


def radius84(lat: float) -> float:
    """
    This function calculates the radius (in metres) from the centre of the
    Earth for a given latitude position, using WGS84 parameters.

    :param lat: The latitude of the points of interest.
    :type lat: float

    :return: The calculated radius in metres for the given position.
    :rtype: float
    """
    lat_rad = math.radians(lat)
    return (_A * _A) / math.sqrt((_A * math.cos(lat_rad)) ** 2 + (_B * math.sin(lat_rad)) ** 2)


def radius84_vec(lat: np.ndarray) -> np.ndarray:
    """
    The vectorised version of the
    :func:`~qnav.util.transformations.radius84` function. This
    calculates the radius (in metres) from the centre of the Earth for a large
    series of given latitude positions, using WGS84 parameters.

    :param lat: The latitude of the points of interest.
    :type lat: numpy.ndarray

    :return: The calculated radius in metres for the given position.
    :rtype: numpy.ndarray
    """

    lat_rad = np.radians(lat)
    return (_A * _A) / np.sqrt((_A * np.cos(lat_rad)) ** 2 + (_B * np.sin(lat_rad)) ** 2)
    #
    # lat_rad = np.radians(lat)
    # sin_lat = np.sin(lat_rad)
    # cos_lat = np.cos(lat_rad)
    #
    # return np.sqrt(
    #     (((_A ** 2) * cos_lat) ** 2 + (_B ** 2 * sin_lat) ** 2) /
    #     ((_A * cos_lat) ** 2 + (_B * sin_lat) ** 2)
    # )


def rotate_3d(psi: float, theta: float, phi: float) -> np.ndarray:
    """
    This function produces a three-dimensional rotation matrix corresponding
    to the Euler angles psi (Heading), theta (Pitch), phi (Roll/Bank).

    :param psi: The heading angle in radians.
    :type psi: float

    :param theta: The pitch angle in radians.
    :type theta: float

    :param phi: The roll/bank angle in radians.
    :type phi: float

    :return: A matrix corresponding to three rotations in the above order.
    :rtype: numpy.ndarray (3-by-3 elements)
    """

    dum1 = np.zeros((3, 3))
    dum2 = np.zeros((3, 3))
    dum3 = np.zeros((3, 3))

    dum1[0][0] = math.cos(psi)
    dum1[1][1] = math.cos(psi)
    dum1[2][2] = 1.0
    dum1[0][1] = math.sin(psi)
    dum1[1][0] = -math.sin(psi)

    dum2[0][0] = math.cos(theta)
    dum2[2][2] = math.cos(theta)
    dum2[1][1] = 1.0
    dum2[2][0] = math.sin(theta)
    dum2[0][2] = -math.sin(theta)

    dum3[1][1] = math.cos(phi)
    dum3[2][2] = math.cos(phi)
    dum3[0][0] = 1.0
    dum3[1][2] = math.sin(phi)
    dum3[2][1] = -math.sin(phi)

    return dum3.dot(dum2.dot(dum1))


def rotate_3d_vec(psi: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.rotate_3d` function. This
    produces multiple three-dimensional rotation matrices corresponding
    to the Euler angles psi (Heading), theta (Pitch), phi (Roll/Bank).

    :param psi: The heading angles in radians.
    :type psi: numpy.ndarray

    :param theta: The pitch angles in radians.
    :type theta: numpy.ndarray

    :param phi: The roll/bank angles in radians.
    :type phi: numpy.ndarray

    :return: The matrices corresponding to three rotations in the above order.
    :rtype: numpy.ndarray (n-3-by-3 elements)
    """

    assert psi.ndim == 1 and theta.ndim == 1 and phi.ndim == 1, \
        "Only 1-dimensional input is supported!"

    amount = psi.shape[0]

    dum1 = np.zeros([amount, 3, 3])
    dum2 = np.zeros([amount, 3, 3])
    dum3 = np.zeros([amount, 3, 3])

    dum1[:, 0, 0] = np.cos(psi)
    dum1[:, 1, 1] = np.cos(psi)
    dum1[:, 2, 2] = 1.0
    dum1[:, 0, 1] = np.sin(psi)
    dum1[:, 1, 0] = -np.sin(psi)

    dum2[:, 0, 0] = np.cos(theta)
    dum2[:, 2, 2] = np.cos(theta)
    dum2[:, 1, 1] = 1.0
    dum2[:, 2, 0] = np.sin(theta)
    dum2[:, 0, 2] = -np.sin(theta)

    dum3[:, 1, 1] = np.cos(phi)
    dum3[:, 2, 2] = np.cos(phi)
    dum3[:, 0, 0] = 1.0
    dum3[:, 1, 2] = np.sin(phi)
    dum3[:, 2, 1] = -np.sin(phi)

    temp_sum = np.einsum('nij, njk -> nik', dum2, dum1)
    to_return = np.einsum('nij, njk -> nik', dum3, temp_sum)

    return to_return


def lla2ned(lla: npt.ArrayLike, ref_lla: npt.ArrayLike) -> np.ndarray:
    """
    This function converts a reference Latitude-Longitude-Altitude (LLA)
    position from a given centre LLA point and translates its location to
    North-East-Down (NED) coordinates.

    :param lla: The centre location (array containing [lat, lon, alt]) with
        lat-lon given in degrees and alt given in metres.
    :type lla: numpy.ndarray (3-elements)

    :param ref_lla: The reference location (array containing [lat, lon, alt])
        with lat-lon given in degrees and alt given in metres.
    :type ref_lla: numpy.ndarray (3-elements)

    :return: An array containing the calculated North, East and Down
        coordinates from the centre position, were all values are given in
        metres.
    :rtype: numpy.ndarray (3-elements)
    """
    xyz = lla2ecef(lla)
    return ecef2ned(xyz, ref_lla)


def lla2ned_vec(lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.lla2ned` function. This
    allows means for converting reference Latitude-Longitude-Altitude (LLA)
    positions from given centre LLA points and translates their location to
    North-East-Down (NED) coordinates.

    :param lla: The centre locations (array rows containing [lat, lon, alt])
        with lat-lon given in degrees and alt given in metres.
    :type lla: numpy.ndarray (n-by-3 elements)

    :param ref_lla: The reference location (array rows containing
        [lat, lon, alt]) with lat-lon given in degrees and alt given in
        metres.
    :type ref_lla: numpy.ndarray (n-by-3 elements)

    :return: An array containing the calculated North, East and Down
        coordinates from the centre position, were all values are given in
        metres.
    :rtype: numpy.ndarray (n-by-3 elements)
    """

    xyz = lla2ecef_vec(lla)
    ref_xyz = lla2ecef_vec(ref_lla)
    t = ecef2ned_transform_vec(ref_lla)
    diff = xyz - ref_xyz

    # if t.ndim == 3:
    #     ned = np.einsum('nji, ni -> nj', t, diff)
    # else:
    #     ned = np.einsum('ji, ni -> nj', t, diff)

    if diff.ndim == 1:
        ned = np.tensordot(t, diff, axes=1)
    elif t.ndim == 2:
        ned = np.einsum('ji, ni -> nj', t, diff)
    else:
        ned = np.einsum('nji, ni -> nj', t, diff)

    ned[abs(ned) < _SMALL_VALUE] = 0.0
    return ned


def ned2lla(ned: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """
    This function converts a reference North-East-Down (NED) position from a
    given centre position and translates its location to Latitude-Longitude-
    Altitude (LLA) coordinates.

    :param ned: The centre location (array containing [north, east, down])
        with all values being given in metres.
    :type ned: numpy.ndarray (3-elements)

    :param ref_lla: The reference location (array containing [lat, lon, alt])
        with lat-lon given in degrees and alt given in metres.
    :type ref_lla: numpy.ndarray (3-elements)

    :return: An array containing the calculated Latitude, Longitude and
        Altitude coordinates from the centre position, with lat-lon is given
        in degrees and alt given in metres.
    :rtype: numpy.ndarray (3-elements)
    """

    # ref_xyz = lla2ecef(ref_lla)
    # t = ecef2ned_transform(ref_lla)
    #
    # xyz = np.linalg.inv(t).dot(ned) + ref_xyz
    # return ecef2lla(xyz)

    xyz = ned2ecef(ned, ref_lla)
    return ecef2lla(xyz)


def ned2lla_vec(ned: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.ned2lla` function. This
    converts reference North-East-Down (NED) positions from a given centre
    positions and translates their location to Latitude-Longitude-Altitude
    (LLA) coordinates.

    :param ned: The centre locations (array rows containing
        [north, east, down]) with all values being given in metres.
    :type ned: numpy.ndarray (n-by-3 elements)

    :param ref_lla: The reference location (array rows containing
        [lat, lon, alt]) with lat-lon given in degrees and alt given in
        metres.
    :type ref_lla: numpy.ndarray (n-by-3 elements)

    :return: Array rows containing the calculated Latitude, Longitude and
        Altitude coordinates from the centre positions, with lat-lon is given
        in degrees and alt given in metres.
    :rtype: numpy.ndarray (n-by-3 elements)
    """

    # ref_xyz = lla2ecef_vec(ref_lla)
    # t = ecef2ned_transform_vec(ref_lla)
    # inv_t = np.linalg.inv(t)
    #
    # # if ned.ndim == 1:
    # #     tmp = np.tensordot(inv_t, ned, axes=1)
    # # else:
    # #     tmp = np.einsum('nji, ni -> nj', inv_t, ned)
    #
    # if ned.ndim == 1:
    #     tmp = np.tensordot(inv_t, ned, axes=1)
    # elif inv_t.ndim == 2:
    #     tmp = np.einsum('ij, nj -> ni', inv_t, ned)
    # else:
    #     tmp = np.einsum('nji, ni -> nj', inv_t, ned)
    #
    # # np.einsum('ij, nj -> ni', inv_t, ned)
    #
    # xyz = tmp + ref_xyz

    xyz = ned2ecef_vec(ned, ref_lla)
    return ecef2lla_vec(xyz)


def lla2ecef(lla: np.ndarray) -> np.ndarray:
    """
    This function converts a reference Latitude-Longitude-Altitude (LLA)
    position to Earth-Centred Earth-Fixed coordinates.

    :param lla: The reference location (array containing [lat, lon, alt]) with
        lat-lon given in degrees and alt given in metres.
    :type lla: numpy.ndarray (3-elements)

    :return: An array containing the calculated X, Y and Z coordinates from
        the centre of the Earth, with all values being given in metres.
    :rtype: numpy.ndarray (3-elements)
    """

    lat = deg2rad(float(lla[0]))
    lon = deg2rad(float(lla[1]))
    alt = lla[2]

    n = _A / math.sqrt(1 - _ECC ** 2 * math.sin(lat) ** 2)

    x = (n + alt) * math.cos(lat) * math.cos(lon)
    y = (n + alt) * math.cos(lat) * math.sin(lon)
    z = (n * (_B ** 2 / _A ** 2) + alt) * math.sin(lat)

    xyz = np.array([x, y, z])
    xyz[abs(xyz) < _SMALL_VALUE] = 0.0

    return xyz


def lla2ecef_vec(lla: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.lla2ecef` function. This
    converts a series of reference Latitude-Longitude-Altitude (LLA) positions
    to Earth-Centred Earth-Fixed coordinates.

    :param lla: The reference locations (array rows containing
        [lat, lon, alt]) with lat-lon given in degrees and alt given in
        metres.
    :type lla: numpy.ndarray (n-by-3 elements)

    :return: An array containing the calculated X, Y and Z coordinates from
        the centre of the Earth, with all values being given in metres.
    :rtype: numpy.ndarray (n-by-3 elements)
    """

    # TODO: New
    if lla.ndim == 1:
        return lla2ecef(lla)

    lat = np.deg2rad(lla[:, 0])
    lon = np.deg2rad(lla[:, 1])
    alt = lla[:, 2]

    n = _A / np.sqrt(1 - _ECC ** 2 * np.sin(lat) ** 2)

    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (_B ** 2 / _A ** 2) + alt) * np.sin(lat)

    xyz = np.column_stack([x, y, z])
    xyz[abs(xyz) < _SMALL_VALUE] = 0.0

    return xyz


def euler_to_quaternion(angles: np.ndarray) -> np.ndarray:
    """
    Converts given attitude angles to quaternion format.
    Given an array representing euler angles, this function converts them to
    quaternion format. On succession, an array containing the a, b, c and d
    quaternion coefficients (i.e. where a + bi + cj + dk) will be returned.

    :param angles: Euler angles given in degrees
    :type angles: numpy.ndarray (3-elements)

    :return: The a, b, c, and d quaternion coefficients in columns.
    :rtype: numpy.ndarray (4-elements)
    """

    # Uses attitude conversion method uses maths described in Diebel, James.
    # "Representing attitude: Euler angles, unit quaternions, and rotation vectors."
    # Matrix 58, no. 15-16 (2006): 1-35.
    # Specifically using the details in Section 8.2 for rotation 1-2-3.

    # Validate the shape of the given input array:
    if angles.ndim != 1 or angles.shape[0] != 3:
        raise ValueError("Input must be array of 3 elements")

    # Convert angles to radians
    angles_rad = np.radians(angles)
    cos_d2 = np.cos(angles_rad / 2)
    sin_d2 = np.sin(angles_rad / 2)

    # Convert Euler angles (heading-pitch-roll) to quaternions (4D)
    a = (cos_d2[2] * cos_d2[1] * cos_d2[0]) + (sin_d2[2] * sin_d2[1] * sin_d2[0])
    b = (sin_d2[2] * cos_d2[1] * cos_d2[0]) - (cos_d2[2] * sin_d2[1] * sin_d2[0])
    c = (cos_d2[2] * sin_d2[1] * cos_d2[0]) + (sin_d2[2] * cos_d2[1] * sin_d2[0])
    d = (cos_d2[2] * cos_d2[1] * sin_d2[0]) - (sin_d2[2] * sin_d2[1] * cos_d2[0])

    # Stack the columns and return as single array
    return np.array((a, b, c, d))


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Converts given quaternions to Euler angle format.
    Given an array representing coefficients for quaternions (i.e.
    values a, b, c and d where a + bi + cj + dk), this function converts them
    to euler angle format. On succession, an array containing the Euler angles
    will be returned.

    :param quat: The a, b, c, and d quaternion coefficients in columns.
    :type quat: numpy.ndarray (4-elements)

    :return: The converted Euler angles given in degrees.
    :rtype: numpy.ndarray (3-elements)
    """

    # Validate the shape of the given input array:
    if quat.ndim != 1 or quat.shape[0] != 4:
        raise ValueError("Input must be array of 4 elements")

    w, x, y, z = quat

    sin_y_cos_p = 2 * (w * z + x * y)
    cos_y_cos_p = 1 - 2 * (y * y + z * z)
    heading = math.atan2(sin_y_cos_p, cos_y_cos_p)
    heading = math.degrees(heading)

    sin_p = np.clip(2 * (w * y - x * z), -1.0, 1.0)
    pitch = math.asin(sin_p)
    pitch = math.degrees(pitch)

    sin_r_cos_p = 2 * (w * x + y * z)
    cos_r_cos_p = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sin_r_cos_p, cos_r_cos_p)
    roll = math.degrees(roll)

    return np.array([heading, pitch, roll])


def euler_to_quaternion_vec(angles: np.ndarray) -> np.ndarray:
    """
    Converts given attitude angles to quaternion format.
    Given an array representing euler angles, this function converts them to
    quaternion format. On succession, an array containing the a, b, c and d
    quaternion coefficients (i.e. where a + bi + cj + dk) will be returned.

    :param angles: Euler angles given in degrees
    :type angles: numpy.ndarray (n-by-3 elements)

    :return: The a, b, c, and d quaternion coefficients in columns.
    :rtype: numpy.ndarray (n-by-4 elements)
    """

    # Uses attitude conversion method uses maths described in Diebel, James.
    # "Representing attitude: Euler angles, unit quaternions, and rotation vectors."
    # Matrix 58, no. 15-16 (2006): 1-35.
    # Specifically using the details in Section 8.2 for rotation 1-2-3.

    # Validate the shape of the given input array:
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError("Input must be of shape n-by-3")

    # Convert angles to radians
    angles_rad = np.radians(angles)
    cos_d2 = np.cos(angles_rad / 2)
    sin_d2 = np.sin(angles_rad / 2)

    # Convert Euler angles (heading-pitch-roll) to quaternions (4D)
    a = (cos_d2[:, 2] * cos_d2[:, 1] * cos_d2[:, 0]) + (sin_d2[:, 2] * sin_d2[:, 1] * sin_d2[:, 0])
    b = (sin_d2[:, 2] * cos_d2[:, 1] * cos_d2[:, 0]) - (cos_d2[:, 2] * sin_d2[:, 1] * sin_d2[:, 0])
    c = (cos_d2[:, 2] * sin_d2[:, 1] * cos_d2[:, 0]) + (sin_d2[:, 2] * cos_d2[:, 1] * sin_d2[:, 0])
    d = (cos_d2[:, 2] * cos_d2[:, 1] * sin_d2[:, 0]) - (sin_d2[:, 2] * sin_d2[:, 1] * cos_d2[:, 0])

    # Stack the columns and return as single array
    return np.column_stack((a, b, c, d))


def quaternion_to_euler_vec(quat: np.ndarray) -> np.ndarray:
    """
    Converts given quaternions to Euler angle format.
    Given an array representing coefficients for quaternions (i.e.
    values a, b, c and d where a + bi + cj + dk), this function converts them
    to euler angle format. On succession, an array containing the Euler angles
    will be returned.

    :param quat: The a, b, c, and d quaternion coefficients in columns.
    :type quat: numpy.ndarray (n-by-4 elements)

    :return: The converted Euler angles given in degrees.
    :rtype: numpy.ndarray (n-by-3 elements)
    """

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    sin_y_cos_p = 2 * (w * z + x * y)
    cos_y_cos_p = 1 - 2 * (y * y + z * z)
    heading = np.arctan2(sin_y_cos_p, cos_y_cos_p)

    sin_p = np.clip(2 * (w * y - x * z), -1.0, 1.0)
    pitch = np.arcsin(sin_p)

    sin_r_cos_p = 2 * (w * x + y * z)
    cos_r_cos_p = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sin_r_cos_p, cos_r_cos_p)

    return np.degrees(np.column_stack([heading, pitch, roll]))


# TODO: MOVE
def haversine(lat1: float, lon1: float, alt1: float,
              lat2: float, lon2: float, alt2: float) -> float:
    """
    Returns the estimated distance in metres between two sets of
    latitude-longitude-altitude positions. This function uses the Haversine
    formula to calculate the distance between latitude-longitude points, then
    employs pythagoras to estimate the actual distance by considering change
    in altitude between positions.

    :param lat1: The latitude position(s) belonging to the first set.
    :type lat1: numpy.ndarray (n-elements)

    :param lon1: The longitude position(s) belonging to the first set.
    :type lon1: numpy.ndarray (n-elements)

    :param alt1: The altitude position(s) belonging to the first set.
    :type alt1: numpy.ndarray (n-elements)

    :param lat2: The latitude position(s) belonging to the second set.
    :type lat2: numpy.ndarray (n-elements)

    :param lon2: The longitude position(s) belonging to the second set.
    :type lon2: numpy.ndarray (n-elements)

    :param alt2: The latitude position(s) belonging to the second set.
    :type alt2: numpy.ndarray (n-elements)

    :return: The distance between the given positions in metres.
    :rtype: numpy.array (n-elements)
    """

    # Convert the given coordinates into radians.
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Calculate the distance between the points using the Haversine formula.
    tmp1 = math.sin(abs(lat2_rad - lat1_rad) / 2)
    tmp2 = math.sin(abs(lon2_rad - lon1_rad) / 2)
    a = (tmp1 * tmp1) + math.cos(lat1_rad) * math.cos(lat2_rad) * (tmp2 * tmp2)
    dist = 2 * _R_E * math.sin(math.sqrt(a))

    # Obtain the difference in altitudes between positions.
    alt_diff = abs(alt1 - alt2)

    # Finally, use pythagoras theorem to estimate the distance between given
    # positions by including change in altitude.
    return math.sqrt((dist * dist) + (alt_diff * alt_diff))


# TODO: MOVE
def haversine_vec(lat1: npt.ArrayLike, lon1: npt.ArrayLike, alt1: npt.ArrayLike,
                  lat2: npt.ArrayLike, lon2: npt.ArrayLike, alt2: npt.ArrayLike) -> np.ndarray:
    """
    Returns the estimated distance in metres between two sets of
    latitude-longitude-altitude positions. This function uses the Haversine
    formula to calculate the distance between latitude-longitude points, then
    employs pythagoras to estimate the actual distance by considering change
    in altitude between positions.

    :param lat1: The latitude position(s) belonging to the first set.
    :type lat1: numpy.ndarray (n-elements)

    :param lon1: The longitude position(s) belonging to the first set.
    :type lon1: numpy.ndarray (n-elements)

    :param alt1: The altitude position(s) belonging to the first set.
    :type alt1: numpy.ndarray (n-elements)

    :param lat2: The latitude position(s) belonging to the second set.
    :type lat2: numpy.ndarray (n-elements)

    :param lon2: The longitude position(s) belonging to the second set.
    :type lon2: numpy.ndarray (n-elements)

    :param alt2: The latitude position(s) belonging to the second set.
    :type alt2: numpy.ndarray (n-elements)

    :return: The distance between the given positions in metres.
    :rtype: numpy.array (n-elements)
    """

    # Convert the given coordinates into radians.
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    # Calculate the distance between the points using the Haversine formula.
    tmp1 = np.sin(np.abs(lat2_rad - lat1_rad) / 2)
    tmp2 = np.sin(np.abs(lon2_rad - lon1_rad) / 2)
    a = (tmp1 * tmp1) + np.cos(lat1_rad) * np.cos(lat2_rad) * (tmp2 * tmp2)
    dist = 2 * _R_E * np.sin(np.sqrt(a))

    # Obtain the difference in altitudes between positions.
    alt_diff = np.abs(alt1 - alt2)

    # Finally, use pythagoras theorem to estimate the distance between given
    # positions by including change in altitude.
    return np.sqrt((dist * dist) + (alt_diff * alt_diff))


def get_time_pos_vel(time: dt.datetime) -> tuple[np.ndarray, np.ndarray]:
    """
    This function calculates the position and velocity matrices for a given
    timestamp.

    :param time: A datetime for the requested time.
    :type time: Python datetime instance

    :return: The position and velocity matrices.
    :rtype: tuple (mat_pos, mat_vel)
    """

    # Earth's rotation rate (rad/sec):
    # omega_rad_sec = 7.2921150e-5

    # Get total days since Jan 1st 2000, 12:00:00
    time0 = dt.datetime(2000, 1, 1, 12)
    d = time - time0
    d = d.total_seconds() / (24 * 60 * 60)

    # Calculate Greenwich Mean Sidereal Time (hours)
    gmst = (6.697374558 + 24.06570982441908 * d - 12) % 24

    # Convert GMST into radians
    gmst_rad = gmst * 2 * math.pi / 24

    # Calculate the longitude of ascending node of the Moon (radians)
    omega_rad = deg2rad(125.04 - 0.052954 * d)

    # Calculate the Mean Longitude of the Sun (rad)
    l_rad = deg2rad(280.47 + 0.98565 * d)

    # Calculate the obliquity
    eta_rad = deg2rad(23.4393 - 0.0000004 * d)

    # Calculate the Equinox
    d_phi_rad = -deg2rad(0.000319 * math.sin(omega_rad)
                         - 0.000024 * math.sin(2 * l_rad))

    # Calculate the apparent sidereal time (rad)
    gast_rad = gmst_rad + d_phi_rad * math.cos(eta_rad)

    # Calculate the angular difference between the ECI and ECEF frames (rad)
    theta_rad = gast_rad % (2 * math.pi)

    # Convert between the rotating reference frames
    mat_pos = np.array([[math.cos(theta_rad), -math.sin(theta_rad), 0],
                        [math.sin(theta_rad), math.cos(theta_rad), 0],
                        [0, 0, 1]])

    mat_vel = np.array([[-math.sin(theta_rad), -math.cos(theta_rad), 0],
                        [math.cos(theta_rad), -math.sin(theta_rad), 0],
                        [0, 0, 0]]) * _OMEGA_E

    return mat_pos, mat_vel


def ecef2eci(time: dt.datetime, r_ecef: np.ndarray, v_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function converts a position and velocity from the ECI frame to the
    ECEF frame.

    :param time: A datetime for the requested time.
    :type time: Python datetime instance

    :param r_ecef: Position in ECI (given in metres).
    :type r_ecef: numpy.ndarray (3-elements)

    :param v_ecef: Velocity in ECI (given in metres/second).
    :type r_ecef: numpy.ndarray (3-elements)

    :return: The position and velocity vectors in ECEF. Position is given in
        metres and velocity is given in metres/second.
    :rtype: tuple (r_eci, v_eci)
    """

    mat_pos, mat_vel = get_time_pos_vel(time)
    r_eci = mat_pos.dot(r_ecef)
    v_eci = mat_vel.dot(r_ecef) + mat_pos.dot(v_ecef)
    return r_eci, v_eci


def eci2ecef(time: dt.datetime, r_eci: np.ndarray, v_eci: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function converts a position and velocity from the ECEF frame to the
    ECI frame.

    :param time: A datetime for the requested time.
    :type time: Python datetime instance

    :param r_eci: Position in ECEF (given in metres).
    :type r_eci: numpy.ndarray (3-elements)

    :param v_eci: Velocity in ECEF (given in metres/second).
    :type v_eci: numpy.ndarray (3-elements)

    :return: The position and velocity vectors in ECI. Position is given in
        metres and velocity is given in metres/second.
    :rtype: tuple (r_ecef, v_ecef)
    """

    mat_pos, mat_vel = get_time_pos_vel(time)
    r_ecef = np.linalg.inv(mat_pos).dot(r_eci)
    v_ecef = np.linalg.inv(mat_pos).dot(v_eci - mat_vel.dot(r_ecef))
    return r_ecef, v_ecef


def ecef2lla(xyz: np.ndarray) -> np.ndarray:
    """
    Converts the given XYZ Earth Centred Earth Fixed coordinates to
    longitude latitude altitude coordinates.

    :param xyz: A position in ECEF (m-m-m)
    :type xyz: numpy.ndarray (3-elements)

    :return: The given position converted to LLA (deg-deg-m).
    :rtype: numpy.ndarray (3-elements)
    """

    x, y, z = xyz
    p = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan((z * _A) / (p * _B))

    lat = math.atan((z + _ECC_PRIME ** 2 * _B * math.sin(theta) ** 3)
                    / (p - _ECC ** 2 * _A * math.cos(theta) ** 3))

    lon = math.atan2(y, x)

    n = _A / math.sqrt(1 - _ECC ** 2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n

    return np.array([rad2deg(lat), rad2deg(lon), alt])


def ecef2lla_vec(xyz: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.ecef2lla` function.
    This converts the given XYZ Earth Centred Earth Fixed coordinates to
    longitude latitude altitude coordinates.

    :param xyz: A series of positions in ECEF (m-m-m)
    :type xyz: numpy.ndarray (n-by-3 elements)

    :return: The given positions converted to LLA (deg-deg-m).
    :rtype: numpy.ndarray (n-by-3 elements)
    """

    # TODO: NEW
    if xyz.ndim == 1:
        return ecef2lla(xyz)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan((z * _A) / (p * _B))

    lat = np.arctan((z + _ECC_PRIME ** 2 * _B * np.sin(theta) ** 3)
                    / (p - _ECC ** 2 * _A * np.cos(theta) ** 3))

    lon = np.arctan2(y, x)
    n = _A / np.sqrt(1 - _ECC ** 2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n

    return np.column_stack([np.rad2deg(lat), np.rad2deg(lon), alt])


def ecef2ned_transform(ref_lla: np.ndarray) -> np.ndarray:
    """
    This function generates a transformation matrix for ECEF to
    North-East-Down (NED) for an input Latitude-Longitude-Altitude (LLA)
    reference.

    :param ref_lla: The reference position in LLA, with lat-lon given in
        degrees and alt given in metres.
    :type ref_lla: numpy.ndarray (3-elements)

    :return: The calculated 3-by-3 transformation matrix.
    :rtype: NumPy Matrix (3-by-3 elements)
    """

    ref_lat = deg2rad(float(ref_lla[0]))
    ref_lon = deg2rad(float(ref_lla[1]))

    s_lat = math.sin(ref_lat)
    c_lat = math.cos(ref_lat)

    s_lon = math.sin(ref_lon)
    c_lon = math.cos(ref_lon)

    return np.array([
        [-s_lat * c_lon, -s_lat * s_lon, c_lat],
        [-s_lon, c_lon, 0.0],
        [-c_lat * c_lon, -c_lat * s_lon, -s_lat]
    ])


def ecef2ned_transform_vec(ref_lla: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.ecef2ned_transform`
    function. This generates transformation matrices for ECEF to
    North-East-Down (NED) for input Latitude-Longitude-Altitude (LLA)
    references.

    :param ref_lla: The reference position in LLAs, with lat-lon given in
        degrees and alt given in metres.
    :type ref_lla: numpy.ndarray (n-by-3 elements)

    :return: The calculated n-by-3-by-3 transformation matrix.
    :rtype: numpy.ndarray (n-3-3 elements)
    """

    # TODO: New
    if ref_lla.ndim == 1:
        return ecef2ned_transform(ref_lla)

    amount = ref_lla.shape[0]

    ref_lat = np.deg2rad(ref_lla[:, 0])
    ref_lon = np.deg2rad(ref_lla[:, 1])

    s_lat = np.sin(ref_lat)
    c_lat = np.cos(ref_lat)

    s_lon = np.sin(ref_lon)
    c_lon = np.cos(ref_lon)

    tmp1 = -s_lat * c_lon
    tmp2 = -s_lat * s_lon
    tmp3 = c_lat

    tmp4 = -s_lon
    tmp5 = c_lon
    tmp6 = np.zeros(c_lon.shape)

    tmp7 = -c_lat * c_lon
    tmp8 = -c_lat * s_lon
    tmp9 = -s_lat

    return np.column_stack([tmp1, tmp2, tmp3,
                            tmp4, tmp5, tmp6,
                            tmp7, tmp8, tmp9]).reshape(amount, 3, 3)


def ecef2ned(xyz: npt.ArrayLike, ref_lla: npt.ArrayLike) -> np.ndarray:
    """
    This function takes an ECEF position and a reference
    Latitude-Longitude-Altitude (LLA) position and returns the relative
    position of the former from the latter in North-East-Down (NED) reference
    frame.

    :param xyz: Position in ECEF, given in metres.
    :type xyz: numpy.ndarray (3-elements)

    :param ref_lla: The reference LLA position, with lat-lon given in degrees
        and alt given in metres.
    :type ref_lla: numpy.ndarray (3-elements)

    :return: The given position converted to NED, given in metres.
    :rtype: numpy.ndarray (3-elements)
    """

    ref_xyz = lla2ecef(ref_lla)
    t = ecef2ned_transform(ref_lla)

    ned = t.dot(xyz - ref_xyz)
    ned[abs(ned) < _SMALL_VALUE] = 0.0
    return ned

def ned2ecef(ned: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """
    This function takes an origin Latitude-Longitude-Altitude (LLA) position
    and a relative position in North-East-Down (NED) from this origin and
    returns the latter position in ECEF.

    :param ned: Position in NED, given in metres.
    :type ned: numpy.ndarray (3-elements)

    :param ref_lla: The reference LLA position, with lat-lon given in degrees
        and alt given in metres.
    :type ref_lla: numpy.ndarray (3-elements)

    :return: The given position converted to ECEF, given in metres.
    :rtype: numpy.ndarray (3-elements)
    """

    ref_xyz = lla2ecef(ref_lla)
    t = ecef2ned_transform(ref_lla)
    return np.linalg.inv(t).dot(ned) + ref_xyz

def ned2ecef_vec(ned: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    """
    This function takes a North-East-Down (NED) position along with its
    relative Latitude-Longitude-Altitude (LLA) reference position and
    returns the latter position in ECEF.

    :param ned: Positions in NED, given in metres.
    :type ned: numpy.ndarray  (n-by-3 elements)

    :param ref_lla: The reference LLA positions, with lat-lon given in
        degrees and alt given in metres.
    :type ref_lla: numpy.ndarray (n-by-3 elements)

    :return: The given position converted to ECEF, given in metres.
    :rtype: numpy.ndarray (n-by-3 elements)
    """

    ref_xyz = lla2ecef_vec(ref_lla)
    t = ecef2ned_transform_vec(ref_lla)
    inv_t = np.linalg.inv(t)

    if ned.ndim == 1:
        xyz = np.tensordot(inv_t, ned, axes=1)
    elif inv_t.ndim == 2:
        xyz = np.einsum('ij, nj -> ni', inv_t, ned)
    else:
        xyz = np.einsum('nji, ni -> nj', inv_t, ned)

    return xyz + ref_xyz


def vned2vecef(vel_ned: np.ndarray, pos_lla: np.ndarray) -> np.ndarray:
    """
    This function takes an origin Latitude-Longitude-Altitude (LLA) position
    and a relative velocity in North-East-Down (NED) components from this
    origin and returns the velocity in ECEF format.

    :param vel_ned: Velocity in NED, given in metres/second.
    :type vel_ned: numpy.ndarray (3-elements)

    :param pos_lla: The reference LLA position, with lat-lon given in degrees
        and alt given in metres.
    :type pos_lla: numpy.ndarray (3-elements)

    :return: The given velocity vector converted to ECEF, given in metres/sec.
    :rtype: NumPy Matrix (3-by-3 elements)
    """

    rne = ecef2ned_transform(pos_lla)
    return np.linalg.lstsq(rne, vel_ned, rcond=None)[0]
    # return np.linalg.solve(rne, vel_ned)  # TODO: Use alternative?


def utc2gps(time: dt.datetime) -> dict:
    """
    This function converts a provided time given in Coordinated Universal Time
    (UTC) to a Python dictionary containing the corresponding week number, day
    number and second.

    :param time: A requested time in Coordinated Universal Time (UTC) format.
    :type time: Python Datetime instance

    :return: The provided time as a Python dictionary that can be used for
        GPS functions.
    :rtype: dict
    """

    leap_sec = 19  # Leap seconds between TAI and GPS time
    leap_sec_atom = 37  # Leap seconds between UTC and TAI

    sec_in_min = 60
    sec_in_hour = 60 * sec_in_min
    sec_in_day = 24 * sec_in_hour
    sec_in_week = 7 * sec_in_day

    gps_time_sec = time.timestamp() - 315964782 + leap_sec - leap_sec_atom
    gps_week_num = math.floor(gps_time_sec / sec_in_week)

    gps_time = {
        'week': gps_week_num,
        'day': gps_time_sec / sec_in_day,
        'seconds': gps_time_sec - gps_week_num * sec_in_week
    }

    return gps_time


def ned2azel(ned: np.ndarray) -> tuple[float, float]:
    """
    This function takes an origin North-East-Down (NED) position and converts
    it into an azimuth and elevation value.

    :param ned: The centre location (array containing [north, east, down])
        with all values being given in metres.
    :type ned: numpy.ndarray (3-elements)

    :return: A tuple containing two elements, the corresponding azimuth and
        elevation in radians.
    :rtype: tuple[float, float]
    """

    az_rad = rad2deg(math.atan2(ned[1], ned[0]))
    dr = math.sqrt(ned[0] ** 2 + ned[1] ** 2)

    if dr > 0:
        el_rad = rad2deg(math.atan(-ned[2] / dr))

    elif ned[2] > 0:
        el_rad = -90.0

    elif ned[2] < 0:
        el_rad = 90.0

    else:
        el_rad = 0.0

    return az_rad, el_rad


def get_transport_rate(position_lla: np.ndarray, velocity_ned: np.ndarray) -> np.ndarray:
    """
    This function computes the transport rate (in Local NED axes) for given
    position and velocity vectors.

    :param position_lla: The current position vector stating the latitude,
        longitude and altitude (degrees-degrees-metres).
    :type position_lla: numpy.ndarray (3-elements)

    :param velocity_ned: The current velocity vector stating the velocity
        North, East and Down (given in metres/second).
    :type velocity_ned: numpy.ndarray (3-elements)

    :return: The transport rate for the given position and velocity.
    :rtype: NumPy Matrix (3-by-3 elements)
    """

    # Obtain the current latitude and altitude.
    lat = deg2rad(float(position_lla[0]))
    h = position_lla[2]

    # Obtain A and B Earth radii.
    a = constants.POLAR_AXIS_A
    b = constants.POLAR_AXIS_B

    # Calculate eccentricity and sin(latitude) squared.
    ecc_sq = (a ** 2 - b ** 2) / a ** 2
    sin_lat_sq = math.sin(lat) * math.sin(lat)

    r_n = a * (1 - ecc_sq) / ((1 - ecc_sq * sin_lat_sq) ** (3 / 2))
    r_e_1 = a / math.sqrt(1 - ecc_sq * sin_lat_sq)

    # Finally, return the transport rate matrix
    # for the given position and velocity.
    return np.array([
        velocity_ned[1] / (r_e_1 + h),
        -velocity_ned[0] / (r_n + h),
        -velocity_ned[1] * math.tan(lat) / (r_e_1 + h)
    ])


def get_transport_rate_vec(position_lla: np.ndarray, velocity_ned: np.ndarray) -> np.ndarray:
    """
    This function is the vectorised version of the
    :func:`~qnav.util.transformations.get_transport_rate`
    function. This computes the transport rate (in Local NED axes)
    for a series of given positions and velocity vectors.

    :param position_lla: The current position vectors stating the latitude,
        longitude and altitude (degrees-degrees-metres).
    :type position_lla: numpy.ndarray (n-by-3 elements)

    :param velocity_ned: The current velocity vectors stating the velocity
        North, East and Down (given in metres/second).
    :type velocity_ned: numpy.ndarray (n-by-3 elements)

    :return: The transport rate for the given positions and velocities.
    :rtype: NumPy Matrix (n-3-3 elements)
    """

    # Obtain the current latitude and altitude.
    lat = np.deg2rad(position_lla[:, 0])
    h = position_lla[:, 2]

    # Obtain A and B Earth radii.
    a = constants.POLAR_AXIS_A
    b = constants.POLAR_AXIS_B

    ecc_sq = (a ** 2 - b ** 2) / a ** 2
    sin_lat_sq = np.sin(lat) * np.sin(lat)

    r_n = a * (1 - ecc_sq) / ((1 - ecc_sq * sin_lat_sq) ** (3 / 2))
    r_e_1 = a / np.sqrt(1 - ecc_sq * sin_lat_sq)

    x = velocity_ned[:, 1] / (r_e_1 + h)
    y = -velocity_ned[:, 0] / (r_n + h)
    z = -velocity_ned[:, 1] * np.tan(lat) / (r_e_1 + h)

    return np.column_stack([x, y, z])


def vel_ecef2vel_ned(vel_ecef: np.ndarray, lla: np.ndarray) -> np.ndarray:
    """
    A function to take ECEF velocity and an LLA location and convert it into
    the NED velocity.

    :param vel_ecef: The velocity given in Earth-Centred Earth-Fixed format.
    :type vel_ecef: numpy.ndarray (3-elements)

    :param lla: The current latitude, longitude and altitude (in deg-deg-m).
    :type vel_ecef: numpy.ndarray (3-elements)

    :return: The given Earth-Centred Earth-Fixed (ECEF) velocity converted
        into local North-East-Down (NED), subject to the given reference
        position.
    :rtype: numpy.ndarray (3-elements)
    """

    lat = deg2rad(float(lla[0]))
    lon = deg2rad(float(lla[1]))

    c_lat = math.cos(lat)
    slat = math.sin(lat)
    c_long = math.cos(lon)
    s_long = math.sin(lon)

    rne = np.array([
        [-slat * c_long, -slat * s_long, c_lat],
        [-s_long, c_long, 0],
        [-c_lat * c_long, -c_lat * s_long, -slat]])

    vel_ned = rne @ vel_ecef.T
    return vel_ned


def __utm_coefficient(e: float, m: float) -> np.ndarray:
    """
    This function calculate the projection coefficients for UTM to
    latitude-longitude conversion calculation.

    :param e: The first ellipsoid eccentricity
    :type e: float

    :param m: The mercator (0 for transverse mercator, 1 for transverse
        mercator reverse coefficients and 2 for meridian arc)
    :type m: float

    :return: 5 coefficients.
    :rtype: NumPy array
    """

    if m == 0:
        c0 = np.array([
            [-175 / 16384, 0, -5 / 256, 0, -3 / 64, 0, -1 / 4, 0, 1],
            [-105 / 4096, 0, -45 / 1024, 0, -3 / 32, 0, -3 / 8, 0, 0],
            [525 / 16384, 0, 45 / 1024, 0, 15 / 256, 0, 0, 0, 0],
            [-175 / 12288, 0, -35 / 3072, 0, 0, 0, 0, 0, 0],
            [315 / 131072, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    elif m == 1:
        c0 = np.array([
            [-175 / 16384, 0, -5 / 256, 0, -3 / 64, 0, -1 / 4, 0, 1],
            [1 / 61440, 0, 7 / 2048, 0, 1 / 48, 0, 1 / 8, 0, 0],
            [559 / 368640, 0, 3 / 1280, 0, 1 / 768, 0, 0, 0, 0],
            [283 / 430080, 0, 17 / 30720, 0, 0, 0, 0, 0, 0],
            [4397 / 41287680, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    else:
        c0 = np.array([
            [-175 / 16384, 0, -5 / 256, 0, -3 / 64, 0, -1 / 4, 0, 1],
            [-901 / 184320, 0, -9 / 1024, 0, -1 / 96, 0, 1 / 8, 0, 0],
            [-311 / 737280, 0, 17 / 5120, 0, 13 / 768, 0, 0, 0, 0],
            [899 / 430080, 0, 61 / 15360, 0, 0, 0, 0, 0, 0],
            [49561 / 41287680, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    c = np.zeros(c0.shape[0])

    for i in range(c0.shape[0]):
        c[i] = np.polyval(c0[i, :], e)

    return c


# TODO: MOVE
def cross_prod_xy(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    An optimised method for computing the cross-product of two 1-by-3 arrays.
    In most cases this method is far quicker that the Numpy cross function.

    :param vec1: The first 1-by-3 tuple/array.
    :type vec1: numpy.ndarray (3 elements)

    :param vec2: The first 1-by-3 tuple/array.
    :type vec2: numpy.ndarray (3 elements)

    :return: The cross product of the two tuples.
    :rtype: numpy.ndarray (3 elements)
    """

    return np.array((
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0]
    ))



# if __name__ == '__main__':
#
#
#     min_angle: float = -np.pi
#     max_angle: float = np.pi
#
#     psi_t = np.random.uniform(min_angle, max_angle, (2, 2))
#     theta_t = np.random.uniform(min_angle, max_angle, (2, 2))
#     phi_t = np.random.uniform(min_angle, max_angle, (2, 2))
#
#     rotate_3d_vec(psi_t, theta_t, phi_t)

    # test_lla_points = np.array([
    #     [52.0, -3.0, 1000],
    #     [52.1, -3.1, 1000],
    #     [52.2, -3.2, 1000],
    # ])
    #
    # test_ref_point = np.array([
    #     [52.1, -3.1, 1000],
    #     [52.1, -3.1, 1000],
    #     [52.1, -3.1, 1000],
    # ])
    #
    # print(lla2ned_vec(test_lla_points, test_ref_point))
    # print(lla2ned_vec(test_lla_points, test_ref_point[0, :]))

# if __name__ == '__main__':
#
#     test = np.array([10, 20, 30])
#     quat = euler_to_quaternion(test)
#     euler = quaternion_to_euler(quat)
#
#     print(f"Input: \t{test}")
#     print(f"Quaternion: \t{quat}")
#     print(f"Output: \t{euler}")
