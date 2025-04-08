
import qnav.util.transformations as trans
import numpy as np
import pytest


def get_random_lla(num_points: int = 1) -> np.ndarray:
    """
    Returns random positions as latitude-longitude-altitude coordinates.
    Generates and returns a requested number of random positions with
    altitudes between 0 and 10,000 metres.

    :param num_points: The number of positions to generate and return.
    :type num_points: int

    :return: LLA positions defined as a n-by-3 numpy array.
    :rtype: np.ndarray
    """
    lat_points = np.random.uniform(-90, 90, num_points)
    lon_points = np.random.uniform(-180, 180, num_points)
    alt_points = np.random.uniform(0, 10000, num_points)
    lla_points = np.column_stack((lat_points, lon_points, alt_points))
    return np.squeeze(lla_points)


def rep_arrays(points_a: np.ndarray, points_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Repeats arrays so that they are the same length.
    Given two arrays, this function will repeat the one with single row so
    that they have the same number of columns. Used for verifying array
    broadcasting is implemented correctly.

    :param points_a: The first array.
    :type points_a: np.ndarray

    :param points_b: The second array.
    :type points_b: np.ndarray

    :return: A tuple containing the given arrays of equal size.
    :rtype: tuple[np.ndarray, np.ndarray]
    """

    if points_a.shape == points_b.shape:
        return points_a, points_b

    if points_a.ndim == 1:
        new_size = (points_b.shape[0], 1)
        return np.tile(points_a, new_size), points_b

    if points_b.ndim == 1:
        new_size = (points_a.shape[0], 1)
        return points_a, np.tile(points_b, new_size)

    raise ValueError("Unsupported size combinations!")


@pytest.fixture()
def position() -> np.ndarray:
    return get_random_lla()


class TestTransforms:

    def test_lla_ecef(self, position):
        ecef = trans.lla2ecef(position)
        lla = trans.ecef2lla(ecef)
        np.testing.assert_allclose(lla, position)

    def test_lla_ned(self, position):

        scale = np.array([10, 10, 1000])
        rand = np.random.randn(3) * scale
        ref_lla = position + rand

        ned = trans.lla2ned(position, ref_lla)
        lla = trans.ned2lla(ned, ref_lla)
        np.testing.assert_allclose(lla, position)

    def test_ned_ecef(self, position):

        scale = np.array([10, 10, 1000])
        rand = np.random.randn(3) * scale
        ref_lla = position + rand

        ned1 = trans.lla2ned(position, ref_lla)
        ecef = trans.ned2ecef(ned1, ref_lla)
        ned2 = trans.ecef2ned(ecef, ref_lla)

        np.testing.assert_allclose(ned1, ned2)


@pytest.mark.parametrize("num_a_points", (1, 1000))
@pytest.mark.parametrize("num_b_points", (1, 1000))
class TestVectorised:

    def test_radius_vec(self, num_a_points: int, num_b_points: int):

        lat_points = np.random.uniform(-90, 90, num_a_points)
        vec_output = trans.radius84_vec(lat_points)

        flat_points = lat_points.flatten()
        flat_output = vec_output.flatten()

        for i in range(len(flat_points)):
            expected = trans.radius84(float(flat_points[i]))
            np.testing.assert_allclose(flat_output[i], expected)

    def test_rotate_3d_vec(self, num_a_points: int, num_b_points: int):

        min_angle: float = -np.pi
        max_angle: float = np.pi

        psi = np.random.uniform(min_angle, max_angle, num_a_points)
        theta = np.random.uniform(min_angle, max_angle, num_a_points)
        phi = np.random.uniform(min_angle, max_angle, num_a_points)
        vec_output = trans.rotate_3d_vec(psi, theta, phi)

        for i in range(num_a_points):
            expected = trans.rotate_3d(
                float(psi[i]), float(theta[i]), float(phi[i]))
            np.testing.assert_allclose(vec_output[i, :], expected)

    def test_lla2ned_vec(self, num_a_points: int, num_b_points: int):

        lla_1 = get_random_lla(num_a_points)
        lla_2 = get_random_lla(num_b_points)
        vec_output = trans.lla2ned_vec(lla_1, lla_2)

        lla_1, lla_2 = rep_arrays(lla_1, lla_2)
        lla_1 = np.atleast_2d(lla_1)
        lla_2 = np.atleast_2d(lla_2)
        vec_output = np.atleast_2d(vec_output)

        for i in range(len(lla_1)):
            expected = trans.lla2ned(lla_1[i], lla_2[i])
            np.testing.assert_allclose(vec_output[i, :], expected)

    def test_lla2ecef_vec(self, num_a_points: int, num_b_points: int):

        lla = get_random_lla(num_a_points)
        vec_output = trans.lla2ecef_vec(lla)

        lla = np.atleast_2d(lla)
        vec_output = np.atleast_2d(vec_output)

        for i in range(num_a_points):
            expected = trans.lla2ecef(lla[i])
            np.testing.assert_allclose(vec_output[i], expected)

    def test_ned2lla_vec(self, num_a_points: int, num_b_points: int):

        lla_1 = get_random_lla(num_a_points)
        lla_2 = get_random_lla(num_b_points)
        vec_output = trans.ned2lla_vec(lla_1, lla_2)

        lla_1, lla_2 = rep_arrays(lla_1, lla_2)
        lla_1 = np.atleast_2d(lla_1)
        lla_2 = np.atleast_2d(lla_2)
        vec_output = np.atleast_2d(vec_output)

        for i in range(len(lla_1)):
            expected = trans.ned2lla(lla_1[i], lla_2[i])
            np.testing.assert_allclose(vec_output[i, :], expected)

    def test_ned2ecef_vec(self, num_a_points: int, num_b_points: int):

        lla_1 = get_random_lla(num_a_points)
        lla_2 = get_random_lla(num_b_points)
        vec_output = trans.ned2ecef_vec(lla_1, lla_2)

        lla_1, lla_2 = rep_arrays(lla_1, lla_2)
        lla_1 = np.atleast_2d(lla_1)
        lla_2 = np.atleast_2d(lla_2)
        vec_output = np.atleast_2d(vec_output)

        for i in range(len(lla_1)):
            expected = trans.ned2ecef(lla_1[i], lla_2[i])
            np.testing.assert_allclose(vec_output[i, :], expected)

    def test_ecef2lla_vec(self, num_a_points: int, num_b_points: int):

        min_val = 6371000
        max_vax = min_val + 10000
        ecef_points = np.random.uniform(
            min_val, max_vax, (num_a_points, 3))

        vec_output = trans.ecef2lla_vec(ecef_points)

        ecef_points = np.atleast_2d(ecef_points)
        vec_output = np.atleast_2d(vec_output)

        for i in range(num_a_points):
            expected = trans.ecef2lla(ecef_points[i])
            np.testing.assert_allclose(vec_output[i], expected)

    def test_ecef2ned_transform_vec(self, num_a_points: int, num_b_points: int):

        lla_points = get_random_lla(num_a_points)
        vec_output = trans.ecef2ned_transform_vec(lla_points)

        lla_points = np.atleast_2d(lla_points)
        vec_output = vec_output.reshape((-1, 3, 3))

        for i in range(len(lla_points)):
            expected = trans.ecef2ned_transform(lla_points[i])
            np.testing.assert_allclose(vec_output[i, :, :], expected)
