

import numpy as np
import pytest


from qnav.gps.zones import FlaggedPolygon


@pytest.fixture(params=[1,])
def axis_range(request) -> float:
    return request.param

@pytest.fixture()
def simple_square(axis_range):

    # Generate equal square
    is_active = True
    scale = 0.5 * axis_range
    lat_points = np.array([-1, 1, 1, -1]) * scale
    lon_points = np.array([1, 1, -1, -1]) * scale

    # Apply random transition offers
    lat_centre = (np.random.randn() - 0.5) * scale
    lon_centre = (np.random.randn() - 0.5) * scale
    lat_points += lat_centre
    lon_points += lon_centre

    return FlaggedPolygon(lat_points, lon_points, is_active), lat_centre, lon_centre

@pytest.fixture()
def simple_circle(axis_range):

    is_active = True
    scale = 0.5 * axis_range

    angle_intervals = np.linspace(0, 2 * np.pi, 361)
    lat_points = np.sin(angle_intervals)
    lon_points = np.cos(angle_intervals)

    # Apply random transition offers
    lat_centre = (np.random.randn() - 0.5) * scale
    lon_centre = (np.random.randn() - 0.5) * scale
    lat_points += lat_centre
    lon_points += lon_centre

    return FlaggedPolygon(lat_points, lon_points, is_active), lat_centre, lon_centre

@pytest.fixture()
def semi_circle(axis_range):

    is_active = True
    percent = np.random.rand()
    offset = np.random.rand() * 2 * np.pi
    scale = 0.5 * axis_range

    angle_intervals = np.linspace(0, 2 * np.pi * percent, 1000)
    angle_intervals += offset

    lat_points = np.sin(angle_intervals)
    lon_points = np.cos(angle_intervals)

    lat_points = np.concatenate(([0], lat_points), axis=0)
    lon_points = np.concatenate(([0], lon_points), axis=0)

    # Apply random transition offers
    lat_centre = (np.random.randn() - 0.5) * scale
    lon_centre = (np.random.randn() - 0.5) * scale
    lat_points += lat_centre
    lon_points += lon_centre

    return FlaggedPolygon(lat_points, lon_points, is_active), lat_centre, lon_centre


@pytest.fixture()
def sample_points(axis_range, request) -> tuple[np.ndarray, np.ndarray]:
    num_points = request.param
    lat_points = np.random.rand(num_points) * 2 - 1
    lon_points = np.random.rand(num_points) * 2 - 1
    lat_points *= axis_range
    lon_points *= axis_range
    return lat_points, lon_points


@pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
def test_simple_square(simple_square, sample_points):

    # Obtain the response
    polygon, _, _ = simple_square
    test_lat, test_lon = sample_points
    is_inside = polygon.is_points_inside(test_lat, test_lon)

    # Obtain the polygon points
    poly_lat = polygon.lat_points
    poly_lon = polygon.lon_points

    # Find the assumed response for points inside
    check_inside = ((test_lat < max(poly_lat))
                    & (test_lat > min(poly_lat))
                    & (test_lon < max(poly_lon))
                    & (test_lon > min(poly_lon)))

    # Check that the response is the same as the assumed values
    np.testing.assert_array_equal(is_inside, check_inside)


@pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
def test_simple_circle(simple_circle, sample_points):

    # Obtain the response
    test_lat, test_lon = sample_points
    polygon, centre_lat, centre_lon = simple_circle
    is_inside = polygon.is_points_inside(test_lat, test_lon)

    # Obtain the polygon points
    poly_lat = polygon.lat_points
    poly_lon = polygon.lon_points

    # poly_centre_lat = (min(poly_lat) + max(poly_lat)) / 2
    # poly_centre_lon = (min(poly_lon) + max(poly_lon)) / 2

    radius = np.sqrt(
        (poly_lat[0] - centre_lat) ** 2
        + (poly_lon[0] - centre_lon) ** 2)

    dists = np.sqrt(
        (test_lat - centre_lat) ** 2
        + (test_lon - centre_lon) ** 2)

    # Check that the response is the same as the assumed values
    check_inside = dists < radius
    np.testing.assert_array_equal(is_inside, check_inside)


def get_angles(y,x):
    to_return = (np.atan2(y, x) + np.pi) % (2 * np.pi)
    return np.degrees(to_return)


@pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
def test_semi_circle(semi_circle, sample_points):

    # Obtain the response
    test_lat, test_lon = sample_points
    polygon, centre_lat, centre_lon = semi_circle
    is_inside = polygon.is_points_inside(test_lat, test_lon)

    # Obtain the polygon points
    poly_lat = polygon.lat_points
    poly_lon = polygon.lon_points

    # Obtain the positions of the first and last angles
    start_lat, start_lon = poly_lat[1], poly_lon[1]
    end_lat, end_lon = poly_lat[-2], poly_lon[-2]

    radius = np.sqrt((start_lat - centre_lat) ** 2 + (start_lon - centre_lon) ** 2)
    start_theta = get_angles(start_lat - centre_lat, start_lon - centre_lon)
    end_theta = get_angles(end_lat - centre_lat, end_lon - centre_lon)
    sample_theta = get_angles(test_lat - centre_lat, test_lon - centre_lon)
    dists = np.sqrt((test_lat - centre_lat) ** 2 + (test_lon - centre_lon) ** 2)

    check_inside = dists < radius

    if start_theta > end_theta:
        tmp = (sample_theta > start_theta) | (sample_theta < end_theta)
        check_inside &= tmp
    else:
        check_inside &= sample_theta > start_theta
        check_inside &= sample_theta < end_theta

    np.testing.assert_array_equal(is_inside, check_inside)

def assert_vectorised_match(polygon, sample_points):

    test_lat, test_lon = sample_points
    vec_output = polygon.is_points_inside(test_lat, test_lon)

    scalar_output = np.zeros_like(vec_output, dtype=bool)
    for i, (lat, lon) in enumerate(zip(test_lat, test_lon)):
        scalar_output[i] = polygon.is_point_inside(lat, lon)

    np.testing.assert_array_equal(vec_output, scalar_output)

@pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
def test_simple_square_vec(simple_square, sample_points):
    polygon, _, _ = simple_square
    assert_vectorised_match(polygon, sample_points)

@pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
def test_simple_circle_vec(simple_circle, sample_points):
    polygon, _, _ = simple_circle
    assert_vectorised_match(polygon, sample_points)

@pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
def test_semi_circle_vec(semi_circle, sample_points):
    polygon, _, _ = semi_circle
    assert_vectorised_match(polygon, sample_points)


@pytest.mark.parametrize('give_closed', [True, False])
@pytest.mark.parametrize('num_points', [10,])
def test_auto_close(give_closed: bool, num_points: int):

    rand_lat = np.random.rand(num_points)
    rand_lon = np.random.rand(num_points)

    if give_closed:
        rand_lat[-1] = rand_lat[0]
        rand_lon[-1] = rand_lon[0]

    else:
        if rand_lat[-1] == rand_lat[0]:
            rand_lon[-1] += 0.1

        if rand_lon[-1] == rand_lon[0]:
            rand_lon[-1] -= 0.1

    polygon = FlaggedPolygon(rand_lat, rand_lon, True)
    lat_points = polygon.lat_points
    lon_points = polygon.lon_points

    assert lat_points[0] == lat_points[-1], 'Failed to auto-close latitudes'
    assert lon_points[0] == lon_points[-1], 'Failed to auto-close longitudes'

    if give_closed:
        np.testing.assert_array_equal(lat_points, rand_lat)
        np.testing.assert_array_equal(lon_points, rand_lon)

    else:
        np.testing.assert_array_equal(lat_points[:-1], rand_lat)
        np.testing.assert_array_equal(lon_points[:-1], rand_lon)


@pytest.mark.parametrize('priority', [-50, -1, 0, 10, 100])
def test_priority_field(priority):
    rand_lat = np.random.rand(5)
    rand_lon = np.random.rand(5)
    polygon = FlaggedPolygon(rand_lat, rand_lon, True, priority)
    assert polygon.priority == priority

@pytest.mark.parametrize('num_points', [10, 1000])
def test_lat_field(num_points):
    rand_lat = np.random.rand(num_points)
    rand_lon = np.random.rand(num_points)
    polygon = FlaggedPolygon(rand_lat, rand_lon, True)
    np.testing.assert_array_equal(polygon.lat_points[:-1], rand_lat)
    assert rand_lat is not polygon.lat_points, \
        'Failed to return copy of lat points'

@pytest.mark.parametrize('num_points', [10, 1000])
def test_lon_field(num_points):
    rand_lat = np.random.rand(num_points)
    rand_lon = np.random.rand(num_points)
    polygon = FlaggedPolygon(rand_lat, rand_lon, True)
    np.testing.assert_array_equal(polygon.lon_points[:-1], rand_lon)
    assert rand_lon is not polygon.lon_points, \
        'Failed to return copy of lon points'


# @pytest.mark.parametrize('num_polygons', [1, 5, 50])
# def test_group_num(num_polygons):
#
#     rand_lat = np.random.rand(10)
#     rand_lon = np.random.rand(10)
#     is_active = True
#     priority = 0
#
#     polygons = []
#     for i in range(num_polygons):
#         polygons.append(FlaggedPolygon(rand_lat, rand_lon, is_active, priority))
#
#     group = FlaggedPolygonGroup(polygons)
#     assert group.num_polygons == num_polygons, \
#         'Returned number of polygons is not what was expected'
#
#
# @pytest.mark.parametrize('num_polygons', [1, 5, 50])
# def test_group_polygons(num_polygons):
#
#     rand_lat = np.random.rand(10)
#     rand_lon = np.random.rand(10)
#     is_active = True
#
#     polygons = []
#     for i in range(num_polygons):
#         priority = i
#         polygons.append(FlaggedPolygon(rand_lat, rand_lon, is_active, priority))
#
#     group = FlaggedPolygonGroup(polygons)
#     assert group.polygons == polygons, 'Returned polygons is not what was expected'
#
#
# @pytest.mark.parametrize('num_polygons', [1, 5, 50])
# def test_group_order(num_polygons):
#
#     rand_lat = np.random.rand(10)
#     rand_lon = np.random.rand(10)
#     is_active = True
#
#     priorities = np.arange(num_polygons)
#     np.random.shuffle(priorities)
#
#     polygons = []
#     for i in range(num_polygons):
#         priority = int(priorities[i])
#         polygons.append(FlaggedPolygon(rand_lat, rand_lon, is_active, priority))
#
#     group = FlaggedPolygonGroup(polygons)
#     group_polygons = group.polygons
#
#     order = np.sort(priorities)
#     polygon_order = [p.priority for p in group_polygons]
#     np.testing.assert_array_equal(order, polygon_order)
#
#
#
# @pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
# @pytest.mark.parametrize('flag_1', [True, False])
# @pytest.mark.parametrize('flag_2', [True, False])
# @pytest.mark.parametrize('flag_3', [True, False])
# @pytest.mark.parametrize('active_by_default', [True, False])
# def test_group_flags(sample_points,
#                      simple_square, simple_circle, semi_circle,
#                      flag_1, flag_2, flag_3, active_by_default):
#
#     polygon_1, _, _ = simple_square
#     polygon_2, _, _ = simple_circle
#     polygon_3, _, _ = semi_circle
#
#     polygons = [
#         FlaggedPolygon(polygon_1.lat_points, polygon_1.lon_points,
#                        flag_1, randint(0, 100)),
#         FlaggedPolygon(polygon_2.lat_points, polygon_2.lon_points,
#                        flag_2, randint(0, 100)),
#         FlaggedPolygon(polygon_3.lat_points, polygon_3.lon_points,
#                        flag_3, randint(0, 100)),
#     ]
#
#     group = FlaggedPolygonGroup(polygons, active_by_default)
#     flags = group.get_point_flags(*sample_points)
#
#     check_flags = np.ones_like(flags) * active_by_default
#     for p in group.polygons:
#         i = p.is_points_inside(*sample_points)
#         if np.any(i):
#             check_flags[i] = p.is_active_area
#
#     np.testing.assert_array_equal(flags, check_flags)
#
#
# @pytest.mark.parametrize('sample_points', [10, 100, 1000], indirect=True)
# @pytest.mark.parametrize('flag_1', [True, False])
# @pytest.mark.parametrize('flag_2', [True, False])
# @pytest.mark.parametrize('flag_3', [True, False])
# @pytest.mark.parametrize('active_by_default', [True, False])
# def test_group_flags_vec(sample_points,
#                          simple_square, simple_circle, semi_circle,
#                          flag_1, flag_2, flag_3, active_by_default):
#
#     polygon_1, _, _ = simple_square
#     polygon_2, _, _ = simple_circle
#     polygon_3, _, _ = semi_circle
#
#     polygons = [
#         FlaggedPolygon(polygon_1.lat_points, polygon_1.lon_points,
#                        flag_1, randint(0, 100)),
#         FlaggedPolygon(polygon_2.lat_points, polygon_2.lon_points,
#                        flag_2, randint(0, 100)),
#         FlaggedPolygon(polygon_3.lat_points, polygon_3.lon_points,
#                        flag_3, randint(0, 100)),
#     ]
#
#     group = FlaggedPolygonGroup(polygons, active_by_default)
#     flags = group.get_point_flags(*sample_points)
#
#     check_flags = np.ones_like(flags) * active_by_default
#     for p in group.polygons:
#         for i, (lat, lon) in enumerate(zip(*sample_points)):
#             if p.is_point_inside(lat, lon):
#                 check_flags[i] = p.is_active_area
#
#     np.testing.assert_array_equal(flags, check_flags)
