
import pytest
import numpy as np
import scipy as sp

from qnav.earth.geoid import GeoidPGM
from qnav.gravity.map import CircularGrid
from qnav.gravity.somigliana import Somigliana
from qnav.gravity.srtm2gravity import SRTM2GravityFull

import qnav.util.transformations as trans


@pytest.fixture(scope='function')
def position() -> np.ndarray:
    return np.array([52.0, -3.0, 1000.0])

@pytest.fixture(scope='function')
def gravity_func():
    return Somigliana()

@pytest.fixture(scope='function')
def geoid_model(project_dir):
    geoid_file = project_dir / 'databases' / 'geoid' / 'egm2008-5.pgm'
    return GeoidPGM(geoid_file)

@pytest.fixture(scope='function')
def gravity_model(gravity_func, geoid_model, project_dir):
    allow_downloads = False
    srtm_dir = project_dir / 'databases' / 'gravity' / 'srtm2gravity'
    return SRTM2GravityFull(gravity_func, geoid_model, srtm_dir, allow_downloads)

@pytest.fixture(scope='function')
def circular_grid() -> CircularGrid :
    return CircularGrid(
        num_rings = 100,        # Number of radial distance steps  (was 1000)
        num_angles = 50,        # Number of azimuth angle steps
        max_radius = 0.05,       # Maximum radial distance (in degrees) (was 0.5)
        l_threshold = 100.0       # Threshold for internal points (in metres)
    )

def test_gravity_gradient(position, gravity_model, circular_grid):

    step_size = 10
    lat, lon, alt = position

    dgz_gz = gravity_model.calc_vertical_grad(
        lat, lon, alt, step_size, circular_grid)

    step = 100.0
    north_offset = trans.ned2lla(np.array([1.0, 0.0, 0.0]) * step, position)
    east_offset = trans.ned2lla(np.array([0.0, 1.0, 0.0]) * step, position)
    down_offset = trans.ned2lla(np.array([0.0, 0.0, 1.0]) * step, position)

    dgz_gz_north = gravity_model.calc_vertical_grad(
        *north_offset, step_size, circular_grid)

    dgz_gz_east = gravity_model.calc_vertical_grad(
        *east_offset, step_size, circular_grid)

    dgz_gz_down = gravity_model.calc_vertical_grad(
        *down_offset, step_size, circular_grid)

    print(f'\nCentre: \t {dgz_gz}')
    # print(f'North diff: \t {dgz_gz - dgz_gz_north}')
    # print(f'East diff: \t {dgz_gz - dgz_gz_east}')
    # print(f'Down diff: \t {dgz_gz - dgz_gz_down}')

    print(f'North: \t\t {dgz_gz_north} \t ({(dgz_gz / dgz_gz_north) - 1} %)')
    print(f'East: \t\t {dgz_gz_east} \t ({(dgz_gz / dgz_gz_north) - 1} %)')
    print(f'Down: \t\t {dgz_gz_down} \t ({(dgz_gz / dgz_gz_north) - 1} %)')


def test_gravity_similarity(position, gravity_model, circular_grid):

    spread = 100
    amount = 1000

    rand_ned = np.random.random((amount, 3)) * spread
    rand_lla = trans.ned2lla_vec(rand_ned, position)

    step_size = 30.0
    centre_grad = gravity_model.calc_vertical_grad(
        *position, step_size, circular_grid)

    rand_grad = gravity_model.calc_vertical_grad_vec(
        rand_lla[:, 0], rand_lla[:, 1], rand_lla[:, 2],
        step_size, circular_grid)

    dist = np.linalg.norm(rand_ned, axis=1)
    index_i = np.argsort(dist)

    diff = np.abs(centre_grad - rand_grad)
    index_j = np.argsort(diff)

    matches = np.sum(index_i == index_j)
    # error = np.mean(np.abs(index_i - index_j))
    error = sp.stats.pearsonr(index_i, index_j)

    # print(centre_grad)
    # print(rand_grad)

    print(f'Num Matches: {matches}')
    print(f'Error Score: {error}')


def test_grid(position, gravity_model, circular_grid):

    step_size = 10
    grid_size = 102
    grid_range = 1000

    grid_steps = np.linspace(-grid_range, grid_range, grid_size)
    north_offset, east_offsets = np.meshgrid(grid_steps, grid_steps, indexing='ij')
    down_offset = np.zeros_like(north_offset)

    ned = np.column_stack((north_offset.flatten(), east_offsets.flatten(), down_offset.flatten()))
    lla = trans.ned2lla_vec(ned, position)
    lla[:, 2] = position[2]

    centre_grad = gravity_model.calc_vertical_grad(
        position[0], position[1], position[2], step_size, circular_grid)

    grid_grad = gravity_model.calc_vertical_grad_vec(
        lla[:, 0], lla[:, 1], lla[:, 2], step_size, circular_grid)

    # centre_grad = gravity_model.calc_gravity_z(
    #     position[0], position[1], position[2])
    #
    # grid_grad = gravity_model.calc_gravity_z_vec(
    #     lla[:, 0], lla[:, 1], lla[:, 2])

    grid_grad = grid_grad.reshape((grid_size, grid_size))
    a = abs(centre_grad - grid_grad)
    ranks = (a - a.min()) / (a.max() - a.min())

    print(f'{np.where(ranks == 0)}')



def test_similarity(position, gravity_model, circular_grid):

    spread = 2.0
    amount = 100
    step_size = 30.0

    rand_ned = np.random.random((amount, 3)) * spread
    offset = np.array([0.0, 10.0, 0.0])
    rand_ned += offset

    rand_lla = trans.ned2lla_vec(rand_ned, position)

    centre_grad = gravity_model.calc_vertical_grad(
        *position, step_size, circular_grid)

    rand_grad = gravity_model.calc_vertical_grad_vec(
        rand_lla[:, 0], rand_lla[:, 1], rand_lla[:, 2],
        step_size, circular_grid)

    print(abs(centre_grad - np.mean(rand_grad)))
