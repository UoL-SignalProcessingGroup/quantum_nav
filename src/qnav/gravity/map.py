"""
======
map.py
======

:summary:
    Module provides template implementation for gravity correction maps.
    Due to their high resolution and spatial complexity, methods from
    the original gravity model base class have had to be adjusted.
    The largest feature provided by this module is the method for
    correctly approximating the vertical gravity gradient.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

import math
import numpy as np

from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import Optional, Self
from abc import ABC, abstractmethod
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

from qnav.earth.geoid import GeoidModel
from qnav.earth.geoid import GeoidDummy
from qnav.gravity.base import GravityModel
from qnav.gravity.base import calculate_curvature
from qnav.util import transformations as trans
from qnav.util.constants import OMEGA_E


@dataclass
class CircularGrid:
    """
    Definition for a circular reference grid from a centre position.
    Circular grid points used for involving the surrounding data for
    accurately computing the vertical gravity gradient.
    """
    num_rings: int = 100         # Number of radial distance steps  (was 1000)
    num_angles: int = 50         # Number of azimuth angle steps
    max_radius: float = 0.25     # Maximum radial distance (in degrees) (was 0.5)
    l_threshold: float = 1.0     # Threshold for internal points (in metres)


    def generate_grid(self, lat: float, lon: float, alt: float) -> dict[str, np.ndarray]:
        """
        Generates a circular grid of query positions surrounding a point.
        On succession, a dictionary containing the positional and
        angular information of points surrounding a position of interest are
        returned. This also includes commonly calculated values step size
        values.

        :param lat: The latitude position of the point (in decimal degrees).
        :type lat: float

        :param lon: The longitude position of the point (in decimal degrees).
        :type lon: float

        :param alt: The altitude position of the point (in metres).
        :type alt: float

        :return: A dictionary holding the calculated positions (lla),
            NED offsets (ned), azimuth angles (alpha), distances from centre
             (psi) and step sizes (delta_psi and delta_alpha).
        :rtype: dict[str, np.ndarray]
        """

        # Compute the psi steps
        psi_max = math.radians(self.max_radius)
        psi = np.linspace(0, psi_max, self.num_rings)  # TODO: CHECK Change!

        # Compute the alpha steps
        alpha = np.linspace(0, 2 * math.pi, self.num_angles + 1)
        alpha = alpha[:-1]

        psi, alpha = np.meshgrid(psi, alpha, indexing='ij')  # TODO: CHECK!
        req_shape = psi.shape

        # Compute the reference NED vector for each
        xyz = np.column_stack([
            np.cos(psi.ravel('F')),
            np.sin(psi.ravel('F')) * np.sin(alpha.ravel('F')),
            np.sin(psi.ravel('F')) * np.cos(alpha.ravel('F')),
        ])

        sin_lat = math.sin(math.radians(lat))
        cos_lat = math.cos(math.radians(lat))
        sin_lon = math.sin(math.radians(lon))
        cos_lon = math.cos(math.radians(lon))

        rot_matrix = np.array([
            [cos_lat * cos_lon, -sin_lon, -sin_lat * cos_lon],
            [cos_lat * sin_lon,  cos_lon, -sin_lat * sin_lon],
            [sin_lat, 0.0, cos_lat]
        ])

        xyz = (rot_matrix @ xyz.T).T
        xyz = np.radians(xyz)

        lats = np.atan2(xyz[:, 2], np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)).reshape(req_shape, order='F')
        lons = np.atan2(xyz[:, 1], xyz[:, 0]).reshape(req_shape, order='F')
        alts = np.zeros_like(lats) + alt

        lla_grid = np.stack([np.degrees(lats), np.degrees(lons), alts], axis=-1)
        # ned_grid = 0

        # north = np.reshape(ned[:, 0], req_shape)
        # east = np.reshape(ned[:, 1], req_shape)
        # down = np.reshape(ned[:, 2], req_shape)
        # ned_grid = np.stack([north, east, down], axis=-1)

        # ref_lla = np.array([lat, lon, alt])
        # lla = trans.ned2lla_vec(ned, ref_lla)
        #
        # latitude = np.reshape(lla[:, 0], req_shape)
        # longitude = np.reshape(lla[:, 1], req_shape)
        # altitude = np.reshape(lla[:, 2], req_shape)
        # lla_grid = np.stack([latitude, longitude, altitude], axis=-1)

        # TODO: psi[1,0] - psi[0,0],       alpha[0,1] - alpha[0,0]
        # delta_psi = float(np.max(psi) / self.num_rings)
        # delta_alpha = 2 * math.pi / self.num_angles

        # NEW
        delta_psi = psi[1,0] - psi[0,0]
        delta_alpha = alpha[0,1] - alpha[0,0]

        return {
            'lla': lla_grid,
            'alpha': alpha,
            'psi': psi,
            'delta_psi': delta_psi,
            'delta_alpha': delta_alpha,
        }

# The default circular grid
_DEFAULT_GRID: CircularGrid = CircularGrid()

def set_default_grid(num_rings: int, num_angles: int, max_radius: float, l_threshold: float):
    """
    Overwrites the settings to use for the default circular reference grid.
    This is the grid that will be used by default for calculating the
    vertical gravity gradient.

    :param num_rings: Number of radial distance steps.
    :type num_rings: int

    :param num_angles: Number of azimuth angle steps.
    :type num_angles: int

    :param max_radius: Maximum radial distance (in degrees).
    :type max_radius: float

    :param l_threshold: Threshold for internal points (in metres).
    :type l_threshold: float
    """
    _DEFAULT_GRID.num_rings = num_rings
    _DEFAULT_GRID.num_angles = num_angles
    _DEFAULT_GRID.max_radius = max_radius
    _DEFAULT_GRID.l_threshold = l_threshold


class GravityMap(GravityModel, ABC):
    """
    An extension to the abstract base class GravityModel that provides
    re-worked methods for handling high-resolution lookup maps.
    """

    # All class properties used by the abstract base class
    __slots__ = '_base_model', '_geoid_model', '_map_dir', '_auto_download'

    def __init__(self, base_model: GravityModel,
                 geoid_model: Optional[GeoidModel],
                 map_dir: Optional[Path],
                 auto_download: bool = True):
        """
        The abstract class that advanced gravity models are to follow.
        Creates a gravity map based on a base function, geoid model and
        downloadable correction values.

        :param base_model: Instance of base gravity function. This is the
            base function that the map will apply corrections for.
        :type base_model: GravityModel

        :param map_dir: The directory holding the gravity map files.
            This is where gravity data is to be written to and loaded from.
        :type map_dir: Path

        :param geoid_model: Geoid height model for altitude corrections.
            Used to determine the height correction (from reference
            ellipsoid) towards the an equipotential gravitational surface.
        :type geoid_model: GeoidModel

        :param auto_download: Whether auto-downloading is to be permitted.
        :type auto_download: bool
        """
        self._base_model = base_model
        self._geoid_model = geoid_model
        self._map_dir = map_dir
        self._auto_download = auto_download

        if geoid_model is None:
            self._geoid_model = GeoidDummy()

    @abstractmethod
    def get_anomaly(self, lat: float, lon: float) -> float:
        """
        Calculates the gravity anomaly for given position.
        Retrieves the gravity anomaly value for the requested position
        (in m/s^2) at the base surface. For gravity maps this value is
        interpolated from its dataset, following possible conversion.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :return: The gravity anomaly value (in m/s^2).
        :rtype: float
        """
        pass

    @abstractmethod
    def get_disturbance(self, lat: float, lon: float) -> float:
        """
        Calculates the gravity disturbance for given position.
        Retrieves the gravity disturbance value for the requested position
        (in m/s^2) at the base surface. For gravity maps this value is
        interpolated from its dataset, following possible conversion.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :return: The gravity disturbance value (in m/s^2).
        :rtype: float
        """
        pass

    @abstractmethod
    def get_anomaly_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Calculates the gravity anomaly for multiple given positions.
        Retrieves the gravity anomaly value for the requested position
        (in m/s^2) at the base surface. For gravity maps this value is
        interpolated from its dataset, following possible conversion.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :return: The gravity anomaly value (in m/s^2).
        :rtype: float
        """
        pass

    @abstractmethod
    def get_disturbance_vec(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Calculates the gravity disturbance for multiple given positions.
        Retrieves the gravity disturbance value for the requested position
        (in m/s^2) at the base surface. For gravity maps this value is
        interpolated from its dataset, following possible conversion.

        :param lat: The latitude coordinate for the position of interest.
            The latitude position value given in decimal degrees.
        :type lat: float

        :param lon: The longitude coordinate for the position of interest.
            The longitude position value given in decimal degrees.
        :type lon: float

        :return: The gravity disturbance value (in m/s^2).
        :rtype: float
        """
        pass

    def calc_gravity_xyz(self, lat: float, lon: float, alt: float) -> np.ndarray:
        [dov_north, dov_east] = self._geoid_model.get_dov(lat, lon)
        g_vec = self._base_model.calc_gravity_xyz(lat, lon, alt)
        g_norm = np.sqrt(np.sum(g_vec * g_vec))
        g_vec[0] += g_norm * dov_north
        g_vec[1] += g_norm * dov_east
        return g_vec

    def calc_gravity_xyz_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray) -> np.ndarray:
        [dov_north, dov_east] = self._geoid_model.get_dov_vec(lat, lon)
        g_vec = self._base_model.calc_gravity_xyz_vec(lat, lon, alt)
        g_norm = np.sqrt(np.sum(g_vec * g_vec, 1))
        g_vec[:, 0] += g_norm * dov_north
        g_vec[:, 1] += g_norm * dov_east
        return g_vec

    def _anomaly_to_disturbance(self, lat: float, lon: float, g_anom: float, dh: float = 100) -> float:
        geoid_height = self._geoid_model.get_height(lat, lon)
        gamma = partial(self._base_model.calc_gravity_z, lat, lon)
        d_gamma_dh = (gamma(dh) - gamma(0)) / dh
        return g_anom - d_gamma_dh * geoid_height

    def _anomaly_to_disturbance_vec(self, lat: np.ndarray, lon: np.ndarray, g_anom: np.ndarray, dh: float = 100) -> np.ndarray:
        dh_vec = np.zeros(lat.shape) + dh
        geoid_height = self._geoid_model.get_height_vec(lat, lon)
        gamma = partial(self._base_model.calc_gravity_z_vec, lat, lon)
        d_gamma_dh = (gamma(dh_vec) - gamma(0)) / dh_vec
        return g_anom - d_gamma_dh * geoid_height

    def _disturbance_to_anomaly(self, lat: float, lon: float, g_dist: float, dh: float = 100) -> float:
        geoid_height = self._geoid_model.get_height(lat, lon)
        gamma = partial(self._base_model.calc_gravity_z, lat, lon)
        d_gamma_dh = (gamma(dh) - gamma(0)) / dh
        return g_dist + d_gamma_dh * geoid_height

    def _disturbance_to_anomaly_vec(self, lat: np.ndarray, lon: np.ndarray, g_dist: np.ndarray, dh: float = 100) -> np.ndarray:
        dh_vec = np.zeros(lat.shape) + dh
        geoid_height = self._geoid_model.get_height_vec(lat, lon)
        gamma = partial(self._base_model.calc_gravity_z_vec, lat, lon)
        d_gamma_dh = (gamma(dh_vec) - gamma(0)) / dh_vec
        return g_dist + d_gamma_dh * geoid_height

    def _get_anomaly_grad(self, lat: float, lon: float, alt: float, dh: float = 100) -> float:
        """
        A internal helper function to calculate the gravity anomaly gradient.
        For given position, this function approximates the gradient to be
        used for extrapolating the gravity anomaly for heights above the
        reference position. This is commonly used by implementations.
        """

        re = trans.radius(lat)
        n = self._geoid_model.get_height(lat, lon)
        gamma = partial(self._base_model.calc_gravity_z, lat, lon)
        d_gamma_dh = (gamma(alt + dh) - gamma(alt)) / dh
        d2_gamma_dh2 = (gamma(alt + dh) - 2 * gamma(alt) + gamma(alt - dh)) / (dh * dh)
        return -n * d2_gamma_dh2 - ((2 * n) / (re + alt)) * d_gamma_dh

    def _get_anomaly_grad_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, dh: float = 100) -> float:
        """
        A internal helper function to calculate the gravity anomaly gradient.
        For given position, this function approximates the gradient to be
        used for extrapolating the gravity anomaly for heights above the
        reference position. This is commonly used by implementations.
        """

        re = trans.radius_vec(lat)
        n = self._geoid_model.get_height_vec(lat, lon)
        gamma = partial(self._base_model.calc_gravity_z_vec, lat, lon)
        d_gamma_dh = (gamma(alt + dh) - gamma(alt)) / dh
        d2_gamma_dh2 = (gamma(alt + dh) - 2 * gamma(alt) + gamma(alt - dh)) / (dh * dh)
        return -n * d2_gamma_dh2 - ((2 * n) / (re + alt)) * d_gamma_dh

    def _interp_anomaly(self, lat: float, lon: float, alt: float, g_anom: float) -> float:
        anom_grad = self._get_anomaly_grad(lat, lon, alt)
        geoid_height = self._geoid_model.get_height(lat, lon)
        return g_anom + anom_grad * (alt - geoid_height)

    def _interp_anomaly_vec(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, g_anom: np.ndarray) -> np.ndarray:
        anom_grad = self._get_anomaly_grad_vec(lat, lon, alt)
        geoid_height = self._geoid_model.get_height_vec(lat, lon)
        return g_anom + anom_grad * (alt - geoid_height)

    def calc_vertical_grad(self, lat: float, lon: float, alt: float, step: float = 100,
                           circular_grid: CircularGrid = None) -> float:

        # TODO: Make circular grid optional, default to gravity function if missing?

        # Create circular grid if one is missing
        if circular_grid is None:
            # circular_grid = CircularGrid()
            circular_grid = _DEFAULT_GRID

        # Generate the reference grid
        grid_values = circular_grid.generate_grid(lat, lon, alt)
        lla_grid = grid_values['lla']
        lats = lla_grid[:, :, 0]
        lons = lla_grid[:, :, 1]
        alts = lla_grid[:, :, 2]

        # GOAL Calculate: (∂g/∂r) = (∂γ/∂r) + (∂Δg/∂r)
        #   - part_1: (∂γ/∂r)
        #   - part_2: (∂Δg/∂r)

        # Calculate (∂γ/∂r) = −2γj −2ω²
        omega_e = OMEGA_E
        gamma_p = self._base_model.calc_gravity_z_vec(lats, lons, alts)
        j = calculate_curvature(lats)
        part_1 = (-2 * gamma_p * j) - (2 * omega_e * omega_e)

        # Calculate Δg(q) = ∂g(q) - (hq-hp)(∂g/∂h)
        grav_anom = self.get_anomaly_vec(lats, lons)
        delta_g = grav_anom + alts * part_1
        delta_g_p = delta_g[0, 0]

        r = trans.radius(lat)
        psi = grid_values['psi']
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)

        l = np.sqrt(r ** 2 + (r + alts) ** 2 - 2 * r * (r + alts) * cos_psi)
        part_2 = ((delta_g - delta_g_p) / l ** 3)
        part_2[np.isnan(part_2)] = 0

        # Grid step sizes
        delta_psi = grid_values['delta_psi']
        delta_alpha = grid_values['delta_alpha']
        delta_sigma = sin_psi * delta_psi * delta_alpha

        # Define the inside and outside bounds
        l_inside = l < circular_grid.l_threshold
        l_outside = ~l_inside

        # Calculate (∂Δg/∂r) = Σi((Δg(i)-Δgₚ)/l³(i))Δσ(i) -(2/R)Δgₚ
        part_2_sum = np.nansum(part_2[l_outside] * delta_sigma[l_outside])
        part_2_sum *= (r ** 2 / (2 * np.pi))
        part_2_sum -= (2 / r) * delta_g_p

        # TODO: Optimise and check
        part_3_sum = 0
        if np.any(l_inside):
            so = trans.radius_vec(lats) * sin_psi  # TODO: CHECK
            grid_grads = _get_grid_gradients(grid_values, delta_g)
            grid_grads *= (so / 4)
            part_3_sum = np.nanmean(grid_grads[l_inside])

        # TODO: Why this negative
        # vgg = -(part_1[0, 0] + part_2_sum + part_3_sum)
        vgg = (part_1[0, 0] + part_2_sum + part_3_sum)
        return float(vgg)

    def calc_vertical_grad_vec(self, lat: np.ndarray, lon: np.ndarray,
                               alt: np.ndarray, step: float = 100,
                               circular_grid: CircularGrid = None) -> np.ndarray:

        # # Create circular grid if one is missing
        # if circular_grid is None:
        #     # circular_grid = CircularGrid()
        #     circular_grid = _DEFAULT_GRID

        if np.shape(lat) != np.shape(lon) != np.shape(alt):
            raise ValueError("lat, lon and alt must have same shape")

        # TODO: Please actually vectorise this!
        vgg = np.zeros(lat.size)
        positions = np.column_stack((lat.ravel(), lon.ravel(), alt.ravel()))
        for i, pos in enumerate(positions):
            vgg[i] = self.calc_vertical_grad(
                *pos, step=step, circular_grid=circular_grid)

        return vgg.reshape(lat.shape)

    @property
    def base_model(self) -> Self | None:
        """
        Returns the base gravity model the model extends from.
        Allows access to the base function/model the instance is using.

        :return: The base gravity model.
        :rtype: GravityModel | None
        """
        return self._base_model

def _get_grid_gradients(grid_values: dict,
                        delta_g: np.ndarray,
                        step_size: float = 45) -> np.ndarray:

    lla_grid = grid_values['lla']

    lla_points = np.column_stack([
        lla_grid[:, :, 0].ravel(),
        lla_grid[:, :, 1].ravel(),
        lla_grid[:, :, 2].ravel(),
    ])

    # Obtain north and east NED offsets
    ned_x = np.array([0.0, 1.0, 0.0]) * step_size
    ned_y = np.array([1.0, 0.0, 0.0]) * step_size

    lla_q1 = trans.ned2lla_vec(ned_x, lla_points)
    lla_q2 = trans.ned2lla_vec(-ned_x, lla_points)
    lla_q3 = trans.ned2lla_vec(ned_y, lla_points)
    lla_q4 = trans.ned2lla_vec(-ned_y, lla_points)

    # Generate an interpolation instance for the geosat data.
    values = delta_g.ravel()
    tri = Delaunay(lla_points[:, 0:2])
    interp = LinearNDInterpolator(tri, values, fill_value=0.0)

    q1 = interp(lla_q1[:, 0], lla_q1[:, 1])
    q2 = interp(lla_q2[:, 0], lla_q2[:, 1])
    q3 = interp(lla_q3[:, 0], lla_q3[:, 1])
    q4 = interp(lla_q4[:, 0], lla_q4[:, 1])

    req_shape = delta_g.shape
    q_sum = np.reshape(q1 + q2 + q3 + q4, req_shape)
    step_size_sq = step_size * step_size
    part_1 = q_sum / step_size_sq
    part_1[np.isnan(part_1)] = 0

    part_2 = (4 * delta_g) / step_size_sq
    return part_1 - part_2
