"""Configurable gravity correction maps."""

from pathlib import Path
from typing import Optional

import numpy as np

from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from qnav.earth.geoid import GeoidModel
from qnav.gravity.base import GravityModel
from qnav.gravity.custom_data import LoadedCustomMap, load_custom_map_data
from qnav.gravity.map import GravityMap
from qnav.input.ini.custom_gravity_config import (
    CustomGravityConfig,
    read_custom_gravity_config,
)


class CustomGravityMap(GravityMap):
    """
    A rectilinear gravity residual map described entirely by INI settings.

    QNav positions are supplied in WGS-84 latitude/longitude coordinates.
    Queries are transformed into the configured map CRS, interpolated in that
    native CRS, and returned as NED acceleration residuals in SI units.
    """

    __slots__ = (
        "_name",
        "_config",
        "_data",
        "_query_transformer",
        "_interpolator",
        "_out_of_bounds",
    )

    def __init__(
        self,
        base_model: GravityModel,
        geoid_model: Optional[GeoidModel],
        config: CustomGravityConfig | Path,
    ) -> None:
        if not isinstance(config, CustomGravityConfig):
            config = read_custom_gravity_config(Path(config))

        data = load_custom_map_data(config)
        super().__init__(
            base_model,
            geoid_model,
            config.config_file.parent,
            auto_download=False,
        )

        self._name = config.name
        self._config = config
        self._data = data
        self._out_of_bounds = config.out_of_bounds
        self._query_transformer = Transformer.from_crs(
            "EPSG:4326", data.crs, always_xy=True)
        self._interpolator = RegularGridInterpolator(
            (data.x, data.y),
            data.residual,
            method=config.interpolation,
            bounds_error=False,
            fill_value=np.nan,
        )

    def __str__(self) -> str:
        return self._name

    def get_residual(
        self,
        lat: float,
        lon: float,
    ) -> np.ndarray:
        """Return the interpolated NED acceleration residual."""

        result = self._query_residual(lat, lon)
        return np.asarray(result, dtype=np.float64)

    def get_residual_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """Vectorized NED acceleration-residual lookup."""

        return self._query_residual(lat, lon)

    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:
        residual = self.get_residual(lat, lon)
        return float(
            self._base_model.calc_gravity_xyz(lat, lon, alt)[2]
            + residual[2]
        )

    def calc_gravity_z_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
    ) -> np.ndarray:
        _validate_query_shapes(lat, lon, alt)
        residual = self.get_residual_vec(lat, lon)
        base = self._base_model.calc_gravity_xyz_vec(lat, lon, alt)
        return base[..., 2] + residual[..., 2]

    def calc_gravity_xyz(
        self,
        lat: float,
        lon: float,
        alt: float,
    ) -> np.ndarray:
        base = self._base_model.calc_gravity_xyz(lat, lon, alt)
        return base + self.get_residual(lat, lon)

    def calc_gravity_xyz_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
    ) -> np.ndarray:
        _validate_query_shapes(lat, lon, alt)
        base = self._base_model.calc_gravity_xyz_vec(lat, lon, alt)
        return base + self.get_residual_vec(lat, lon)

    def get_disturbance(self, lat: float, lon: float) -> float:
        return float(self.get_residual(lat, lon)[2])

    def get_disturbance_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        return self.get_residual_vec(lat, lon)[..., 2]

    def get_anomaly(self, lat: float, lon: float) -> float:
        disturbance = self.get_disturbance(lat, lon)
        return self._disturbance_to_anomaly(lat, lon, disturbance)

    def get_anomaly_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        disturbance = self.get_disturbance_vec(lat, lon)
        return self._disturbance_to_anomaly_vec(lat, lon, disturbance)

    @property
    def config(self) -> CustomGravityConfig:
        """Return the parsed map configuration."""

        return self._config

    @property
    def map_data(self) -> LoadedCustomMap:
        """Return normalized axes and residual data."""

        return self._data

    def _query_residual(
        self,
        lat: float | np.ndarray,
        lon: float | np.ndarray,
    ) -> np.ndarray:
        lat_values = np.asarray(lat, dtype=np.float64)
        lon_values = np.asarray(lon, dtype=np.float64)
        if lat_values.shape != lon_values.shape:
            raise ValueError("lat and lon must have the same shape")

        x_values, y_values = self._query_transformer.transform(
            lon_values, lat_values)
        x_values = _clip_transform_roundoff(
            np.asarray(x_values, dtype=np.float64), self._data.x)
        y_values = _clip_transform_roundoff(
            np.asarray(y_values, dtype=np.float64), self._data.y)
        points = np.column_stack((
            x_values.ravel(),
            y_values.ravel(),
        ))
        residual = np.asarray(
            self._interpolator(points), dtype=np.float64)
        invalid = ~np.all(np.isfinite(residual), axis=1)

        if np.any(invalid):
            if self._out_of_bounds == "error":
                count = int(np.count_nonzero(invalid))
                raise ValueError(
                    f"Custom gravity map '{self._name}' cannot provide "
                    f"{count} requested position(s).")
            residual[invalid, :] = 0.0

        result_shape = lat_values.shape + (3,)
        return residual.reshape(result_shape)


def _validate_query_shapes(
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
) -> None:
    if np.shape(lat) != np.shape(lon) or np.shape(lat) != np.shape(alt):
        raise ValueError("lat, lon and alt must have the same shape")


def _clip_transform_roundoff(
    values: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """
    Clip only floating-point CRS round-off at interpolation boundaries.

    A projected grid node transformed to WGS-84 and back can land a few ULPs
    outside its original bound.  This should remain an in-coverage query, but
    genuine out-of-map coordinates must not be extrapolated.
    """

    lower = float(axis[0])
    upper = float(axis[-1])
    scale = max(1.0, abs(lower), abs(upper))
    tolerance = np.finfo(np.float64).eps * scale * 64
    return np.where(
        (values < lower) & (values >= lower - tolerance),
        lower,
        np.where(
            (values > upper) & (values <= upper + tolerance),
            upper,
            values,
        ),
    )
