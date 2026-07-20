"""Configurable gravity correction maps."""

from pathlib import Path
from typing import Optional

import numpy as np

from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from qnav.earth.geoid import GeoidModel
from qnav.gravity.base import GravityModel
from qnav.gravity.custom_data import (
    LoadedCustomMap,
    geographic_longitude_period,
    load_custom_map_data,
)
from qnav.gravity.map import GravityMap
from qnav.input.ini.custom_gravity_config import (
    CustomGravityConfig,
    read_custom_gravity_config,
)
from qnav.input.config_handler import NavConfigError


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

        if (
            config.field.representation == "scalar"
            and config.field.quantity == "free_air_anomaly"
            and geoid_model is None
        ):
            raise NavConfigError(
                "Field",
                "quantity",
                "Missing geoid model",
                "A free-air anomaly map requires a configured geoid model "
                "for anomaly-to-disturbance and altitude corrections.",
            )

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

        values, valid = self._query_map(lat, lon)
        if self._data.quantity == "free_air_anomaly":
            result = np.zeros(3, dtype=np.float64)
            result[2] = self._anomaly_to_disturbance(
                lat, lon, float(values[2])) if bool(valid) else 0.0
            return result
        return np.asarray(values, dtype=np.float64)

    def get_residual_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """Vectorized NED acceleration-residual lookup."""

        values, valid = self._query_map(lat, lon)
        if self._data.quantity != "free_air_anomaly":
            return values
        result = np.zeros_like(values)
        valid_flat = valid.ravel()
        if np.any(valid_flat):
            result[..., 2].reshape(-1)[valid_flat] = (
                self._anomaly_to_disturbance_vec(
                    np.asarray(lat, dtype=np.float64).ravel()[valid_flat],
                    np.asarray(lon, dtype=np.float64).ravel()[valid_flat],
                    values[..., 2].ravel()[valid_flat],
                )
            )
        return result

    def calc_gravity_z(self, lat: float, lon: float, alt: float) -> float:
        if self._data.quantity == "free_air_anomaly":
            values, valid = self._query_map(lat, lon)
            correction = (
                self._interp_anomaly(lat, lon, alt, float(values[2]))
                if bool(valid)
                else 0.0
            )
            return float(
                self._base_model.calc_gravity_z(lat, lon, alt) + correction)
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
        if self._data.quantity == "free_air_anomaly":
            values, valid = self._query_map(lat, lon)
            correction = self._interpolate_valid_anomalies(
                lat, lon, alt, values[..., 2], valid)
            return self._base_model.calc_gravity_z_vec(
                lat, lon, alt) + correction
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
        if self._data.quantity == "free_air_anomaly":
            values, valid = self._query_map(lat, lon)
            if bool(valid):
                base[2] += self._interp_anomaly(
                    lat, lon, alt, float(values[2]))
            return base
        return base + self.get_residual(lat, lon)

    def calc_gravity_xyz_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
    ) -> np.ndarray:
        _validate_query_shapes(lat, lon, alt)
        base = self._base_model.calc_gravity_xyz_vec(lat, lon, alt)
        if self._data.quantity == "free_air_anomaly":
            values, valid = self._query_map(lat, lon)
            base[..., 2] += self._interpolate_valid_anomalies(
                lat, lon, alt, values[..., 2], valid)
            return base
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
        if self._data.quantity == "free_air_anomaly":
            values, _ = self._query_map(lat, lon)
            return float(values[2])
        _, valid = self._query_map(lat, lon)
        if not bool(valid):
            return 0.0
        disturbance = self.get_disturbance(lat, lon)
        return self._disturbance_to_anomaly(lat, lon, disturbance)

    def get_anomaly_vec(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        values, valid = self._query_map(lat, lon)
        if self._data.quantity == "free_air_anomaly":
            return values[..., 2]
        anomaly = np.zeros_like(values[..., 2])
        valid_flat = valid.ravel()
        if np.any(valid_flat):
            anomaly.reshape(-1)[valid_flat] = (
                self._disturbance_to_anomaly_vec(
                    np.asarray(lat, dtype=np.float64).ravel()[valid_flat],
                    np.asarray(lon, dtype=np.float64).ravel()[valid_flat],
                    values[..., 2].ravel()[valid_flat],
                )
            )
        return anomaly

    @property
    def config(self) -> CustomGravityConfig:
        """Return the parsed map configuration."""

        return self._config

    @property
    def map_data(self) -> LoadedCustomMap:
        """Return normalized axes and residual data."""

        return self._data

    def _interpolate_valid_anomalies(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        alt: np.ndarray,
        anomaly: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Apply altitude conversion without querying invalid map points."""

        correction = np.zeros_like(anomaly)
        valid_flat = valid.ravel()
        if np.any(valid_flat):
            correction.reshape(-1)[valid_flat] = self._interp_anomaly_vec(
                np.asarray(lat, dtype=np.float64).ravel()[valid_flat],
                np.asarray(lon, dtype=np.float64).ravel()[valid_flat],
                np.asarray(alt, dtype=np.float64).ravel()[valid_flat],
                anomaly.ravel()[valid_flat],
            )
        return correction

    def _query_map(
        self,
        lat: float | np.ndarray,
        lon: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lat_values = np.asarray(lat, dtype=np.float64)
        lon_values = np.asarray(lon, dtype=np.float64)
        if lat_values.shape != lon_values.shape:
            raise ValueError("lat and lon must have the same shape")

        x_values, y_values = self._query_transformer.transform(
            lon_values, lat_values)
        x_values = np.asarray(x_values, dtype=np.float64)
        if self._data.crs.is_geographic:
            x_values = _normalize_periodic_axis(
                x_values,
                self._data.x,
                geographic_longitude_period(self._data.crs),
            )
        x_values = _clip_transform_roundoff(
            x_values, self._data.x)
        y_values = _clip_transform_roundoff(
            np.asarray(y_values, dtype=np.float64), self._data.y)
        points = np.column_stack((
            x_values.ravel(),
            y_values.ravel(),
        ))
        finite_points = np.all(np.isfinite(points), axis=1)
        residual = np.full((points.shape[0], 3), np.nan, dtype=np.float64)
        if np.any(finite_points):
            residual[finite_points] = np.asarray(
                self._interpolator(points[finite_points]),
                dtype=np.float64,
            )
        invalid = ~np.all(np.isfinite(residual), axis=1)

        # Linear interpolation can propagate a neighbouring NaN even when a
        # query lies exactly on a valid grid node (zero times NaN is NaN).
        # Preserve the source value at exact nodes without interpolating
        # across genuinely missing cells.
        for index in np.flatnonzero(invalid & finite_points):
            x_index = _exact_axis_index(points[index, 0], self._data.x)
            y_index = _exact_axis_index(points[index, 1], self._data.y)
            if x_index is None or y_index is None:
                continue
            node_value = self._data.residual[x_index, y_index]
            if np.all(np.isfinite(node_value)):
                residual[index] = node_value
                invalid[index] = False

        if np.any(invalid):
            if self._out_of_bounds == "error":
                count = int(np.count_nonzero(invalid))
                raise ValueError(
                    f"Custom gravity map '{self._name}' cannot provide "
                    f"{count} requested position(s).")
            residual[invalid, :] = 0.0

        result_shape = lat_values.shape + (3,)
        valid_shape = lat_values.shape
        return residual.reshape(result_shape), (~invalid).reshape(valid_shape)


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


def _normalize_periodic_axis(
    values: np.ndarray,
    axis: np.ndarray,
    period: float,
) -> np.ndarray:
    """Choose the equivalent periodic coordinate nearest the map axis."""

    midpoint = (float(axis[0]) + float(axis[-1])) / 2.0
    normalized = values.copy()
    finite = np.isfinite(values)
    finite_values = values[finite]
    revolutions = np.round((midpoint - finite_values) / period)
    normalized[finite] = finite_values + revolutions * period
    return normalized


def _exact_axis_index(value: float, axis: np.ndarray) -> Optional[int]:
    """Return the index when a value is equal to a grid node within ULPs."""

    candidate = int(np.searchsorted(axis, value))
    for index in (candidate, candidate - 1):
        if index < 0 or index >= axis.size:
            continue
        scale = max(1.0, abs(value), abs(float(axis[index])))
        tolerance = np.finfo(np.float64).eps * scale * 64
        if abs(value - float(axis[index])) <= tolerance:
            return index
    return None
