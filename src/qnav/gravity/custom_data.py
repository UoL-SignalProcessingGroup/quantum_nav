"""Data loading and normalization for configurable gravity maps."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from scipy.io import loadmat

from qnav.input.config_handler import NavConfigError
from qnav.input.ini.custom_gravity_config import (
    CustomGravityConfig,
    GridSourceConfig,
    TensorSourceConfig,
    VectorSourceConfig,
)
from qnav.util import constants
from qnav.util.transformations import (
    ecef2ned_transform_vec,
    lla2ecef_vec,
)


_T = TypeVar("_T")


@dataclass(frozen=True)
class LoadedGrid:
    """Normalized axes and metadata for a rectilinear grid."""

    x: np.ndarray
    y: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    height: Optional[np.ndarray]
    ellipsoid_height: np.ndarray
    ecef: np.ndarray
    geocentric_latitude: np.ndarray
    flip_x: bool
    flip_y: bool


@dataclass(frozen=True)
class LoadedCustomMap:
    """A custom map normalized to NED residual acceleration in SI units."""

    x: np.ndarray
    y: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    height: Optional[np.ndarray]
    ellipsoid_height: np.ndarray
    residual: np.ndarray
    crs: CRS


class CustomMapDataLoader:
    """Load all configured map resources with a scoped MATLAB file cache."""

    def __init__(self, config: CustomGravityConfig):
        self._config = config
        self._mat_cache: dict[Path, dict[str, Any]] = {}

    def load(self) -> LoadedCustomMap:
        """Load, validate, and normalize the configured custom map."""

        crs = self._read_crs()
        grid = self._load_grid(crs)
        field = self._load_vector(self._config.field, grid)

        if self._config.mode == "total_minus_reference":
            reference = self._load_vector(self._config.reference, grid)
            residual = field - reference
        else:
            residual = field

        if self._config.tensor is not None:
            self._validate_tensor(self._config.tensor, grid)

        return LoadedCustomMap(
            x=grid.x,
            y=grid.y,
            latitude=grid.latitude,
            longitude=grid.longitude,
            height=grid.height,
            ellipsoid_height=grid.ellipsoid_height,
            residual=residual,
            crs=crs,
        )

    def _read_crs(self) -> CRS:
        try:
            crs = CRS.from_user_input(self._config.crs)
        except CRSError as error:
            raise NavConfigError(
                "CustomGravityMap",
                "crs",
                "Invalid CRS",
                f"The configured CRS '{self._config.crs}' could not be "
                f"loaded: {error}",
            ) from error
        if (
            crs.is_compound
            or crs.is_vertical
            or crs.is_geocentric
            or len(crs.axis_info) != 2
            or not (crs.is_projected or crs.is_geographic)
        ):
            raise NavConfigError(
                "CustomGravityMap",
                "crs",
                "Unsupported CRS",
                "Custom gravity grids require a two-dimensional horizontal "
                "projected or geographic CRS.",
            )
        return crs

    def _load_grid(self, crs: CRS) -> LoadedGrid:
        source = self._config.grid
        if source.format == "csv":
            x, y, height, geoid_undulation = self._load_csv_grid(source)
        else:
            x, y, height, geoid_undulation = self._load_mat_grid(source)

        x, flip_x = _normalize_axis(x, "x")
        y, flip_y = _normalize_axis(y, "y")
        if height is not None:
            if flip_x:
                height = np.flip(height, axis=0)
            if flip_y:
                height = np.flip(height, axis=1)
        if geoid_undulation is not None:
            if flip_x:
                geoid_undulation = np.flip(geoid_undulation, axis=0)
            if flip_y:
                geoid_undulation = np.flip(
                    geoid_undulation, axis=1)

        x_grid: np.ndarray
        y_grid: np.ndarray
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
        try:
            inverse = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            longitude, latitude = inverse.transform(x_grid, y_grid)
        except Exception as error:
            raise NavConfigError(
                "CustomGravityMap",
                "crs",
                "CRS transformation failed",
                f"Grid coordinates could not be transformed to WGS-84: "
                f"{error}",
            ) from error

        latitude = np.asarray(latitude, dtype=np.float64)
        longitude = np.asarray(longitude, dtype=np.float64)
        if not np.all(np.isfinite(latitude)) or not np.all(
                np.isfinite(longitude)):
            raise _data_error(
                source.file,
                "Grid coordinates transform to non-finite WGS-84 values.")

        if height is None:
            ellipsoid_height = np.zeros_like(latitude)
        elif source.height_reference == "orthometric":
            if geoid_undulation is None:
                raise _data_error(
                    source.file,
                    "Orthometric heights require geoid undulation values.",
                )
            ellipsoid_height = height + geoid_undulation
        else:
            ellipsoid_height = height
        if not np.all(np.isfinite(ellipsoid_height)):
            raise _data_error(
                source.file, "Grid ellipsoid heights must all be finite.")

        lla = np.column_stack((
            latitude.ravel(),
            longitude.ravel(),
            ellipsoid_height.ravel(),
        ))
        ecef = lla2ecef_vec(lla).reshape(latitude.shape + (3,))
        geocentric_latitude = np.degrees(np.arctan2(
            ecef[:, :, 2],
            np.hypot(ecef[:, :, 0], ecef[:, :, 1]),
        ))

        return LoadedGrid(
            x=x,
            y=y,
            latitude=latitude,
            longitude=longitude,
            height=height,
            ellipsoid_height=ellipsoid_height,
            ecef=ecef,
            geocentric_latitude=geocentric_latitude,
            flip_x=flip_x,
            flip_y=flip_y,
        )

    def _load_csv_grid(
        self,
        source: GridSourceConfig,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        x_column = _required_mapping(
            source.x_column, source.file, "Grid xColumn")
        y_column = _required_mapping(
            source.y_column, source.file, "Grid yColumn")
        columns = [x_column, y_column]
        if source.height_column is not None:
            columns.append(source.height_column)
        if source.geoid_undulation_column is not None:
            columns.append(source.geoid_undulation_column)
        data = _read_csv(source.file, columns)

        x_values = _numeric_column(data, x_column, source.file)
        y_values = _numeric_column(data, y_column, source.file)
        x_axis, y_axis = _axes_from_rows(
            x_values, y_values, source.row_order, source.file)

        height = None
        if source.height_column is not None:
            height_values = _numeric_column(
                data, source.height_column, source.file)
            height = _reshape_rows(
                height_values, x_axis.size, y_axis.size, source.row_order)

        geoid_undulation = None
        if source.geoid_undulation_column is not None:
            undulation_values = _numeric_column(
                data, source.geoid_undulation_column, source.file)
            geoid_undulation = _reshape_rows(
                undulation_values,
                x_axis.size,
                y_axis.size,
                source.row_order,
            )

        return x_axis, y_axis, height, geoid_undulation

    def _load_mat_grid(
        self,
        source: GridSourceConfig,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        mat_data = self._load_mat(source.file)
        x_variable = _required_mapping(
            source.x_variable, source.file, "Grid xVariable")
        y_variable = _required_mapping(
            source.y_variable, source.file, "Grid yVariable")
        x_axis = _numeric_array(
            _resolve_mat_value(mat_data, x_variable, source.file),
            x_variable,
            source.file,
        ).squeeze()
        y_axis = _numeric_array(
            _resolve_mat_value(mat_data, y_variable, source.file),
            y_variable,
            source.file,
        ).squeeze()
        if x_axis.ndim != 1 or y_axis.ndim != 1:
            raise _data_error(
                source.file,
                "MAT grid xVariable and yVariable must both be 1-D arrays.")

        height = None
        if source.height_variable is not None:
            height = _numeric_array(
                _resolve_mat_value(
                    mat_data, source.height_variable, source.file),
                source.height_variable,
                source.file,
            )
            expected = (x_axis.size, y_axis.size)
            if height.shape != expected:
                raise _data_error(
                    source.file,
                    f"Height array has shape {height.shape}; expected "
                    f"{expected}.")

        geoid_undulation = None
        if source.geoid_undulation_variable is not None:
            geoid_undulation = _numeric_array(
                _resolve_mat_value(
                    mat_data,
                    source.geoid_undulation_variable,
                    source.file,
                ),
                source.geoid_undulation_variable,
                source.file,
            )
            expected = (x_axis.size, y_axis.size)
            if geoid_undulation.shape != expected:
                raise _data_error(
                    source.file,
                    f"Geoid-undulation array has shape "
                    f"{geoid_undulation.shape}; expected {expected}.",
                )
        return x_axis, y_axis, height, geoid_undulation

    def _load_vector(
        self,
        source: Optional[VectorSourceConfig],
        grid: LoadedGrid,
    ) -> np.ndarray:
        if source is None:
            raise ValueError("A configured vector source is required.")

        if source.format == "csv":
            component_columns = _required_mapping(
                source.component_columns, source.file,
                f"{source.section} component columns")
            requested_columns = list(component_columns)
            if source.coordinate_columns is not None:
                requested_columns.extend(source.coordinate_columns)
            data = _read_csv(source.file, requested_columns)
            _validate_source_coordinates(data, source, grid)
            values = np.column_stack([
                _numeric_column(data, column, source.file)
                for column in component_columns
            ])
            values = _reshape_rows(
                values, grid.x.size, grid.y.size, source.row_order)
        else:
            mat_data = self._load_mat(source.file)
            data_variable = _required_mapping(
                source.data_variable, source.file,
                f"{source.section} dataVariable")
            component_indices = _required_mapping(
                source.component_indices, source.file,
                f"{source.section} component indexes")
            packed = _numeric_array(
                _resolve_mat_value(
                    mat_data, data_variable, source.file),
                data_variable,
                source.file,
            )
            if packed.ndim != 3:
                raise _data_error(
                    source.file,
                    f"MAT vector data must be 3-D, not {packed.ndim}-D.")
            permutation = tuple(
                source.axis_order.index(axis)
                for axis in ("x", "y", "component")
            )
            packed = np.transpose(packed, permutation)
            max_index = max(component_indices)
            if packed.shape[2] <= max_index:
                raise _data_error(
                    source.file,
                    f"Component index {max_index} exceeds the packed "
                    f"component axis of size {packed.shape[2]}.")
            values = packed[:, :, component_indices]

        expected = (grid.x.size, grid.y.size, 3)
        if values.shape != expected:
            raise _data_error(
                source.file,
                f"Vector field has shape {values.shape}; expected {expected}.")
        if grid.flip_x:
            values = np.flip(values, axis=0)
        if grid.flip_y:
            values = np.flip(values, axis=1)

        values = np.asarray(values, dtype=np.float64)
        values *= _acceleration_scale(source.units)
        values = _to_ned(values, source, grid)
        return _to_effective_gravity(values, source.quantity, grid)

    def _validate_tensor(
        self,
        source: TensorSourceConfig,
        grid: LoadedGrid,
    ) -> None:
        if source.format == "csv":
            columns = _required_mapping(
                source.columns, source.file, "Tensor component columns")
            data = _read_csv(source.file, list(columns))
            values = np.column_stack([
                _numeric_column(data, column, source.file)
                for column in columns
            ])
            values = _reshape_rows(
                values, grid.x.size, grid.y.size, source.row_order)
        else:
            mat_data = self._load_mat(source.file)
            data_variable = _required_mapping(
                source.data_variable, source.file, "Tensor dataVariable")
            indices = _required_mapping(
                source.indices, source.file, "Tensor component indexes")
            packed = _numeric_array(
                _resolve_mat_value(
                    mat_data, data_variable, source.file),
                data_variable,
                source.file,
            )
            if packed.ndim != 3:
                raise _data_error(
                    source.file,
                    f"MAT tensor data must be 3-D, not {packed.ndim}-D.")
            permutation = tuple(
                source.axis_order.index(axis)
                for axis in ("x", "y", "component")
            )
            packed = np.transpose(packed, permutation)
            max_index = max(indices)
            if packed.shape[2] <= max_index:
                raise _data_error(
                    source.file,
                    f"Tensor index {max_index} exceeds the packed component "
                    f"axis of size {packed.shape[2]}.")
            values = packed[:, :, indices]

        expected = (grid.x.size, grid.y.size, 6)
        if values.shape != expected:
            raise _data_error(
                source.file,
                f"Tensor field has shape {values.shape}; expected {expected}.")
        if not np.all(np.isfinite(values)):
            raise _data_error(
                source.file, "Tensor field must contain only finite values.")
        if grid.flip_x:
            values = np.flip(values, axis=0)
        if grid.flip_y:
            values = np.flip(values, axis=1)

        values = np.asarray(values, dtype=np.float64)
        values *= _tensor_scale(source.units)
        matrices = np.empty(values.shape[:2] + (3, 3))
        matrices[:, :, 0, 0] = values[:, :, 0]
        matrices[:, :, 1, 1] = values[:, :, 1]
        matrices[:, :, 2, 2] = values[:, :, 2]
        matrices[:, :, 0, 1] = matrices[:, :, 1, 0] = values[:, :, 3]
        matrices[:, :, 0, 2] = matrices[:, :, 2, 0] = values[:, :, 4]
        matrices[:, :, 1, 2] = matrices[:, :, 2, 1] = values[:, :, 5]
        rotations = _source_to_ned_rotations(
            source.frame, source.custom_to_ned, grid, "Tensor")
        transformed = np.einsum(
            "xyik,xykl,xyjl->xyij",
            rotations,
            matrices,
            rotations,
        )
        if not np.all(np.isfinite(transformed)):
            raise _data_error(
                source.file,
                "Tensor frame conversion produced non-finite values.",
            )

    def _load_mat(self, file_path: Path) -> dict[str, Any]:
        if file_path not in self._mat_cache:
            if not file_path.is_file():
                raise FileNotFoundError(
                    f"Custom gravity data file does not exist: {file_path}")
            try:
                self._mat_cache[file_path] = loadmat(
                    file_path, squeeze_me=True, struct_as_record=False)
            except NotImplementedError as error:
                raise _data_error(
                    file_path,
                    "MATLAB v7.3/HDF5 files are not supported; use a "
                    "v5-v7.2 MAT file or CSV.",
                ) from error
            except Exception as error:
                raise _data_error(
                    file_path, f"Unable to read MATLAB data: {error}") from error
        return self._mat_cache[file_path]


def load_custom_map_data(config: CustomGravityConfig) -> LoadedCustomMap:
    """Convenience wrapper for loading a custom gravity map."""

    return CustomMapDataLoader(config).load()


def _read_csv(file_path: Path, columns: list[str]) -> pd.DataFrame:
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Custom gravity data file does not exist: {file_path}")
    try:
        return pd.read_csv(
            file_path, usecols=columns, skipinitialspace=True)
    except ValueError as error:
        raise _data_error(
            file_path,
            f"Required CSV columns could not be read: {error}") from error
    except Exception as error:
        raise _data_error(
            file_path, f"Unable to read CSV data: {error}") from error


def _numeric_column(
    data: pd.DataFrame,
    column: str,
    file_path: Path,
) -> np.ndarray:
    try:
        values = pd.to_numeric(data[column], errors="raise").to_numpy(
            dtype=np.float64)
    except Exception as error:
        raise _data_error(
            file_path,
            f"CSV column '{column}' contains nonnumeric values.") from error
    return values


def _numeric_array(value: Any, variable: str, file_path: Path) -> np.ndarray:
    try:
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"non-numeric dtype {array.dtype}")
        return array.astype(np.float64, copy=False)
    except Exception as error:
        raise _data_error(
            file_path,
            f"MAT variable '{variable}' is not a numeric array.") from error


def _required_mapping(
    value: Optional[_T],
    file_path: Path,
    name: str,
) -> _T:
    """Reject incomplete dataclass configurations with a useful data error."""

    if value is None:
        raise _data_error(file_path, f"{name} is required for this format.")
    return value


def _resolve_mat_value(
    mat_data: dict[str, Any],
    variable: str,
    file_path: Path,
) -> Any:
    current: Any = mat_data
    try:
        for part in variable.split("."):
            if isinstance(current, dict):
                current = current[part]
            else:
                current = getattr(current, part)
    except (KeyError, AttributeError) as error:
        raise _data_error(
            file_path,
            f"MAT variable '{variable}' does not exist.") from error
    return current


def _axes_from_rows(
    x_values: np.ndarray,
    y_values: np.ndarray,
    row_order: str,
    file_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if x_values.size != y_values.size or x_values.size == 0:
        raise _data_error(
            file_path, "Grid coordinate columns must have equal nonzero size.")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(y_values)):
        raise _data_error(file_path, "Grid coordinates must all be finite.")

    nx = np.unique(x_values).size
    ny = np.unique(y_values).size
    if nx * ny != x_values.size:
        raise _data_error(
            file_path,
            "Grid coordinates do not form a complete Cartesian product.")

    expected_x: np.ndarray
    expected_y: np.ndarray
    if row_order == "x_fastest":
        x_axis = x_values[:nx]
        y_axis = y_values[::nx]
        expected_x = np.tile(x_axis, ny)
        expected_y = np.repeat(y_axis, nx)
    else:
        x_axis = x_values[::ny]
        y_axis = y_values[:ny]
        expected_x = np.repeat(x_axis, ny)
        expected_y = np.tile(y_axis, nx)

    if (
        np.unique(x_axis).size != nx
        or np.unique(y_axis).size != ny
        or not np.allclose(x_values, expected_x, rtol=1e-12, atol=1e-9)
        or not np.allclose(y_values, expected_y, rtol=1e-12, atol=1e-9)
    ):
        raise _data_error(
            file_path,
            f"Grid coordinate rows do not match rowOrder={row_order}.")
    return x_axis, y_axis


def _normalize_axis(
    axis: np.ndarray,
    name: str,
) -> tuple[np.ndarray, bool]:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(
            f"Custom gravity {name} axis must contain at least two finite "
            f"values.")
    differences = np.diff(axis)
    if np.all(differences > 0):
        return axis, False
    if np.all(differences < 0):
        return axis[::-1], True
    raise ValueError(
        f"Custom gravity {name} axis must be strictly monotonic.")


def _reshape_rows(
    values: np.ndarray,
    nx: int,
    ny: int,
    row_order: str,
) -> np.ndarray:
    if values.shape[0] != nx * ny:
        raise ValueError(
            f"Custom gravity data has {values.shape[0]} rows; expected "
            f"{nx * ny}.")
    trailing_shape = values.shape[1:]
    if row_order == "x_fastest":
        reshaped = values.reshape((ny, nx, *trailing_shape))
        axes = (1, 0, *range(2, reshaped.ndim))
        return np.transpose(reshaped, axes)
    return values.reshape((nx, ny, *trailing_shape))


def _acceleration_scale(units: str) -> float:
    return {
        "m/s2": 1.0,
        "gal": 1e-2,
        "mgal": 1e-5,
    }[units]


def _tensor_scale(units: str) -> float:
    return {
        "e": 1e-9,
        "s-2": 1.0,
    }[units]


def _validate_source_coordinates(
    data: pd.DataFrame,
    source: VectorSourceConfig,
    grid: LoadedGrid,
) -> None:
    if source.coordinate_frame == "none":
        return
    columns = _required_mapping(
        source.coordinate_columns,
        source.file,
        f"{source.section} coordinate columns",
    )
    coordinates = np.column_stack([
        _numeric_column(data, column, source.file)
        for column in columns
    ])
    coordinates = _reshape_rows(
        coordinates, grid.x.size, grid.y.size, source.row_order)
    if grid.flip_x:
        coordinates = np.flip(coordinates, axis=0)
    if grid.flip_y:
        coordinates = np.flip(coordinates, axis=1)

    if source.coordinate_frame == "grid":
        expected_x: np.ndarray
        expected_y: np.ndarray
        expected_x, expected_y = np.meshgrid(
            grid.x, grid.y, indexing="ij")
        expected = np.stack((expected_x, expected_y), axis=-1)
        valid = np.allclose(
            coordinates, expected, rtol=1e-12, atol=1e-6)
    else:
        latitude_error = np.abs(coordinates[:, :, 0] - grid.latitude)
        longitude_error = np.abs(
            (coordinates[:, :, 1] - grid.longitude + 180.0) % 360.0
            - 180.0
        )
        valid = bool(
            np.all(latitude_error <= 1e-8)
            and np.all(longitude_error <= 1e-8)
        )
    if not valid:
        raise _data_error(
            source.file,
            f"{source.section} coordinates do not align with the grid.",
        )


def _to_ned(
    values: np.ndarray,
    source: VectorSourceConfig,
    grid: LoadedGrid,
) -> np.ndarray:
    rotations = _source_to_ned_rotations(
        source.frame,
        source.custom_to_ned,
        grid,
        source.section,
    )
    return np.einsum("xyij,xyj->xyi", rotations, values)


def _source_to_ned_rotations(
    frame: str,
    custom_to_ned: Optional[tuple[float, ...]],
    grid: LoadedGrid,
    section: str,
) -> np.ndarray:
    shape = grid.latitude.shape + (3, 3)
    if frame == "ned":
        return np.broadcast_to(np.eye(3), shape)
    if frame == "enu":
        return np.broadcast_to(
            np.array([
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ]),
            shape,
        )
    if frame == "ecef":
        lla = np.column_stack((
            grid.latitude.ravel(),
            grid.longitude.ravel(),
            grid.ellipsoid_height.ravel(),
        ))
        return ecef2ned_transform_vec(lla).reshape(shape)
    if frame == "geocentric_ned":
        delta = np.radians(
            grid.latitude - grid.geocentric_latitude)
        cosine = np.cos(delta)
        sine = np.sin(delta)
        rotations = np.zeros(shape)
        rotations[:, :, 0, 0] = cosine
        rotations[:, :, 0, 2] = sine
        rotations[:, :, 1, 1] = 1.0
        rotations[:, :, 2, 0] = -sine
        rotations[:, :, 2, 2] = cosine
        return rotations

    matrix = np.asarray(custom_to_ned).reshape(3, 3)
    if not np.allclose(
        matrix @ matrix.T, np.eye(3), rtol=0, atol=1e-8
    ) or not np.isclose(np.linalg.det(matrix), 1.0, rtol=0, atol=1e-8):
        raise NavConfigError(
            section,
            "customToNed",
            "Invalid rotation",
            "The custom-to-NED matrix must be orthonormal with "
            "determinant +1.",
        )
    return np.broadcast_to(matrix, shape)


def _to_effective_gravity(
    values: np.ndarray,
    quantity: str,
    grid: LoadedGrid,
) -> np.ndarray:
    if quantity != "gravitational_attraction":
        return values

    ecef = grid.ecef.reshape(-1, 3)
    centrifugal_ecef = np.column_stack((
        constants.OMEGA_E ** 2 * ecef[:, 0],
        constants.OMEGA_E ** 2 * ecef[:, 1],
        np.zeros(ecef.shape[0]),
    ))
    lla = np.column_stack((
        grid.latitude.ravel(),
        grid.longitude.ravel(),
        grid.ellipsoid_height.ravel(),
    ))
    transforms = ecef2ned_transform_vec(lla)
    centrifugal_ned = np.einsum(
        "nij,nj->ni", transforms, centrifugal_ecef)
    return values + centrifugal_ned.reshape(values.shape)


def _data_error(file_path: Path, description: str) -> ValueError:
    return ValueError(
        f"Invalid custom gravity data '{file_path}': {description}")
