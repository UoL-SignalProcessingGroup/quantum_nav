"""Data loading and normalization for configurable gravity maps."""

from importlib import import_module
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, cast

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError
from scipy.io import loadmat

from qnav.input.config_handler import NavConfigError
from qnav.input.ini.custom_gravity_config import (
    CustomGravityConfig,
    DelimitedSourceConfig,
    GridSourceConfig,
    SubsetConfig,
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
    source_x_size: int
    source_y_size: int
    x_indices: np.ndarray
    y_indices: np.ndarray


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
    quantity: str
    crs: CRS


class CustomMapDataLoader:
    """Load all configured map resources with a scoped MATLAB file cache."""

    def __init__(self, config: CustomGravityConfig):
        self._config = config
        self._mat_cache: dict[Path, dict[str, Any]] = {}
        self._delimited_cache: dict[
            tuple[Path, DelimitedSourceConfig], pd.DataFrame
        ] = {}
        self._grid_source_axes: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._grid_auxiliary_preselected = False

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

        quantity = (
            self._config.field.quantity
            if self._config.field.representation == "scalar"
            else "residual"
        )

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
            quantity=quantity,
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
        elif source.format == "delimited":
            x, y, height, geoid_undulation = self._load_delimited_grid(
                source, crs)
        elif source.format == "geotiff":
            x, y, height, geoid_undulation = self._load_geotiff_grid(
                source, crs)
        elif source.format == "netcdf":
            x, y, height, geoid_undulation = self._load_netcdf_grid(
                source, crs)
        else:
            x, y, height, geoid_undulation = self._load_mat_grid(source)

        source_x_size = int(x.size)
        source_y_size = int(y.size)
        self._grid_source_axes = (
            np.asarray(x, dtype=np.float64).copy(),
            np.asarray(y, dtype=np.float64).copy(),
        )
        x_indices, y_indices = _subset_indices(
            x, y, self._config.subset, crs)
        x = x[x_indices]
        y = y[y_indices]
        if height is not None and not self._grid_auxiliary_preselected:
            height = height[np.ix_(x_indices, y_indices)]
        if (
            geoid_undulation is not None
            and not self._grid_auxiliary_preselected
        ):
            geoid_undulation = geoid_undulation[np.ix_(
                x_indices, y_indices)]
        cell_count = int(x.size * y.size)
        if cell_count > self._config.max_cells:
            raise _data_error(
                source.file,
                f"Selected grid contains {cell_count:,} cells; maxCells is "
                f"{self._config.max_cells:,}. Configure a smaller [Subset] "
                "or explicitly raise maxCells.",
            )

        x, flip_x = _normalize_axis(x, "x")
        y, flip_y = _normalize_axis(y, "y")
        if crs.is_geographic:
            period = geographic_longitude_period(crs)
            span = float(x[-1] - x[0])
            scale = max(1.0, abs(span), period)
            tolerance = np.finfo(np.float64).eps * scale * 64
            if span >= period - tolerance:
                raise _data_error(
                    source.file,
                    "Geographic longitude axes must span less than one "
                    "revolution so equivalent seam coordinates are not "
                    "duplicated.",
                )
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
            source_x_size=source_x_size,
            source_y_size=source_y_size,
            x_indices=x_indices,
            y_indices=y_indices,
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

    def _load_delimited_grid(
        self,
        source: GridSourceConfig,
        crs: CRS,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        options = _required_mapping(
            source.delimited, source.file, "Grid delimited options")
        x_mapping: str | int
        y_mapping: str | int
        height_mapping: str | int | None
        geoid_mapping: str | int | None
        if options.header == "none":
            x_mapping = _required_mapping(
                source.x_index, source.file, "Grid xIndex")
            y_mapping = _required_mapping(
                source.y_index, source.file, "Grid yIndex")
            height_mapping = source.height_index
            geoid_mapping = source.geoid_undulation_index
        else:
            x_mapping = _required_mapping(
                source.x_column, source.file, "Grid xColumn")
            y_mapping = _required_mapping(
                source.y_column, source.file, "Grid yColumn")
            height_mapping = source.height_column
            geoid_mapping = source.geoid_undulation_column

        if self._can_stream_subset_delimited(source.file):
            data = self._load_delimited_subset(
                source.file,
                options,
                x_mapping,
                y_mapping,
                crs,
            )
        else:
            data = self._load_delimited(source.file, options)

        x_values = _numeric_column(data, x_mapping, source.file)
        y_values = _numeric_column(data, y_mapping, source.file)
        x_axis, y_axis = _axes_from_rows(
            x_values, y_values, source.row_order, source.file)

        height = None
        if height_mapping is not None:
            height = _reshape_rows(
                _numeric_column(data, height_mapping, source.file),
                x_axis.size,
                y_axis.size,
                source.row_order,
            )
        geoid_undulation = None
        if geoid_mapping is not None:
            geoid_undulation = _reshape_rows(
                _numeric_column(data, geoid_mapping, source.file),
                x_axis.size,
                y_axis.size,
                source.row_order,
            )
        return x_axis, y_axis, height, geoid_undulation

    def _load_geotiff_grid(
        self,
        source: GridSourceConfig,
        crs: CRS,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        rasterio = _optional_module("rasterio", "GeoTIFF")
        if not source.file.is_file():
            raise FileNotFoundError(
                f"Custom gravity data file does not exist: {source.file}")
        try:
            with rasterio.open(source.file) as dataset:
                _validate_embedded_crs(dataset.crs, crs, source.file)
                x, y = _geotiff_axes(dataset, source.file)
                x_indices, y_indices = _subset_indices(
                    x, y, self._config.subset, crs)
                window = rasterio.windows.Window(
                    col_off=int(x_indices[0]),
                    row_off=int(y_indices[0]),
                    width=int(x_indices.size),
                    height=int(y_indices.size),
                )
                height = _read_geotiff_bands(
                    dataset,
                    (source.height_band,),
                    source.file,
                    window=window,
                )[:, :, 0] if source.height_band is not None else None
                geoid = _read_geotiff_bands(
                    dataset,
                    (source.geoid_undulation_band,),
                    source.file,
                    window=window,
                )[:, :, 0] if source.geoid_undulation_band is not None else None
                self._grid_auxiliary_preselected = True
        except (FileNotFoundError, ValueError, NavConfigError):
            raise
        except Exception as error:
            raise _data_error(
                source.file, f"Unable to read GeoTIFF data: {error}") from error
        return x, y, height, geoid

    def _load_netcdf_grid(
        self,
        source: GridSourceConfig,
        crs: CRS,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        xarray = _optional_module("xarray", "NetCDF/GRD")
        if not source.file.is_file():
            raise FileNotFoundError(
                f"Custom gravity data file does not exist: {source.file}")
        x_name = _required_mapping(
            source.x_variable, source.file, "Grid xVariable")
        y_name = _required_mapping(
            source.y_variable, source.file, "Grid yVariable")
        try:
            with xarray.open_dataset(source.file, mask_and_scale=True) as data:
                _validate_netcdf_crs(data, crs, source.file)
                x = _netcdf_axis(data, x_name, source.file)
                y = _netcdf_axis(data, y_name, source.file)
                x_indices, y_indices = _subset_indices(
                    x, y, self._config.subset, crs)
                x_dimension = data[x_name].dims[0]
                y_dimension = data[y_name].dims[0]
                height = (
                    _netcdf_array(
                        data,
                        source.height_variable,
                        x_dimension,
                        y_dimension,
                        source.file,
                        x_indices=x_indices,
                        y_indices=y_indices,
                    )
                    if source.height_variable is not None
                    else None
                )
                geoid = (
                    _netcdf_array(
                        data,
                        source.geoid_undulation_variable,
                        x_dimension,
                        y_dimension,
                        source.file,
                        x_indices=x_indices,
                        y_indices=y_indices,
                    )
                    if source.geoid_undulation_variable is not None
                    else None
                )
                self._grid_auxiliary_preselected = True
        except (FileNotFoundError, ValueError, NavConfigError):
            raise
        except Exception as error:
            raise _data_error(
                source.file, f"Unable to read NetCDF data: {error}") from error
        return x, y, height, geoid

    def _load_geotiff_field(
        self,
        source: VectorSourceConfig,
        grid: LoadedGrid,
    ) -> np.ndarray:
        bands = _required_mapping(
            source.component_bands,
            source.file,
            f"{source.section} component bands",
        )
        return self._read_aligned_geotiff(source.file, bands, grid)

    def _load_geotiff_scalar(
        self,
        source: VectorSourceConfig,
        grid: LoadedGrid,
    ) -> np.ndarray:
        band = _required_mapping(
            source.value_band, source.file, f"{source.section} valueBand")
        return self._read_aligned_geotiff(
            source.file, (band,), grid)[:, :, 0]

    def _read_aligned_geotiff(
        self,
        file_path: Path,
        bands: tuple[int, ...],
        grid: LoadedGrid,
    ) -> np.ndarray:
        rasterio = _optional_module("rasterio", "GeoTIFF")
        expected_axes = _required_mapping(
            self._grid_source_axes, file_path, "Grid source axes")
        try:
            with rasterio.open(file_path) as dataset:
                _validate_embedded_crs(
                    dataset.crs, self._data_crs(), file_path)
                x, y = _geotiff_axes(dataset, file_path)
                _validate_aligned_axes(x, y, expected_axes, file_path)
                window = rasterio.windows.Window(
                    col_off=int(grid.x_indices[0]),
                    row_off=int(grid.y_indices[0]),
                    width=int(grid.x_indices.size),
                    height=int(grid.y_indices.size),
                )
                return _read_geotiff_bands(
                    dataset, bands, file_path, window=window)
        except (FileNotFoundError, ValueError, NavConfigError):
            raise
        except Exception as error:
            raise _data_error(
                file_path, f"Unable to read GeoTIFF data: {error}") from error

    def _load_netcdf_field(
        self,
        source: VectorSourceConfig,
        grid: LoadedGrid,
    ) -> np.ndarray:
        variables = _required_mapping(
            source.component_variables,
            source.file,
            f"{source.section} component variables",
        )
        arrays = [
            self._read_aligned_netcdf(source.file, variable, grid)
            for variable in variables
        ]
        return np.stack(arrays, axis=-1)

    def _load_netcdf_scalar(
        self,
        source: VectorSourceConfig,
        grid: LoadedGrid,
    ) -> np.ndarray:
        variable = _required_mapping(
            source.value_variable,
            source.file,
            f"{source.section} valueVariable",
        )
        return self._read_aligned_netcdf(source.file, variable, grid)

    def _read_aligned_netcdf(
        self,
        file_path: Path,
        variable: str,
        grid: LoadedGrid,
    ) -> np.ndarray:
        xarray = _optional_module("xarray", "NetCDF/GRD")
        grid_source = self._config.grid
        x_name = _required_mapping(
            grid_source.x_variable, file_path, "Grid xVariable")
        y_name = _required_mapping(
            grid_source.y_variable, file_path, "Grid yVariable")
        expected_axes = _required_mapping(
            self._grid_source_axes, file_path, "Grid source axes")
        try:
            with xarray.open_dataset(file_path, mask_and_scale=True) as data:
                _validate_netcdf_crs(data, self._data_crs(), file_path)
                x = _netcdf_axis(data, x_name, file_path)
                y = _netcdf_axis(data, y_name, file_path)
                _validate_aligned_axes(x, y, expected_axes, file_path)
                values = _netcdf_array(
                    data,
                    variable,
                    data[x_name].dims[0],
                    data[y_name].dims[0],
                    file_path,
                    x_indices=grid.x_indices,
                    y_indices=grid.y_indices,
                )
                return values
        except (FileNotFoundError, ValueError, NavConfigError):
            raise
        except Exception as error:
            raise _data_error(
                file_path, f"Unable to read NetCDF data: {error}") from error

    def _data_crs(self) -> CRS:
        """Return the already-validated configured CRS."""
        return self._read_crs()

    def _load_vector(
        self,
        source: Optional[VectorSourceConfig],
        grid: LoadedGrid,
    ) -> np.ndarray:
        if source is None:
            raise ValueError("A configured vector source is required.")

        if source.representation == "scalar":
            return self._load_scalar(source, grid)

        if source.format in {"csv", "delimited"}:
            if source.format == "csv":
                data = _read_csv(
                    source.file,
                    list(_required_mapping(
                        source.component_columns,
                        source.file,
                        f"{source.section} component columns",
                    )) + list(source.coordinate_columns or ()),
                )
                component_mappings: tuple[Any, Any, Any] = _required_mapping(
                    source.component_columns,
                    source.file,
                    f"{source.section} component columns",
                )
            else:
                options = _required_mapping(
                    source.delimited,
                    source.file,
                    f"{source.section} delimited options",
                )
                data = self._load_delimited(source.file, options)
                component_mappings = _required_mapping(
                    source.component_indices
                    if options.header == "none"
                    else source.component_columns,
                    source.file,
                    f"{source.section} component mappings",
                )
            _validate_source_coordinates(data, source, grid)
            values = np.column_stack([
                _numeric_column(data, mapping, source.file)
                for mapping in component_mappings
            ])
            values = _reshape_rows(
                values,
                grid.source_x_size,
                grid.source_y_size,
                source.row_order,
            )
            values = values[np.ix_(grid.x_indices, grid.y_indices)]
        elif source.format == "geotiff":
            values = self._load_geotiff_field(source, grid)
        elif source.format == "netcdf":
            values = self._load_netcdf_field(source, grid)
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
            if values.shape[:2] != (
                grid.source_x_size, grid.source_y_size
            ):
                raise _data_error(
                    source.file,
                    f"Vector field has source shape {values.shape[:2]}; "
                    f"expected {(grid.source_x_size, grid.source_y_size)}.",
                )
            values = values[np.ix_(grid.x_indices, grid.y_indices)]

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
        if np.any(np.isinf(values)):
            raise _data_error(
                source.file, "Vector field must not contain infinite values.")
        missing = np.any(np.isnan(values), axis=2)
        values[missing, :] = np.nan
        values *= _acceleration_scale(source.units)
        values = _to_ned(values, source, grid)
        return _to_effective_gravity(values, source.quantity, grid)

    def _load_scalar(
        self,
        source: VectorSourceConfig,
        grid: LoadedGrid,
    ) -> np.ndarray:
        """Load a scalar vertical disturbance or free-air anomaly."""

        if source.format in {"csv", "delimited"}:
            if source.format == "csv":
                value_mapping: Any = _required_mapping(
                    source.value_column,
                    source.file,
                    f"{source.section} valueColumn",
                )
                requested_columns = [value_mapping]
                if source.coordinate_columns is not None:
                    requested_columns.extend(source.coordinate_columns)
                data = _read_csv(source.file, requested_columns)
            else:
                options = _required_mapping(
                    source.delimited,
                    source.file,
                    f"{source.section} delimited options",
                )
                data = self._load_delimited(source.file, options)
                value_mapping = _required_mapping(
                    source.value_index
                    if options.header == "none"
                    else source.value_column,
                    source.file,
                    f"{source.section} value mapping",
                )
            _validate_source_coordinates(data, source, grid)
            scalar = _numeric_column(data, value_mapping, source.file)
            scalar = _reshape_rows(
                scalar,
                grid.source_x_size,
                grid.source_y_size,
                source.row_order,
            )
            scalar = scalar[np.ix_(grid.x_indices, grid.y_indices)]
        elif source.format == "geotiff":
            scalar = self._load_geotiff_scalar(source, grid)
        elif source.format == "netcdf":
            scalar = self._load_netcdf_scalar(source, grid)
        else:
            value_variable = _required_mapping(
                source.value_variable,
                source.file,
                f"{source.section} valueVariable",
            )
            scalar = _numeric_array(
                _resolve_mat_value(
                    self._load_mat(source.file),
                    value_variable,
                    source.file,
                ),
                value_variable,
                source.file,
            )
            if scalar.ndim != 2:
                raise _data_error(
                    source.file,
                    f"MAT scalar data must be 2-D, not {scalar.ndim}-D.",
                )
            permutation = tuple(
                source.axis_order.index(axis) for axis in ("x", "y"))
            scalar = np.transpose(scalar, permutation)
            if scalar.shape != (grid.source_x_size, grid.source_y_size):
                raise _data_error(
                    source.file,
                    f"Scalar field has source shape {scalar.shape}; expected "
                    f"{(grid.source_x_size, grid.source_y_size)}.",
                )
            scalar = scalar[np.ix_(grid.x_indices, grid.y_indices)]

        expected = (grid.x.size, grid.y.size)
        if scalar.shape != expected:
            raise _data_error(
                source.file,
                f"Scalar field has shape {scalar.shape}; expected {expected}.",
            )
        if grid.flip_x:
            scalar = np.flip(scalar, axis=0)
        if grid.flip_y:
            scalar = np.flip(scalar, axis=1)

        scalar = np.asarray(scalar, dtype=np.float64)
        if np.any(np.isinf(scalar)):
            raise _data_error(
                source.file, "Scalar field must not contain infinite values.")
        scalar *= _acceleration_scale(source.units)
        if source.vertical_direction == "up":
            scalar *= -1.0

        values = np.zeros(expected + (3,), dtype=np.float64)
        values[:, :, 2] = scalar
        missing = np.isnan(scalar)
        values[missing, :] = np.nan
        return values

    def _load_delimited(
        self,
        file_path: Path,
        options: DelimitedSourceConfig,
    ) -> pd.DataFrame:
        key = (file_path, options)
        if key in self._delimited_cache:
            return self._delimited_cache[key]
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Custom gravity data file does not exist: {file_path}")
        try:
            data = pd.read_csv(file_path, **_delimited_read_kwargs(options))
        except Exception as error:
            raise _data_error(
                file_path, f"Unable to read delimited data: {error}") from error
        if data.empty:
            raise _data_error(file_path, "Delimited data file is empty.")
        self._delimited_cache[key] = data
        return data

    def _can_stream_subset_delimited(self, grid_file: Path) -> bool:
        if self._config.subset is None:
            return False
        sources = [self._config.field]
        if self._config.reference is not None:
            sources.append(self._config.reference)
        return all(
            source.format == "delimited" and source.file == grid_file
            for source in sources
        )

    def _load_delimited_subset(
        self,
        file_path: Path,
        options: DelimitedSourceConfig,
        x_mapping: str | int,
        y_mapping: str | int,
        crs: CRS,
    ) -> pd.DataFrame:
        key = (file_path, options)
        if key in self._delimited_cache:
            return self._delimited_cache[key]
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Custom gravity data file does not exist: {file_path}")
        kwargs = _delimited_read_kwargs(options)
        try:
            x_seen: list[np.ndarray] = []
            y_seen: list[np.ndarray] = []
            for chunk in pd.read_csv(
                file_path,
                usecols=[x_mapping, y_mapping],
                chunksize=100_000,
                **kwargs,
            ):
                x_seen.append(chunk[x_mapping].to_numpy())
                y_seen.append(chunk[y_mapping].to_numpy())
            if not x_seen:
                raise _data_error(file_path, "Delimited data file is empty.")
            x_axis = pd.unique(np.concatenate(x_seen)).astype(np.float64)
            y_axis = pd.unique(np.concatenate(y_seen)).astype(np.float64)
            x_indices, y_indices = _subset_indices(
                x_axis, y_axis, self._config.subset, crs)
            selected_x = x_axis[x_indices]
            selected_y = y_axis[y_indices]
            selected: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                file_path, chunksize=100_000, **kwargs
            ):
                keep = (
                    chunk[x_mapping].isin(selected_x)
                    & chunk[y_mapping].isin(selected_y)
                )
                if bool(keep.any()):
                    selected.append(chunk.loc[keep])
            if not selected:
                raise _data_error(
                    file_path, "Configured subset contains no text rows.")
            data = pd.concat(selected, ignore_index=True)
        except (ValueError, NavConfigError):
            raise
        except Exception as error:
            raise _data_error(
                file_path,
                f"Unable to subset delimited data: {error}",
            ) from error
        self._delimited_cache[key] = data
        return data

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
                values,
                grid.source_x_size,
                grid.source_y_size,
                source.row_order,
            )
            values = values[np.ix_(grid.x_indices, grid.y_indices)]
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
            if values.shape[:2] != (
                grid.source_x_size, grid.source_y_size
            ):
                raise _data_error(
                    source.file,
                    f"Tensor field has source shape {values.shape[:2]}; "
                    f"expected {(grid.source_x_size, grid.source_y_size)}.",
                )
            values = values[np.ix_(grid.x_indices, grid.y_indices)]

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


def geographic_longitude_period(crs: CRS) -> float:
    """Return one revolution in the geographic CRS longitude unit."""

    for axis in crs.axis_info:
        if axis.direction.casefold() in {"east", "west"}:
            return float(2.0 * np.pi / axis.unit_conversion_factor)
    raise ValueError("Geographic CRS has no longitude axis.")


def _optional_module(name: str, description: str) -> Any:
    try:
        return import_module(name)
    except ImportError as error:
        raise NavConfigError(
            "CustomGravityMap",
            "format",
            "Missing optional dependency",
            f"{description} custom maps require the optional map readers. "
            "Install QNav with 'qnav[maps]'.",
        ) from error


def _validate_embedded_crs(
    embedded: Any,
    configured: CRS,
    file_path: Path,
) -> None:
    if embedded is None:
        return
    try:
        embedded_crs = CRS.from_user_input(embedded)
    except CRSError as error:
        raise _data_error(
            file_path, f"Embedded CRS metadata is invalid: {error}") from error
    if not configured.equals(embedded_crs, ignore_axis_order=True):
        raise _data_error(
            file_path,
            "Embedded CRS does not match the configured CRS: "
            f"embedded='{embedded_crs.to_string()}', "
            f"configured='{configured.to_string()}'.",
        )


def _geotiff_axes(dataset: Any, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    transform = dataset.transform
    if bool(transform.is_identity):
        raise _data_error(
            file_path, "GeoTIFF must contain an explicit affine transform.")
    scale = max(
        1.0,
        abs(float(transform.a)),
        abs(float(transform.e)),
    )
    tolerance = np.finfo(np.float64).eps * scale * 64
    if (
        abs(float(transform.b)) > tolerance
        or abs(float(transform.d)) > tolerance
        or abs(float(transform.a)) <= tolerance
        or abs(float(transform.e)) <= tolerance
    ):
        raise _data_error(
            file_path,
            "GeoTIFF transform must describe a non-rotated rectilinear grid.",
        )
    x = float(transform.c) + float(transform.a) * (
        np.arange(dataset.width, dtype=np.float64) + 0.5)
    y = float(transform.f) + float(transform.e) * (
        np.arange(dataset.height, dtype=np.float64) + 0.5)
    return x, y


def _read_geotiff_bands(
    dataset: Any,
    bands: tuple[int, ...],
    file_path: Path,
    window: Any = None,
) -> np.ndarray:
    if not bands or min(bands) < 1 or max(bands) > dataset.count:
        raise _data_error(
            file_path,
            f"Requested GeoTIFF bands {bands} exceed available bands "
            f"1..{dataset.count}.",
        )
    masked = dataset.read(list(bands), window=window, masked=True)
    values = np.ma.filled(masked, np.nan).astype(np.float64, copy=False)
    for output_index, band in enumerate(bands):
        scale = float(dataset.scales[band - 1])
        offset = float(dataset.offsets[band - 1])
        if not np.isfinite(scale) or not np.isfinite(offset):
            raise _data_error(
                file_path,
                f"GeoTIFF band {band} has invalid scale/offset metadata.",
            )
        values[output_index] *= scale
        values[output_index] += offset
    return np.transpose(values, (2, 1, 0))


def _netcdf_axis(data: Any, variable: str, file_path: Path) -> np.ndarray:
    if variable not in data:
        raise _data_error(
            file_path, f"NetCDF variable '{variable}' does not exist.")
    values = data[variable]
    if values.ndim != 1 or len(values.dims) != 1:
        raise _data_error(
            file_path, f"NetCDF axis '{variable}' must be one-dimensional.")
    return np.asarray(values.to_numpy(), dtype=np.float64)


def _netcdf_array(
    data: Any,
    variable: Optional[str],
    x_dimension: str,
    y_dimension: str,
    file_path: Path,
    x_indices: Optional[np.ndarray] = None,
    y_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    name = _required_mapping(variable, file_path, "NetCDF data variable")
    if name not in data:
        raise _data_error(
            file_path, f"NetCDF variable '{name}' does not exist.")
    values = data[name]
    if x_dimension not in values.dims or y_dimension not in values.dims:
        raise _data_error(
            file_path,
            f"NetCDF variable '{name}' must use dimensions "
            f"'{x_dimension}' and '{y_dimension}'.",
        )
    for dimension in tuple(values.dims):
        if dimension in {x_dimension, y_dimension}:
            continue
        if int(values.sizes[dimension]) != 1:
            raise _data_error(
                file_path,
                f"NetCDF variable '{name}' has unsupported non-singleton "
                f"dimension '{dimension}'.",
            )
        values = values.isel({dimension: 0}, drop=True)
    if x_indices is not None:
        values = values.isel({x_dimension: x_indices})
    if y_indices is not None:
        values = values.isel({y_dimension: y_indices})
    values = values.transpose(x_dimension, y_dimension)
    result = np.asarray(values.to_numpy(), dtype=np.float64)
    if np.any(np.isinf(result)):
        raise _data_error(
            file_path, f"NetCDF variable '{name}' contains infinite values.")
    return result


def _validate_netcdf_crs(data: Any, configured: CRS, file_path: Path) -> None:
    candidates: list[Any] = []
    for key in ("crs_wkt", "spatial_ref", "epsg_code"):
        if key in data.attrs:
            candidates.append(data.attrs[key])
    for variable in data.variables.values():
        for key in ("crs_wkt", "spatial_ref", "epsg_code"):
            if key in variable.attrs:
                candidates.append(variable.attrs[key])
    for candidate in candidates:
        try:
            embedded = CRS.from_user_input(candidate)
        except (CRSError, ValueError, TypeError):
            continue
        _validate_embedded_crs(embedded, configured, file_path)
        return


def _validate_aligned_axes(
    x: np.ndarray,
    y: np.ndarray,
    expected: tuple[np.ndarray, np.ndarray],
    file_path: Path,
) -> None:
    expected_x, expected_y = expected
    if (
        x.shape != expected_x.shape
        or y.shape != expected_y.shape
        or not np.allclose(x, expected_x, rtol=1e-12, atol=1e-9)
        or not np.allclose(y, expected_y, rtol=1e-12, atol=1e-9)
    ):
        raise _data_error(
            file_path, "Field grid does not align with the configured grid.")


def _subset_indices(
    x: np.ndarray,
    y: np.ndarray,
    subset: Optional[SubsetConfig],
    crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    if subset is None:
        return np.arange(x.size), np.arange(y.size)
    minimum_x = subset.minimum_x
    maximum_x = subset.maximum_x
    minimum_y = subset.minimum_y
    maximum_y = subset.maximum_y
    if subset.coordinate_frame == "wgs84":
        try:
            transformer = Transformer.from_crs(
                "EPSG:4326", crs, always_xy=True)
            minimum_x, minimum_y, maximum_x, maximum_y = (
                transformer.transform_bounds(
                    minimum_x,
                    minimum_y,
                    maximum_x,
                    maximum_y,
                    densify_pts=21,
                )
            )
        except Exception as error:
            raise NavConfigError(
                "Subset",
                "bounds",
                "Subset transformation failed",
                f"WGS-84 subset bounds could not be transformed into the "
                f"map CRS: {error}",
            ) from error
        if crs.is_geographic:
            period = geographic_longitude_period(crs)
            midpoint = (float(np.min(x)) + float(np.max(x))) / 2.0
            centre = (minimum_x + maximum_x) / 2.0
            shift = np.round((midpoint - centre) / period) * period
            minimum_x += shift
            maximum_x += shift
    x_indices = np.flatnonzero((x >= minimum_x) & (x <= maximum_x))
    y_indices = np.flatnonzero((y >= minimum_y) & (y <= maximum_y))
    if x_indices.size == 0 or y_indices.size == 0:
        raise ValueError("Configured custom gravity subset does not overlap the grid.")
    x_indices = _pad_indices(x_indices, x.size, subset.padding_cells)
    y_indices = _pad_indices(y_indices, y.size, subset.padding_cells)
    if x_indices.size < 2 or y_indices.size < 2:
        raise ValueError(
            "Configured custom gravity subset must retain at least two x "
            "and two y coordinates.")
    return x_indices, y_indices


def _pad_indices(indices: np.ndarray, size: int, padding: int) -> np.ndarray:
    start = max(0, int(indices[0]) - padding)
    stop = min(size, int(indices[-1]) + padding + 1)
    return np.arange(start, stop)


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


def _delimited_read_kwargs(options: DelimitedSourceConfig) -> dict[str, Any]:
    separator = (
        r"\s+" if options.delimiter == "whitespace" else options.delimiter)
    return {
        "sep": separator,
        "header": 0 if options.header == "present" else None,
        "skiprows": options.skip_rows,
        "comment": options.comment_prefix,
        "skipinitialspace": True,
    }


def _numeric_column(
    data: pd.DataFrame,
    column: str | int,
    file_path: Path,
) -> np.ndarray:
    try:
        values = pd.to_numeric(data[column], errors="raise").to_numpy(
            dtype=np.float64)
    except Exception as error:
        raise _data_error(
            file_path,
            f"Data column '{column}' contains nonnumeric values.") from error
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
    mappings = cast(
        tuple[str | int, str | int],
        _required_mapping(
            source.coordinate_indices
            if source.format == "delimited"
            and source.delimited is not None
            and source.delimited.header == "none"
            else source.coordinate_columns,
            source.file,
            f"{source.section} coordinate mappings",
        ),
    )
    coordinates = np.column_stack([
        _numeric_column(data, mapping, source.file)
        for mapping in mappings
    ])
    coordinates = _reshape_rows(
        coordinates,
        grid.source_x_size,
        grid.source_y_size,
        source.row_order,
    )
    coordinates = coordinates[np.ix_(grid.x_indices, grid.y_indices)]
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
