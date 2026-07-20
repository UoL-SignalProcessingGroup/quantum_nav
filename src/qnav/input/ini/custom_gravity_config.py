"""
Configuration parsing for user-supplied gravity correction maps.

Custom gravity maps deliberately use a separate INI file.  This keeps file
layout, coordinate reference system, units, and vector-frame information
alongside the data source rather than hard-coding them in QNav.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from qnav.input.config_handler import ConfigHandler, NavConfigError


_MAP_SECTION = "CustomGravityMap"
_GRID_SECTION = "Grid"
_FIELD_SECTION = "Field"
_REFERENCE_SECTION = "Reference"
_TENSOR_SECTION = "Tensor"

_FORMATS = {"csv", "mat"}
_ROW_ORDERS = {"x_fastest", "y_fastest"}
_UNITS = {"m/s2", "gal", "mgal"}
_FRAMES = {"ned", "enu", "ecef", "custom"}
_MODES = {"residual", "total_minus_reference"}
_INTERPOLATION = {"linear", "nearest"}
_COVERAGE = {"base", "error"}
_HEIGHT_REFERENCES = {"ellipsoid", "orthometric", "geoid_surface"}
_TENSOR_UNITS = {"e", "s-2"}


@dataclass(frozen=True)
class GridSourceConfig:
    """Description of the rectilinear map grid."""

    format: str
    file: Path
    row_order: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    height_column: Optional[str] = None
    x_variable: Optional[str] = None
    y_variable: Optional[str] = None
    height_variable: Optional[str] = None
    height_reference: str = "ellipsoid"


@dataclass(frozen=True)
class VectorSourceConfig:
    """Description of a three-component gravity field."""

    section: str
    format: str
    file: Path
    units: str
    frame: str
    row_order: str
    north_column: Optional[str] = None
    east_column: Optional[str] = None
    down_column: Optional[str] = None
    data_variable: Optional[str] = None
    axis_order: tuple[str, ...] = ()
    north_index: Optional[int] = None
    east_index: Optional[int] = None
    down_index: Optional[int] = None
    custom_to_ned: Optional[tuple[float, ...]] = None


@dataclass(frozen=True)
class TensorSourceConfig:
    """Optional validation mapping for six gravity-gradient components."""

    format: str
    file: Path
    units: str
    row_order: str
    columns: Optional[tuple[str, ...]] = None
    data_variable: Optional[str] = None
    axis_order: tuple[str, ...] = ()
    indices: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class CustomGravityConfig:
    """Complete configuration for a custom gravity map."""

    config_file: Path
    name: str
    crs: str
    mode: str
    interpolation: str
    out_of_bounds: str
    grid: GridSourceConfig
    field: VectorSourceConfig
    reference: Optional[VectorSourceConfig]
    tensor: Optional[TensorSourceConfig]


def read_custom_gravity_config(config_file: Path) -> CustomGravityConfig:
    """Read and validate a custom gravity-map INI file."""

    config_file = Path(config_file).resolve()
    config = ConfigHandler(config_file)

    _require_section(config, _MAP_SECTION)
    _require_section(config, _GRID_SECTION)
    _require_section(config, _FIELD_SECTION)

    name = _optional_str(config, _MAP_SECTION, "name", config_file.stem)
    crs = _required_str(config, _MAP_SECTION, "crs")
    mode = _choice(config, _MAP_SECTION, "mode", _MODES)
    interpolation = _choice(
        config, _MAP_SECTION, "interpolation", _INTERPOLATION, "linear")
    out_of_bounds = _choice(
        config, _MAP_SECTION, "outOfBounds", _COVERAGE, "base")

    grid = _read_grid(config, config_file.parent)
    field = _read_vector_source(config, _FIELD_SECTION, config_file.parent)

    reference = None
    if mode == "total_minus_reference":
        _require_section(config, _REFERENCE_SECTION)
        reference = _read_vector_source(
            config, _REFERENCE_SECTION, config_file.parent)

    tensor = _read_tensor_source(config, config_file.parent)

    return CustomGravityConfig(
        config_file=config_file,
        name=name,
        crs=crs,
        mode=mode,
        interpolation=interpolation,
        out_of_bounds=out_of_bounds,
        grid=grid,
        field=field,
        reference=reference,
        tensor=tensor,
    )


def _read_grid(config: ConfigHandler, base_dir: Path) -> GridSourceConfig:
    source_format = _choice(config, _GRID_SECTION, "format", _FORMATS)
    source_file = _source_path(config, _GRID_SECTION, base_dir)
    row_order = _choice(
        config, _GRID_SECTION, "rowOrder", _ROW_ORDERS, "x_fastest")
    height_reference = _choice(
        config, _GRID_SECTION, "heightReference",
        _HEIGHT_REFERENCES, "ellipsoid")

    if source_format == "csv":
        return GridSourceConfig(
            format=source_format,
            file=source_file,
            row_order=row_order,
            x_column=_required_str(config, _GRID_SECTION, "xColumn"),
            y_column=_required_str(config, _GRID_SECTION, "yColumn"),
            height_column=_optional_str(
                config, _GRID_SECTION, "heightColumn"),
            height_reference=height_reference,
        )

    return GridSourceConfig(
        format=source_format,
        file=source_file,
        row_order=row_order,
        x_variable=_required_str(config, _GRID_SECTION, "xVariable"),
        y_variable=_required_str(config, _GRID_SECTION, "yVariable"),
        height_variable=_optional_str(
            config, _GRID_SECTION, "heightVariable"),
        height_reference=height_reference,
    )


def _read_vector_source(
        config: ConfigHandler,
        section: str,
        base_dir: Path,
) -> VectorSourceConfig:
    source_format = _choice(config, section, "format", _FORMATS)
    source_file = _source_path(config, section, base_dir)
    units = _choice(config, section, "units", _UNITS)
    frame = _choice(config, section, "frame", _FRAMES)
    row_order = _choice(
        config, section, "rowOrder", _ROW_ORDERS, "x_fastest")

    custom_to_ned = None
    if frame == "custom":
        values = config.get_csv_numeric(section, "customToNed")
        if values is None or np.size(values) != 9:
            raise _error(
                section, "customToNed", "Invalid matrix",
                "A custom vector frame requires nine comma-separated "
                "custom-to-NED matrix values.")
        custom_to_ned = tuple(float(value) for value in values)

    common = {
        "section": section,
        "format": source_format,
        "file": source_file,
        "units": units,
        "frame": frame,
        "row_order": row_order,
        "custom_to_ned": custom_to_ned,
    }

    if source_format == "csv":
        return VectorSourceConfig(
            **common,
            north_column=_required_str(config, section, "northColumn"),
            east_column=_required_str(config, section, "eastColumn"),
            down_column=_required_str(config, section, "downColumn"),
        )

    axis_order = _axis_order(config, section)
    indices = (
        _nonnegative_int(config, section, "northIndex"),
        _nonnegative_int(config, section, "eastIndex"),
        _nonnegative_int(config, section, "downIndex"),
    )
    if len(set(indices)) != 3:
        raise _error(
            section, "northIndex/eastIndex/downIndex", "Duplicate index",
            "North, east, and down components must use distinct indexes.")

    return VectorSourceConfig(
        **common,
        data_variable=_required_str(config, section, "dataVariable"),
        axis_order=axis_order,
        north_index=indices[0],
        east_index=indices[1],
        down_index=indices[2],
    )


def _read_tensor_source(
        config: ConfigHandler,
        base_dir: Path,
) -> Optional[TensorSourceConfig]:
    if not config.parser.has_section(_TENSOR_SECTION):
        return None

    source_format = _choice(config, _TENSOR_SECTION, "format", _FORMATS)
    source_file = _source_path(config, _TENSOR_SECTION, base_dir)
    units = _choice(
        config, _TENSOR_SECTION, "units", _TENSOR_UNITS)
    row_order = _choice(
        config, _TENSOR_SECTION, "rowOrder", _ROW_ORDERS, "x_fastest")

    component_names = ("nn", "ee", "dd", "ne", "nd", "ed")
    if source_format == "csv":
        columns = tuple(
            _required_str(config, _TENSOR_SECTION, f"{name}Column")
            for name in component_names
        )
        return TensorSourceConfig(
            format=source_format,
            file=source_file,
            units=units,
            row_order=row_order,
            columns=columns,
        )

    indices = tuple(
        _nonnegative_int(config, _TENSOR_SECTION, f"{name}Index")
        for name in component_names
    )
    if len(set(indices)) != 6:
        raise _error(
            _TENSOR_SECTION, "component indexes", "Duplicate index",
            "All six tensor components must use distinct indexes.")

    return TensorSourceConfig(
        format=source_format,
        file=source_file,
        units=units,
        row_order=row_order,
        data_variable=_required_str(
            config, _TENSOR_SECTION, "dataVariable"),
        axis_order=_axis_order(config, _TENSOR_SECTION),
        indices=indices,
    )


def _axis_order(config: ConfigHandler, section: str) -> tuple[str, ...]:
    raw_value = _optional_str(
        config, section, "axisOrder", "component,x,y")
    axes = tuple(axis.strip().casefold() for axis in raw_value.split(","))
    if len(axes) != 3 or set(axes) != {"component", "x", "y"}:
        raise _error(
            section, "axisOrder", "Invalid axes",
            "axisOrder must contain component, x, and y exactly once.")
    return axes


def _source_path(
        config: ConfigHandler,
        section: str,
        base_dir: Path,
) -> Path:
    raw_path = _required_str(config, section, "file")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _choice(
        config: ConfigHandler,
        section: str,
        name: str,
        choices: set[str],
        fallback: Optional[str] = None,
) -> str:
    value = _optional_str(config, section, name, fallback)
    if value is None:
        raise _error(
            section, name, "Missing value",
            f"A value is required; supported values are {sorted(choices)}.")
    normalized = value.strip().casefold()
    if normalized not in choices:
        raise _error(
            section, name, "Unknown value",
            f"Unsupported value '{value}'; supported values are "
            f"{sorted(choices)}.")
    return normalized


def _nonnegative_int(
        config: ConfigHandler,
        section: str,
        name: str,
) -> int:
    value = config.get_int(section, name)
    if value < 0:
        raise _error(
            section, name, "Invalid index",
            "Component indexes must be non-negative.")
    return value


def _required_str(
        config: ConfigHandler,
        section: str,
        name: str,
) -> str:
    value = _optional_str(config, section, name)
    if value is None or value.strip() == "":
        raise _error(
            section, name, "Missing value",
            "A non-empty value is required.")
    return value.strip()


def _optional_str(
        config: ConfigHandler,
        section: str,
        name: str,
        fallback: Optional[str] = None,
) -> Optional[str]:
    if not config.parser.has_option(section, name):
        return fallback
    return config.get_str(section, name)


def _require_section(config: ConfigHandler, section: str) -> None:
    if not config.parser.has_section(section):
        raise _error(
            section, "", "Missing section",
            f"The [{section}] section is required.")


def _error(
        section: str,
        variable: str,
        error_type: str,
        description: str,
) -> NavConfigError:
    return NavConfigError(section, variable, error_type, description)
