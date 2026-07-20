"""
Configuration parsing for user-supplied gravity correction maps.

Custom gravity maps deliberately use a separate INI file.  This keeps file
layout, coordinate reference system, units, and vector-frame information
alongside the data source rather than hard-coding them in QNav.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, overload

import numpy as np

from qnav.input.config_handler import ConfigHandler, NavConfigError


_MAP_SECTION = "CustomGravityMap"
_GRID_SECTION = "Grid"
_FIELD_SECTION = "Field"
_REFERENCE_SECTION = "Reference"
_TENSOR_SECTION = "Tensor"

_FORMATS = {"csv", "delimited", "mat"}
_TENSOR_FORMATS = {"csv", "mat"}
_ROW_ORDERS = {"x_fastest", "y_fastest"}
_UNITS = {"m/s2", "gal", "mgal"}
_REPRESENTATIONS = {"vector", "scalar"}
_VERTICAL_DIRECTIONS = {"down", "up"}
_FRAMES = {"ned", "enu", "ecef", "geocentric_ned", "custom"}
_TENSOR_FRAMES = {"ned", "ecef", "geocentric_ned", "custom"}
_QUANTITIES = {
    "residual",
    "effective_gravity",
    "gravitational_attraction",
    "gravity_disturbance",
    "free_air_anomaly",
}
_COORDINATE_FRAMES = {"none", "grid", "wgs84"}
_MODES = {"residual", "total_minus_reference"}
_INTERPOLATION = {"linear", "nearest"}
_COVERAGE = {"base", "error"}
_HEIGHT_REFERENCES = {"ellipsoid", "orthometric", "geoid_surface"}
_TENSOR_UNITS = {"e", "s-2"}
_DELIMITED_HEADERS = {"none", "present"}


@dataclass(frozen=True)
class DelimitedSourceConfig:
    """Parsing options for configurable delimited text files."""

    delimiter: str
    header: str
    skip_rows: int
    comment_prefix: Optional[str]


@dataclass(frozen=True)
class GridSourceConfig:
    """Description of the rectilinear map grid."""

    format: str
    file: Path
    row_order: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    height_column: Optional[str] = None
    geoid_undulation_column: Optional[str] = None
    x_variable: Optional[str] = None
    y_variable: Optional[str] = None
    height_variable: Optional[str] = None
    geoid_undulation_variable: Optional[str] = None
    height_reference: str = "ellipsoid"
    delimited: Optional[DelimitedSourceConfig] = None
    x_index: Optional[int] = None
    y_index: Optional[int] = None
    height_index: Optional[int] = None
    geoid_undulation_index: Optional[int] = None


@dataclass(frozen=True)
class VectorSourceConfig:
    """Description of a three-component gravity field."""

    section: str
    format: str
    file: Path
    units: str
    frame: str
    quantity: str
    row_order: str
    representation: str = "vector"
    vertical_direction: Optional[str] = None
    coordinate_frame: str = "none"
    coordinate_columns: Optional[tuple[str, str]] = None
    component_columns: Optional[tuple[str, str, str]] = None
    value_column: Optional[str] = None
    data_variable: Optional[str] = None
    axis_order: tuple[str, ...] = ()
    component_indices: Optional[tuple[int, int, int]] = None
    value_variable: Optional[str] = None
    custom_to_ned: Optional[tuple[float, ...]] = None
    delimited: Optional[DelimitedSourceConfig] = None
    coordinate_indices: Optional[tuple[int, int]] = None
    value_index: Optional[int] = None


@dataclass(frozen=True)
class TensorSourceConfig:
    """Optional validation mapping for six gravity-gradient components."""

    format: str
    file: Path
    units: str
    frame: str
    row_order: str
    columns: Optional[tuple[str, ...]] = None
    data_variable: Optional[str] = None
    axis_order: tuple[str, ...] = ()
    indices: Optional[tuple[int, ...]] = None
    custom_to_ned: Optional[tuple[float, ...]] = None


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
    default_field_quantity = (
        "residual" if mode == "residual" else "effective_gravity")
    field = _read_vector_source(
        config,
        _FIELD_SECTION,
        config_file.parent,
        default_field_quantity,
    )

    reference = None
    if mode == "total_minus_reference":
        _require_section(config, _REFERENCE_SECTION)
        reference = _read_vector_source(
            config,
            _REFERENCE_SECTION,
            config_file.parent,
            "effective_gravity",
        )

    if mode == "residual" and field.quantity not in {
        "residual", "gravity_disturbance", "free_air_anomaly"
    }:
        raise _error(
            _FIELD_SECTION,
            "quantity",
            "Incompatible quantity",
            "mode=residual requires a residual, gravity disturbance, or "
            "free-air anomaly.",
        )
    if field.representation == "scalar" and field.quantity not in {
        "gravity_disturbance", "free_air_anomaly"
    }:
        raise _error(
            _FIELD_SECTION,
            "quantity",
            "Incompatible scalar quantity",
            "Scalar fields require quantity=gravity_disturbance or "
            "quantity=free_air_anomaly.",
        )
    if mode == "total_minus_reference" and (
        field.representation != "vector"
        or (reference is not None and reference.representation != "vector")
    ):
        raise _error(
            _MAP_SECTION,
            "mode",
            "Incompatible scalar field",
            "total_minus_reference requires vector field and reference "
            "sources.",
        )
    if mode == "total_minus_reference" and (
        field.quantity == "residual"
        or reference is None
        or reference.quantity == "residual"
    ):
        raise _error(
            _MAP_SECTION,
            "mode",
            "Incompatible quantity",
            "total_minus_reference requires total effective-gravity or "
            "gravitational-attraction fields.",
        )

    tensor = _read_tensor_source(config, config_file.parent)
    _validate_shared_delimited_options(grid, field, reference)

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


def _validate_shared_delimited_options(
    grid: GridSourceConfig,
    field: VectorSourceConfig,
    reference: Optional[VectorSourceConfig],
) -> None:
    sources = [field]
    if reference is not None:
        sources.append(reference)
    for source in sources:
        if (
            grid.format == "delimited"
            and source.format == "delimited"
            and grid.file == source.file
            and grid.delimited != source.delimited
        ):
            raise _error(
                source.section,
                "delimiter/header/skipRows/commentPrefix",
                "Inconsistent text parsing",
                "Sections reading the same delimited file must use "
                "identical parsing options.",
            )


def _read_grid(config: ConfigHandler, base_dir: Path) -> GridSourceConfig:
    source_format = _choice(config, _GRID_SECTION, "format", _FORMATS)
    source_file = _source_path(config, _GRID_SECTION, base_dir)
    row_order = _choice(
        config, _GRID_SECTION, "rowOrder", _ROW_ORDERS, "x_fastest")
    height_reference = _choice(
        config, _GRID_SECTION, "heightReference",
        _HEIGHT_REFERENCES, "ellipsoid")

    if source_format in {"csv", "delimited"}:
        delimited = (
            _read_delimited_options(config, _GRID_SECTION)
            if source_format == "delimited"
            else None
        )
        if delimited is not None and delimited.header == "none":
            height_index = _optional_nonnegative_int(
                config, _GRID_SECTION, "heightIndex")
            geoid_index = _optional_nonnegative_int(
                config, _GRID_SECTION, "geoidUndulationIndex")
            if (
                height_reference == "orthometric"
                and height_index is not None
                and geoid_index is None
            ):
                raise _error(
                    _GRID_SECTION,
                    "geoidUndulationIndex",
                    "Missing vertical datum conversion",
                    "Orthometric heights require a geoid-undulation index.",
                )
            return GridSourceConfig(
                format=source_format,
                file=source_file,
                row_order=row_order,
                height_reference=height_reference,
                delimited=delimited,
                x_index=_nonnegative_int(config, _GRID_SECTION, "xIndex"),
                y_index=_nonnegative_int(config, _GRID_SECTION, "yIndex"),
                height_index=height_index,
                geoid_undulation_index=geoid_index,
            )
        height_column = _optional_str(
            config, _GRID_SECTION, "heightColumn")
        geoid_undulation_column = _optional_str(
            config, _GRID_SECTION, "geoidUndulationColumn")
        if (
            height_reference == "orthometric"
            and height_column is not None
            and geoid_undulation_column is None
        ):
            raise _error(
                _GRID_SECTION,
                "geoidUndulationColumn",
                "Missing vertical datum conversion",
                "Orthometric heights require a geoid-undulation column so "
                "they can be converted to WGS-84 ellipsoid heights.",
            )
        return GridSourceConfig(
            format=source_format,
            file=source_file,
            row_order=row_order,
            x_column=_required_str(config, _GRID_SECTION, "xColumn"),
            y_column=_required_str(config, _GRID_SECTION, "yColumn"),
            height_column=height_column,
            geoid_undulation_column=geoid_undulation_column,
            height_reference=height_reference,
            delimited=delimited,
        )

    height_variable = _optional_str(
        config, _GRID_SECTION, "heightVariable")
    geoid_undulation_variable = _optional_str(
        config, _GRID_SECTION, "geoidUndulationVariable")
    if (
        height_reference == "orthometric"
        and height_variable is not None
        and geoid_undulation_variable is None
    ):
        raise _error(
            _GRID_SECTION,
            "geoidUndulationVariable",
            "Missing vertical datum conversion",
            "Orthometric heights require a geoid-undulation variable so "
            "they can be converted to WGS-84 ellipsoid heights.",
        )
    return GridSourceConfig(
        format=source_format,
        file=source_file,
        row_order=row_order,
        x_variable=_required_str(config, _GRID_SECTION, "xVariable"),
        y_variable=_required_str(config, _GRID_SECTION, "yVariable"),
        height_variable=height_variable,
        geoid_undulation_variable=geoid_undulation_variable,
        height_reference=height_reference,
    )


def _read_vector_source(
        config: ConfigHandler,
        section: str,
        base_dir: Path,
        default_quantity: str,
) -> VectorSourceConfig:
    source_format = _choice(config, section, "format", _FORMATS)
    source_file = _source_path(config, section, base_dir)
    units = _choice(config, section, "units", _UNITS)
    representation = _choice(
        config, section, "representation", _REPRESENTATIONS, "vector")
    quantity = _choice(
        config, section, "quantity", _QUANTITIES, default_quantity)
    row_order = _choice(
        config, section, "rowOrder", _ROW_ORDERS, "x_fastest")

    if representation == "scalar":
        vertical_direction = _choice(
            config, section, "verticalDirection", _VERTICAL_DIRECTIONS)
        return _read_scalar_source(
            config,
            section,
            source_format,
            source_file,
            units,
            quantity,
            row_order,
            vertical_direction,
        )

    frame = _choice(config, section, "frame", _FRAMES)
    custom_to_ned = _read_custom_rotation(config, section, frame)

    component_names = {
        "ned": ("north", "east", "down"),
        "enu": ("east", "north", "up"),
        "ecef": ("x", "y", "z"),
        "geocentric_ned": ("north", "east", "down"),
        "custom": ("x", "y", "z"),
    }[frame]

    if source_format in {"csv", "delimited"}:
        delimited = (
            _read_delimited_options(config, section)
            if source_format == "delimited"
            else None
        )
        coordinate_frame = _choice(
            config,
            section,
            "coordinateFrame",
            _COORDINATE_FRAMES,
            "none",
        )
        if delimited is not None and delimited.header == "none":
            coordinate_indices = _read_coordinate_indices(
                config, section, coordinate_frame)
            indices = tuple(
                _nonnegative_int(config, section, f"{name}Index")
                for name in component_names
            )
            if len(set(indices)) != 3:
                raise _error(
                    section,
                    "component indexes",
                    "Duplicate index",
                    "All vector components require distinct indexes.",
                )
            return VectorSourceConfig(
                section=section,
                format=source_format,
                file=source_file,
                units=units,
                frame=frame,
                quantity=quantity,
                row_order=row_order,
                representation=representation,
                coordinate_frame=coordinate_frame,
                component_indices=indices,
                coordinate_indices=coordinate_indices,
                custom_to_ned=custom_to_ned,
                delimited=delimited,
            )
        coordinate_columns = _read_coordinate_columns(
            config, section, coordinate_frame)
        columns = tuple(
            _required_str(config, section, f"{name}Column")
            for name in component_names
        )
        return VectorSourceConfig(
            section=section,
            format=source_format,
            file=source_file,
            units=units,
            frame=frame,
            quantity=quantity,
            row_order=row_order,
            representation=representation,
            coordinate_frame=coordinate_frame,
            coordinate_columns=coordinate_columns,
            component_columns=(columns[0], columns[1], columns[2]),
            custom_to_ned=custom_to_ned,
            delimited=delimited,
        )

    coordinate_frame = _choice(
        config,
        section,
        "coordinateFrame",
        _COORDINATE_FRAMES,
        "none",
    )
    if coordinate_frame != "none":
        raise _error(
            section,
            "coordinateFrame",
            "Unsupported MAT coordinate mapping",
            "Per-row coordinate validation is currently supported for CSV "
            "vector sources only.",
        )
    axis_order = _axis_order(config, section)
    indices = tuple(
        _nonnegative_int(config, section, f"{name}Index")
        for name in component_names
    )
    if len(set(indices)) != 3:
        raise _error(
            section, "northIndex/eastIndex/downIndex", "Duplicate index",
            "North, east, and down components must use distinct indexes.")

    return VectorSourceConfig(
        section=section,
        format=source_format,
        file=source_file,
        units=units,
        frame=frame,
        quantity=quantity,
        row_order=row_order,
        representation=representation,
        data_variable=_required_str(config, section, "dataVariable"),
        axis_order=axis_order,
        component_indices=(indices[0], indices[1], indices[2]),
        custom_to_ned=custom_to_ned,
    )


def _read_scalar_source(
    config: ConfigHandler,
    section: str,
    source_format: str,
    source_file: Path,
    units: str,
    quantity: str,
    row_order: str,
    vertical_direction: str,
) -> VectorSourceConfig:
    """Read a scalar vertical gravity source."""

    if source_format in {"csv", "delimited"}:
        delimited = (
            _read_delimited_options(config, section)
            if source_format == "delimited"
            else None
        )
        coordinate_frame = _choice(
            config,
            section,
            "coordinateFrame",
            _COORDINATE_FRAMES,
            "none",
        )
        if delimited is not None and delimited.header == "none":
            return VectorSourceConfig(
                section=section,
                format=source_format,
                file=source_file,
                units=units,
                frame="ned",
                quantity=quantity,
                row_order=row_order,
                representation="scalar",
                vertical_direction=vertical_direction,
                coordinate_frame=coordinate_frame,
                coordinate_indices=_read_coordinate_indices(
                    config, section, coordinate_frame),
                value_index=_nonnegative_int(
                    config, section, "valueIndex"),
                delimited=delimited,
            )
        return VectorSourceConfig(
            section=section,
            format=source_format,
            file=source_file,
            units=units,
            frame="ned",
            quantity=quantity,
            row_order=row_order,
            representation="scalar",
            vertical_direction=vertical_direction,
            coordinate_frame=coordinate_frame,
            coordinate_columns=_read_coordinate_columns(
                config, section, coordinate_frame),
            value_column=_required_str(config, section, "valueColumn"),
            delimited=delimited,
        )

    coordinate_frame = _choice(
        config, section, "coordinateFrame", _COORDINATE_FRAMES, "none")
    if coordinate_frame != "none":
        raise _error(
            section,
            "coordinateFrame",
            "Unsupported MAT coordinate mapping",
            "Per-row coordinate validation is supported for CSV scalar "
            "sources only.",
        )
    return VectorSourceConfig(
        section=section,
        format=source_format,
        file=source_file,
        units=units,
        frame="ned",
        quantity=quantity,
        row_order=row_order,
        representation="scalar",
        vertical_direction=vertical_direction,
        value_variable=_required_str(config, section, "valueVariable"),
        axis_order=_scalar_axis_order(config, section),
    )


def _read_tensor_source(
        config: ConfigHandler,
        base_dir: Path,
) -> Optional[TensorSourceConfig]:
    assert config.parser is not None
    if not config.parser.has_section(_TENSOR_SECTION):
        return None

    source_format = _choice(
        config, _TENSOR_SECTION, "format", _TENSOR_FORMATS)
    source_file = _source_path(config, _TENSOR_SECTION, base_dir)
    units = _choice(
        config, _TENSOR_SECTION, "units", _TENSOR_UNITS)
    frame = _choice(
        config, _TENSOR_SECTION, "frame", _TENSOR_FRAMES, "ned")
    custom_to_ned = _read_custom_rotation(
        config, _TENSOR_SECTION, frame)
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
            frame=frame,
            row_order=row_order,
            columns=columns,
            custom_to_ned=custom_to_ned,
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
        frame=frame,
        row_order=row_order,
        data_variable=_required_str(
            config, _TENSOR_SECTION, "dataVariable"),
        axis_order=_axis_order(config, _TENSOR_SECTION),
        indices=indices,
        custom_to_ned=custom_to_ned,
    )


def _read_coordinate_columns(
    config: ConfigHandler,
    section: str,
    coordinate_frame: str,
) -> Optional[tuple[str, str]]:
    if coordinate_frame == "none":
        return None
    if coordinate_frame == "grid":
        return (
            _required_str(config, section, "gridXColumn"),
            _required_str(config, section, "gridYColumn"),
        )
    return (
        _required_str(config, section, "latitudeColumn"),
        _required_str(config, section, "longitudeColumn"),
    )


def _read_coordinate_indices(
    config: ConfigHandler,
    section: str,
    coordinate_frame: str,
) -> Optional[tuple[int, int]]:
    if coordinate_frame == "none":
        return None
    if coordinate_frame == "grid":
        return (
            _nonnegative_int(config, section, "gridXIndex"),
            _nonnegative_int(config, section, "gridYIndex"),
        )
    return (
        _nonnegative_int(config, section, "latitudeIndex"),
        _nonnegative_int(config, section, "longitudeIndex"),
    )


def _read_delimited_options(
    config: ConfigHandler,
    section: str,
) -> DelimitedSourceConfig:
    delimiter = _optional_str(
        config, section, "delimiter", "whitespace").strip()
    if delimiter.casefold() == "whitespace":
        delimiter = "whitespace"
    elif delimiter == r"\t":
        delimiter = "\t"
    elif len(delimiter) != 1:
        raise _error(
            section,
            "delimiter",
            "Invalid delimiter",
            "delimiter must be 'whitespace', '\\t', or one character.",
        )
    header = _choice(
        config, section, "header", _DELIMITED_HEADERS, "none")
    skip_rows = _optional_nonnegative_int(
        config, section, "skipRows", fallback=0)
    comment_prefix = _optional_str(config, section, "commentPrefix")
    if comment_prefix is not None and len(comment_prefix) != 1:
        raise _error(
            section,
            "commentPrefix",
            "Invalid comment prefix",
            "commentPrefix must contain exactly one character.",
        )
    return DelimitedSourceConfig(
        delimiter=delimiter,
        header=header,
        skip_rows=skip_rows,
        comment_prefix=comment_prefix,
    )


def _read_custom_rotation(
    config: ConfigHandler,
    section: str,
    frame: str,
) -> Optional[tuple[float, ...]]:
    if frame != "custom":
        return None
    values = config.get_csv_numeric(section, "customToNed")
    if values is None or np.size(values) != 9:
        raise _error(
            section,
            "customToNed",
            "Invalid matrix",
            "A custom frame requires nine comma-separated custom-to-NED "
            "matrix values.",
        )
    return tuple(float(value) for value in np.asarray(values).ravel())


def _axis_order(config: ConfigHandler, section: str) -> tuple[str, ...]:
    raw_value = _optional_str(
        config, section, "axisOrder", "component,x,y")
    axes = tuple(axis.strip().casefold() for axis in raw_value.split(","))
    if len(axes) != 3 or set(axes) != {"component", "x", "y"}:
        raise _error(
            section, "axisOrder", "Invalid axes",
            "axisOrder must contain component, x, and y exactly once.")
    return axes


def _scalar_axis_order(
    config: ConfigHandler,
    section: str,
) -> tuple[str, ...]:
    raw_value = _optional_str(config, section, "axisOrder", "x,y")
    axes = tuple(axis.strip().casefold() for axis in raw_value.split(","))
    if len(axes) != 2 or set(axes) != {"x", "y"}:
        raise _error(
            section,
            "axisOrder",
            "Invalid axes",
            "Scalar axisOrder must contain x and y exactly once.",
        )
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


def _optional_nonnegative_int(
    config: ConfigHandler,
    section: str,
    name: str,
    fallback: Optional[int] = None,
) -> Optional[int]:
    if not config.is_value_set(section, name):
        return fallback
    return _nonnegative_int(config, section, name)


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


@overload
def _optional_str(
        config: ConfigHandler,
        section: str,
        name: str,
        fallback: str,
) -> str:
    ...


@overload
def _optional_str(
        config: ConfigHandler,
        section: str,
        name: str,
        fallback: None = None,
) -> Optional[str]:
    ...


def _optional_str(
        config: ConfigHandler,
        section: str,
        name: str,
        fallback: Optional[str] = None,
) -> Optional[str]:
    assert config.parser is not None
    if not config.parser.has_option(section, name):
        return fallback
    return config.get_str(section, name)


def _require_section(config: ConfigHandler, section: str) -> None:
    assert config.parser is not None
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
