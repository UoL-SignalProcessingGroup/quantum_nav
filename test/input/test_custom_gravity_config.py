from pathlib import Path

import pytest

from qnav.input.config_handler import NavConfigError
from qnav.input.ini.custom_gravity_config import (
    read_custom_gravity_config,
)


def _write_config(tmp_path: Path, body: str) -> Path:
    config_file = tmp_path / "map.ini"
    config_file.write_text(body, encoding="utf-8")
    return config_file


def _valid_config() -> str:
    return """
[CustomGravityMap]
name = Synthetic map
crs = EPSG:3413
mode = total_minus_reference
interpolation = linear
outOfBounds = base

[Grid]
format = csv
file = coordinates.csv
xColumn = x
yColumn = y
heightColumn = h
heightReference = ellipsoid
rowOrder = x_fastest

[Field]
format = mat
file = field.mat
dataVariable = field
axisOrder = component,x,y
northIndex = 1
eastIndex = 2
downIndex = 3
units = mGal
frame = NED
quantity = effective_gravity

[Reference]
format = csv
file = reference.csv
northColumn = north
eastColumn = east
upColumn = up
units = m/s2
frame = ENU
quantity = gravitational_attraction
rowOrder = x_fastest
"""


def test_reads_custom_gravity_config(tmp_path: Path):
    config_file = _write_config(tmp_path, _valid_config())

    result = read_custom_gravity_config(config_file)

    assert result.name == "Synthetic map"
    assert result.crs == "EPSG:3413"
    assert result.mode == "total_minus_reference"
    assert result.grid.file == (tmp_path / "coordinates.csv").resolve()
    assert result.grid.height_column == "h"
    assert result.field.data_variable == "field"
    assert result.field.axis_order == ("component", "x", "y")
    assert result.field.component_indices == (1, 2, 3)
    assert result.field.quantity == "effective_gravity"
    assert result.reference is not None
    assert result.reference.frame == "enu"
    assert result.reference.quantity == "gravitational_attraction"


def test_residual_mode_does_not_require_reference(tmp_path: Path):
    body = _valid_config().replace(
        "mode = total_minus_reference", "mode = residual")
    body = body.replace("quantity = effective_gravity\n", "", 1)
    body = body.split("[Reference]")[0]
    config_file = _write_config(tmp_path, body)

    result = read_custom_gravity_config(config_file)

    assert result.mode == "residual"
    assert result.reference is None


def test_reads_custom_rotation(tmp_path: Path):
    body = _valid_config().replace(
        "frame = NED",
        "frame = custom\ncustomToNed = 1,0,0,0,1,0,0,0,1",
        1,
    )
    body = body.replace("northIndex = 1", "xIndex = 1", 1)
    body = body.replace("eastIndex = 2", "yIndex = 2", 1)
    body = body.replace("downIndex = 3", "zIndex = 3", 1)
    config_file = _write_config(tmp_path, body)

    result = read_custom_gravity_config(config_file)

    assert result.field.custom_to_ned == (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )


def test_reads_geocentric_frame_and_coordinate_validation(tmp_path: Path):
    body = _valid_config().replace(
        "format = mat\nfile = field.mat",
        "format = csv\nfile = field.csv",
        1,
    )
    body = body.replace(
        """dataVariable = field
axisOrder = component,x,y
northIndex = 1
eastIndex = 2
downIndex = 3""",
        """northColumn = north
eastColumn = east
downColumn = down
coordinateFrame = wgs84
latitudeColumn = latitude
longitudeColumn = longitude""",
        1,
    )
    body = body.replace("frame = NED", "frame = geocentric_ned", 1)
    config_file = _write_config(tmp_path, body)

    result = read_custom_gravity_config(config_file)

    assert result.field.frame == "geocentric_ned"
    assert result.field.coordinate_frame == "wgs84"
    assert result.field.coordinate_columns == ("latitude", "longitude")


@pytest.mark.parametrize(
    "declaration",
    [
        "quantity = effective_gravity\n",
        "quantity = gravitational_attraction\n",
    ],
)
def test_total_minus_reference_requires_explicit_quantities(
    tmp_path: Path,
    declaration: str,
):
    body = _valid_config().replace(declaration, "", 1)
    config_file = _write_config(tmp_path, body)

    with pytest.raises(NavConfigError, match="explicit quantity"):
        read_custom_gravity_config(config_file)


@pytest.mark.parametrize(
    "old,new",
    [
        ("crs = EPSG:3413", ""),
        ("mode = total_minus_reference", "mode = unsupported"),
        ("axisOrder = component,x,y", "axisOrder = component,x,x"),
        ("northIndex = 1", "northIndex = -1"),
        (
            "quantity = effective_gravity",
            "quantity = residual",
        ),
    ],
)
def test_rejects_invalid_configuration(
    tmp_path: Path,
    old: str,
    new: str,
):
    config_file = _write_config(
        tmp_path, _valid_config().replace(old, new))

    with pytest.raises(NavConfigError):
        read_custom_gravity_config(config_file)


def test_reference_is_required_for_subtraction(tmp_path: Path):
    body = _valid_config().split("[Reference]")[0]
    config_file = _write_config(tmp_path, body)

    with pytest.raises(NavConfigError, match="Reference"):
        read_custom_gravity_config(config_file)


def test_residual_mode_rejects_total_quantity(tmp_path: Path):
    body = _valid_config().replace(
        "mode = total_minus_reference", "mode = residual")
    body = body.split("[Reference]")[0]
    config_file = _write_config(tmp_path, body)

    with pytest.raises(NavConfigError, match="mode=residual"):
        read_custom_gravity_config(config_file)


def test_rejects_vector_free_air_anomaly(tmp_path: Path):
    body = _valid_config().replace(
        "mode = total_minus_reference", "mode = residual")
    body = body.split("[Reference]")[0]
    body = body.replace(
        "quantity = effective_gravity", "quantity = free_air_anomaly")
    config_file = _write_config(tmp_path, body)

    with pytest.raises(
        NavConfigError, match="free_air_anomaly is a scalar quantity"
    ):
        read_custom_gravity_config(config_file)


@pytest.mark.parametrize("section", ["field", "reference"])
def test_total_minus_reference_rejects_disturbance_quantities(
    tmp_path: Path, section: str,
):
    body = _valid_config()
    if section == "field":
        body = body.replace(
            "quantity = effective_gravity",
            "quantity = gravity_disturbance",
        )
    else:
        body = body.replace(
            "quantity = gravitational_attraction",
            "quantity = gravity_disturbance",
        )
    config_file = _write_config(tmp_path, body)

    with pytest.raises(
        NavConfigError,
        match="requires total effective-gravity or gravitational-attraction",
    ):
        read_custom_gravity_config(config_file)


def test_orthometric_height_requires_geoid_undulation(tmp_path: Path):
    body = _valid_config().replace(
        "heightReference = ellipsoid",
        "heightReference = orthometric",
    )
    config_file = _write_config(tmp_path, body)

    with pytest.raises(NavConfigError, match="geoid-undulation"):
        read_custom_gravity_config(config_file)
