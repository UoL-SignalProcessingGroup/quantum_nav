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

[Reference]
format = csv
file = reference.csv
northColumn = north
eastColumn = east
downColumn = down
units = m/s2
frame = ENU
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
    assert result.field.north_index == 1
    assert result.reference is not None
    assert result.reference.frame == "enu"


def test_residual_mode_does_not_require_reference(tmp_path: Path):
    body = _valid_config().replace(
        "mode = total_minus_reference", "mode = residual")
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
    config_file = _write_config(tmp_path, body)

    result = read_custom_gravity_config(config_file)

    assert result.field.custom_to_ned == (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("crs = EPSG:3413", ""),
        ("mode = total_minus_reference", "mode = unsupported"),
        ("axisOrder = component,x,y", "axisOrder = component,x,x"),
        ("northIndex = 1", "northIndex = -1"),
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
