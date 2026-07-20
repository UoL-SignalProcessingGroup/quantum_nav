from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scipy.io import savemat

from qnav.gravity.custom_data import load_custom_map_data
from qnav.input.config_handler import NavConfigError
from qnav.input.ini.custom_gravity_config import (
    read_custom_gravity_config,
)
from qnav.util import constants
from qnav.util.transformations import (
    ecef2ned_transform_vec,
    lla2ecef_vec,
)


def _write_csv_inputs(tmp_path: Path):
    x = np.array([0.0, 100.0, 200.0])
    y = np.array([1000.0, 1100.0])
    xx, yy = np.meshgrid(x, y, indexing="xy")

    coordinates = pd.DataFrame({
        "x": xx.ravel(),
        "y": yy.ravel(),
        "height": np.full(xx.size, 50.0),
    })
    coordinates.to_csv(tmp_path / "coordinates.csv", index=False)

    north = 10.0 + xx + 2.0 * yy
    east = 20.0 + 3.0 * xx - yy
    down = 30.0 - xx + 0.5 * yy
    field = pd.DataFrame({
        "north": north.ravel(),
        "east": east.ravel(),
        "down": down.ravel(),
    })
    field.to_csv(tmp_path / "field.csv", index=False)

    reference = pd.DataFrame({
        "north": np.full(xx.size, 1.0),
        "east": np.full(xx.size, 2.0),
        "down": np.full(xx.size, 3.0),
    })
    reference.to_csv(tmp_path / "reference.csv", index=False)
    total = np.stack((north, east, down), axis=-1)
    return x, y, np.transpose(total, (1, 0, 2))


def _write_config(
    tmp_path: Path,
    field_section: str,
    reference_section: str = "",
    mode: str = "total_minus_reference",
    crs: str = "EPSG:3413",
) -> Path:
    config_file = tmp_path / "map.ini"
    config_file.write_text(
        f"""
[CustomGravityMap]
name = Synthetic map
crs = {crs}
mode = {mode}
interpolation = linear
outOfBounds = base

[Grid]
format = csv
file = coordinates.csv
xColumn = x
yColumn = y
heightColumn = height
rowOrder = x_fastest

{field_section}

{reference_section}
""",
        encoding="utf-8",
    )
    return config_file


def _csv_field(section: str, file_name: str, frame: str = "NED") -> str:
    if frame == "ENU":
        columns = """
eastColumn = east
northColumn = north
upColumn = up
"""
    else:
        columns = """
northColumn = north
eastColumn = east
downColumn = down
"""
    return f"""
[{section}]
format = csv
file = {file_name}
{columns}
units = mGal
frame = {frame}
rowOrder = x_fastest
"""


def test_loads_and_subtracts_csv_fields(tmp_path: Path):
    x, y, total = _write_csv_inputs(tmp_path)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        _csv_field("Reference", "reference.csv"),
    )

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    expected_reference = np.array([1.0, 2.0, 3.0]) * 1e-5
    np.testing.assert_allclose(loaded.x, x)
    np.testing.assert_allclose(loaded.y, y)
    np.testing.assert_allclose(
        loaded.residual, total * 1e-5 - expected_reference)
    assert loaded.height.shape == (3, 2)


def test_loads_mat_field_and_matches_csv(tmp_path: Path):
    _, _, total = _write_csv_inputs(tmp_path)
    packed = np.moveaxis(total, -1, 0)
    savemat(tmp_path / "field.mat", {"gravity": packed})
    field_section = """
[Field]
format = mat
file = field.mat
dataVariable = gravity
axisOrder = component,x,y
northIndex = 0
eastIndex = 1
downIndex = 2
units = m/s2
frame = NED
"""
    config_file = _write_config(
        tmp_path, field_section, mode="residual")

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    np.testing.assert_allclose(loaded.residual, total)


def test_converts_enu_to_ned(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    enu = pd.DataFrame({
        "east": np.full(6, 1.0),
        "north": np.full(6, 2.0),
        "up": np.full(6, 3.0),
    })
    enu.to_csv(tmp_path / "enu.csv", index=False)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "enu.csv", "ENU"),
        mode="residual",
    )

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    expected = np.broadcast_to(
        np.array([2.0, 1.0, -3.0]) * 1e-5,
        loaded.residual.shape,
    )
    np.testing.assert_allclose(loaded.residual, expected)


def test_rejects_invalid_crs(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        mode="residual",
        crs="NOT-A-CRS",
    )

    with pytest.raises(NavConfigError, match="Invalid CRS"):
        load_custom_map_data(
            read_custom_gravity_config(config_file))


def test_rejects_three_dimensional_crs(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        mode="residual",
        crs="EPSG:4979",
    )

    with pytest.raises(NavConfigError, match="two-dimensional"):
        load_custom_map_data(
            read_custom_gravity_config(config_file))


def test_rejects_wrong_row_order(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        mode="residual",
    )
    body = config_file.read_text(encoding="utf-8")
    config_file.write_text(
        body.replace(
            "rowOrder = x_fastest", "rowOrder = y_fastest", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rowOrder"):
        load_custom_map_data(
            read_custom_gravity_config(config_file))


def test_loads_mat_grid_struct(tmp_path: Path):
    _, _, total = _write_csv_inputs(tmp_path)
    savemat(
        tmp_path / "map.mat",
        {
            "coord": {
                "x": np.array([0.0, 100.0, 200.0]),
                "y": np.array([1000.0, 1100.0]),
                "height": np.full((3, 2), 50.0),
            },
            "gravity": np.moveaxis(total, -1, 0),
        },
    )
    config_file = tmp_path / "map.ini"
    config_file.write_text(
        """
[CustomGravityMap]
crs = EPSG:3413
mode = residual

[Grid]
format = mat
file = map.mat
xVariable = coord.x
yVariable = coord.y
heightVariable = coord.height

[Field]
format = mat
file = map.mat
dataVariable = gravity
axisOrder = component,x,y
northIndex = 0
eastIndex = 1
downIndex = 2
units = m/s2
frame = NED
""",
        encoding="utf-8",
    )

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    np.testing.assert_allclose(loaded.residual, total)
    np.testing.assert_allclose(loaded.height, 50.0)


def test_converts_custom_frame_to_ned(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    pd.DataFrame({
        "x": np.full(6, 1.0),
        "y": np.full(6, 2.0),
        "z": np.full(6, 3.0),
    }).to_csv(tmp_path / "custom.csv", index=False)
    field_section = """
[Field]
format = csv
file = custom.csv
xColumn = x
yColumn = y
zColumn = z
units = m/s2
frame = custom
customToNed = 0,-1,0,1,0,0,0,0,1
rowOrder = x_fastest
"""
    config_file = _write_config(
        tmp_path, field_section, mode="residual")

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    expected = np.broadcast_to(
        np.array([-2.0, 1.0, 3.0]), loaded.residual.shape)
    np.testing.assert_allclose(loaded.residual, expected)


def test_rejects_non_rotation_custom_matrix(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    pd.DataFrame({
        "x": np.full(6, 1.0),
        "y": np.full(6, 2.0),
        "z": np.full(6, 3.0),
    }).to_csv(tmp_path / "custom.csv", index=False)
    field_section = """
[Field]
format = csv
file = custom.csv
xColumn = x
yColumn = y
zColumn = z
units = m/s2
frame = custom
customToNed = 1,0,0,0,1,0,0,0,2
rowOrder = x_fastest
"""
    config_file = _write_config(
        tmp_path, field_section, mode="residual")

    with pytest.raises(NavConfigError, match="orthonormal"):
        load_custom_map_data(
            read_custom_gravity_config(config_file))


def test_converts_ecef_vectors_at_each_grid_node(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    pd.DataFrame({
        "x": np.full(6, 1.0),
        "y": np.full(6, 2.0),
        "z": np.full(6, 3.0),
    }).to_csv(tmp_path / "ecef.csv", index=False)
    field_section = """
[Field]
format = csv
file = ecef.csv
xColumn = x
yColumn = y
zColumn = z
units = m/s2
frame = ECEF
rowOrder = x_fastest
"""
    config_file = _write_config(
        tmp_path, field_section, mode="residual")

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    lla = np.column_stack((
        loaded.latitude.ravel(),
        loaded.longitude.ravel(),
        np.zeros(loaded.latitude.size),
    ))
    expected = np.einsum(
        "nij,j->ni",
        ecef2ned_transform_vec(lla),
        np.array([1.0, 2.0, 3.0]),
    ).reshape(loaded.residual.shape)
    np.testing.assert_allclose(loaded.residual, expected)


def test_converts_geocentric_local_frame_to_geodetic_ned(tmp_path: Path):
    longitude = np.array([10.0, 10.1])
    latitude = np.array([70.0, 70.1])
    lon_grid, lat_grid = np.meshgrid(
        longitude, latitude, indexing="xy")
    height = np.full(lon_grid.size, 50.0)
    pd.DataFrame({
        "x": lon_grid.ravel(),
        "y": lat_grid.ravel(),
        "height": height,
    }).to_csv(tmp_path / "coordinates.csv", index=False)

    lla = np.column_stack((
        lat_grid.ravel(),
        lon_grid.ravel(),
        height,
    ))
    ecef = lla2ecef_vec(lla)
    geocentric_latitude = np.degrees(np.arctan2(
        ecef[:, 2], np.hypot(ecef[:, 0], ecef[:, 1])))
    delta = np.radians(lat_grid.ravel() - geocentric_latitude)
    desired_down = 9.8
    pd.DataFrame({
        "north": -np.sin(delta) * desired_down,
        "east": np.zeros(delta.size),
        "down": np.cos(delta) * desired_down,
    }).to_csv(tmp_path / "field.csv", index=False)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv", "geocentric_ned"),
        mode="residual",
        crs="EPSG:4326",
    )

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    expected = np.broadcast_to(
        np.array([0.0, 0.0, desired_down * 1e-5]),
        loaded.residual.shape,
    )
    np.testing.assert_allclose(loaded.residual, expected, atol=1e-14)


def test_normalizes_gravitational_attraction_to_effective_gravity(
    tmp_path: Path,
):
    _write_csv_inputs(tmp_path)
    field = pd.read_csv(tmp_path / "field.csv")
    field.loc[:, :] = 0.0
    field.to_csv(tmp_path / "field.csv", index=False)

    reference = pd.read_csv(tmp_path / "reference.csv")
    reference.loc[:, :] = 0.0
    reference.to_csv(tmp_path / "reference.csv", index=False)
    reference_section = _csv_field("Reference", "reference.csv")
    reference_section += "\nquantity = gravitational_attraction\n"
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        reference_section,
    )

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    flat_ecef = lla2ecef_vec(np.column_stack((
        loaded.latitude.ravel(),
        loaded.longitude.ravel(),
        loaded.height.ravel(),
    )))
    centrifugal_ecef = np.column_stack((
        constants.OMEGA_E ** 2 * flat_ecef[:, 0],
        constants.OMEGA_E ** 2 * flat_ecef[:, 1],
        np.zeros(flat_ecef.shape[0]),
    ))
    lla = np.column_stack((
        loaded.latitude.ravel(),
        loaded.longitude.ravel(),
        loaded.height.ravel(),
    ))
    expected_reference = np.einsum(
        "nij,nj->ni",
        ecef2ned_transform_vec(lla),
        centrifugal_ecef,
    ).reshape(loaded.residual.shape)
    np.testing.assert_allclose(loaded.residual, -expected_reference)


def test_validates_vector_source_coordinates(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    coordinates = pd.read_csv(tmp_path / "coordinates.csv")
    field = pd.read_csv(tmp_path / "field.csv")
    field["grid_x"] = coordinates["x"]
    field["grid_y"] = coordinates["y"]
    field.loc[3, "grid_x"] += 1.0
    field.to_csv(tmp_path / "field.csv", index=False)
    field_section = _csv_field("Field", "field.csv")
    field_section += """
coordinateFrame = grid
gridXColumn = grid_x
gridYColumn = grid_y
"""
    config_file = _write_config(
        tmp_path, field_section, mode="residual")

    with pytest.raises(ValueError, match="do not align"):
        load_custom_map_data(
            read_custom_gravity_config(config_file))


def test_converts_orthometric_to_ellipsoid_height(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    coordinates = pd.read_csv(tmp_path / "coordinates.csv")
    coordinates["height"] = 20.0
    coordinates["undulation"] = 30.0
    coordinates.to_csv(tmp_path / "coordinates.csv", index=False)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        mode="residual",
    )
    body = config_file.read_text(encoding="utf-8")
    body = body.replace(
        "heightColumn = height",
        "heightColumn = height\n"
        "heightReference = orthometric\n"
        "geoidUndulationColumn = undulation",
    )
    config_file.write_text(body, encoding="utf-8")

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    np.testing.assert_allclose(loaded.ellipsoid_height, 50.0)


def test_validates_csv_tensor_shape_and_columns(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    tensor_file = tmp_path / "tensor.csv"
    pd.DataFrame({
        name: np.arange(6, dtype=float)
        for name in ("nn", "ee", "dd", "ne", "nd", "ed")
    }).to_csv(tensor_file, index=False)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        mode="residual",
    )
    with config_file.open("a", encoding="utf-8") as stream:
        stream.write(
            """
[Tensor]
format = csv
file = tensor.csv
nnColumn = nn
eeColumn = ee
ddColumn = dd
neColumn = ne
ndColumn = nd
edColumn = ed
units = E
rowOrder = x_fastest
"""
        )

    loaded = load_custom_map_data(
        read_custom_gravity_config(config_file))

    assert loaded.residual.shape == (3, 2, 3)


def test_rejects_nonfinite_tensor_values(tmp_path: Path):
    _write_csv_inputs(tmp_path)
    tensor = {
        name: np.arange(6, dtype=float)
        for name in ("nn", "ee", "dd", "ne", "nd", "ed")
    }
    tensor["nd"][2] = np.nan
    pd.DataFrame(tensor).to_csv(tmp_path / "tensor.csv", index=False)
    config_file = _write_config(
        tmp_path,
        _csv_field("Field", "field.csv"),
        mode="residual",
    )
    with config_file.open("a", encoding="utf-8") as stream:
        stream.write(
            """
[Tensor]
format = csv
file = tensor.csv
nnColumn = nn
eeColumn = ee
ddColumn = dd
neColumn = ne
ndColumn = nd
edColumn = ed
units = E
rowOrder = x_fastest
"""
        )

    with pytest.raises(ValueError, match="only finite"):
        load_custom_map_data(
            read_custom_gravity_config(config_file))
