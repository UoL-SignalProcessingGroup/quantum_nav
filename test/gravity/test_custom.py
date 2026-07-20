from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyproj import Transformer

from qnav.gravity.custom import CustomGravityMap
from qnav.gravity.simple import FixedValue


def _make_map(
    tmp_path: Path,
    out_of_bounds: str = "base",
    crs: str = "EPSG:4326",
    nan_cell: bool = False,
) -> tuple[CustomGravityMap, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([10.0, 10.1, 10.2])
    y = np.array([70.0, 70.1, 70.2])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coordinates = pd.DataFrame({
        "x": xx.ravel(),
        "y": yy.ravel(),
    })
    coordinates.to_csv(tmp_path / "coordinates.csv", index=False)

    north = 1e-3 + 2e-3 * (xx - 10.0) + 3e-3 * (yy - 70.0)
    east = -2e-3 + 4e-3 * (xx - 10.0)
    down = 5e-3 - 2e-3 * (yy - 70.0)
    residual_rows = np.stack((north, east, down), axis=-1)
    if nan_cell:
        residual_rows[1, 1, :] = np.nan
    pd.DataFrame({
        "north": residual_rows[:, :, 0].ravel(),
        "east": residual_rows[:, :, 1].ravel(),
        "down": residual_rows[:, :, 2].ravel(),
    }).to_csv(tmp_path / "residual.csv", index=False)

    if crs != "EPSG:4326":
        projected_x = np.array([1_000_000.0, 1_000_100.0, 1_000_200.0])
        projected_y = np.array(
            [-1_700_000.0, -1_699_900.0, -1_699_800.0])
        grid_x, grid_y = np.meshgrid(
            projected_x, projected_y, indexing="xy")
        coordinates["x"] = grid_x.ravel()
        coordinates["y"] = grid_y.ravel()
        coordinates.to_csv(tmp_path / "coordinates.csv", index=False)

    config_file = tmp_path / "map.ini"
    config_file.write_text(
        f"""
[CustomGravityMap]
name = Synthetic custom map
crs = {crs}
mode = residual
interpolation = linear
outOfBounds = {out_of_bounds}

[Grid]
format = csv
file = coordinates.csv
xColumn = x
yColumn = y
rowOrder = x_fastest

[Field]
format = csv
file = residual.csv
northColumn = north
eastColumn = east
downColumn = down
units = m/s2
frame = NED
rowOrder = x_fastest
""",
        encoding="utf-8",
    )
    expected = np.transpose(residual_rows, (1, 0, 2))
    return CustomGravityMap(FixedValue(), None, config_file), x, y, expected


def test_interpolates_full_ned_residual(tmp_path: Path):
    model, _, _, expected = _make_map(tmp_path)

    result = model.get_residual(70.1, 10.1)

    np.testing.assert_allclose(result, expected[1, 1])
    np.testing.assert_allclose(
        model.calc_gravity_xyz(70.1, 10.1, 200.0),
        np.array([0.0, 0.0, model.base_model.calc_gravity_z()]) +
        expected[1, 1],
    )
    assert model.calc_gravity_z(
        70.1, 10.1, 200.0) == pytest.approx(
            model.calc_gravity_xyz(70.1, 10.1, 200.0)[2])


def test_linear_interpolation_at_cell_centre(tmp_path: Path):
    model, _, _, expected = _make_map(tmp_path)

    result = model.get_residual(70.05, 10.05)
    corners = expected[0:2, 0:2, :]

    np.testing.assert_allclose(result, np.mean(corners, axis=(0, 1)))


def test_vectorized_results_match_scalar_results(tmp_path: Path):
    model, _, _, _ = _make_map(tmp_path)
    lat = np.array([70.0, 70.05, 70.2])
    lon = np.array([10.0, 10.05, 10.2])
    alt = np.array([0.0, 100.0, 200.0])

    residual = model.get_residual_vec(lat, lon)
    gravity = model.calc_gravity_xyz_vec(lat, lon, alt)

    assert residual.shape == (3, 3)
    assert gravity.shape == (3, 3)
    for index in range(3):
        np.testing.assert_allclose(
            residual[index], model.get_residual(lat[index], lon[index]))
        np.testing.assert_allclose(
            gravity[index],
            model.calc_gravity_xyz(lat[index], lon[index], alt[index]),
        )


def test_base_fallback_outside_map(tmp_path: Path):
    model, _, _, _ = _make_map(tmp_path)
    base = model.base_model.calc_gravity_xyz(0.0, 0.0, 0.0)

    np.testing.assert_allclose(
        model.calc_gravity_xyz(0.0, 0.0, 0.0), base)
    assert model.get_disturbance(0.0, 0.0) == 0.0


def test_error_policy_outside_map(tmp_path: Path):
    model, _, _, _ = _make_map(tmp_path, out_of_bounds="error")

    with pytest.raises(ValueError, match="cannot provide"):
        model.get_residual(0.0, 0.0)


def test_nan_cell_uses_coverage_policy(tmp_path: Path):
    model, _, _, _ = _make_map(tmp_path, nan_cell=True)

    np.testing.assert_allclose(model.get_residual(70.1, 10.1), 0.0)


def test_projected_crs_lookup(tmp_path: Path):
    model, _, _, expected = _make_map(tmp_path, crs="EPSG:3413")
    inverse = Transformer.from_crs(
        "EPSG:3413", "EPSG:4326", always_xy=True)
    longitude, latitude = inverse.transform(
        model.map_data.x[1], model.map_data.y[1])

    np.testing.assert_allclose(
        model.get_residual(latitude, longitude),
        expected[1, 1],
        atol=1e-12,
    )


def test_rejects_mismatched_query_shapes(tmp_path: Path):
    model, _, _, _ = _make_map(tmp_path)

    with pytest.raises(ValueError, match="same shape"):
        model.get_residual_vec(
            np.array([70.0, 70.1]), np.array([10.0]))
