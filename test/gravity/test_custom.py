import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyproj import Transformer

from qnav.gravity.custom import CustomGravityMap
from qnav.gravity.nima import WGS84Gravity
from qnav.gravity.simple import FixedValue
from qnav.input.config_handler import NavConfigError


class _ConstantGeoid:
    """Minimal non-dummy geoid used to verify anomaly conversions."""

    def __init__(self, height: float = 25.0):
        self.height = height

    def get_height(self, lat: float, lon: float) -> float:
        return self.height

    def get_height_vec(
        self, lat: np.ndarray, lon: np.ndarray,
    ) -> np.ndarray:
        return np.full(np.shape(lat), self.height, dtype=np.float64)


class _LimitedGeoid(_ConstantGeoid):
    """Fail if fallback handling queries outside the custom map."""

    def get_height_vec(
        self, lat: np.ndarray, lon: np.ndarray,
    ) -> np.ndarray:
        if np.any(lat < 60.0) or np.any(lon < 0.0):
            raise AssertionError("Geoid queried outside custom-map coverage")
        return super().get_height_vec(lat, lon)


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


def _make_scalar_map(
    tmp_path: Path,
    quantity: str = "gravity_disturbance",
    vertical_direction: str = "down",
    geoid: object | None = None,
) -> CustomGravityMap:
    x = np.array([10.0, 10.1, 10.2])
    y = np.array([70.0, 70.1, 70.2])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pd.DataFrame({"x": xx.ravel(), "y": yy.ravel()}).to_csv(
        tmp_path / "scalar_coordinates.csv", index=False)
    values = 10.0 + 2.0 * (xx - 10.0) + 3.0 * (yy - 70.0)
    pd.DataFrame({"value": values.ravel()}).to_csv(
        tmp_path / "scalar.csv", index=False)
    config_file = tmp_path / "scalar.ini"
    config_file.write_text(
        f"""
[CustomGravityMap]
name = Scalar custom map
crs = EPSG:4326
mode = residual
interpolation = linear
outOfBounds = base

[Grid]
format = csv
file = scalar_coordinates.csv
xColumn = x
yColumn = y
rowOrder = x_fastest

[Field]
format = csv
file = scalar.csv
representation = scalar
valueColumn = value
units = mGal
quantity = {quantity}
verticalDirection = {vertical_direction}
rowOrder = x_fastest
""",
        encoding="utf-8",
    )
    base = WGS84Gravity() if quantity == "free_air_anomaly" else FixedValue()
    return CustomGravityMap(base, geoid, config_file)


def test_scalar_disturbance_becomes_down_residual(tmp_path: Path):
    model = _make_scalar_map(tmp_path)

    np.testing.assert_allclose(
        model.get_residual(70.0, 10.0), [0.0, 0.0, 10e-5])
    assert model.get_disturbance(70.0, 10.0) == pytest.approx(10e-5)


def test_upward_scalar_direction_is_negated(tmp_path: Path):
    model = _make_scalar_map(tmp_path, vertical_direction="up")

    assert model.get_disturbance(70.0, 10.0) == pytest.approx(-10e-5)


def test_free_air_anomaly_preserves_anomaly_and_converts_disturbance(
    tmp_path: Path,
):
    model = _make_scalar_map(
        tmp_path, quantity="free_air_anomaly", geoid=_ConstantGeoid())

    anomaly = model.get_anomaly(70.0, 10.0)
    expected_disturbance = model._anomaly_to_disturbance(
        70.0, 10.0, anomaly)

    assert anomaly == pytest.approx(10e-5)
    assert model.get_disturbance(70.0, 10.0) == pytest.approx(
        expected_disturbance)
    assert model.get_residual(70.0, 10.0)[2] == pytest.approx(
        expected_disturbance)


def test_free_air_anomaly_requires_geoid_model(tmp_path: Path):
    with pytest.raises(NavConfigError, match="requires a configured geoid"):
        _make_scalar_map(tmp_path, quantity="free_air_anomaly")


def test_scalar_anomaly_outside_coverage_returns_base(tmp_path: Path):
    model = _make_scalar_map(
        tmp_path, quantity="free_air_anomaly", geoid=_ConstantGeoid())
    base = model.base_model.calc_gravity_z(0.0, 0.0, 100.0)

    assert model.get_anomaly(0.0, 0.0) == 0.0
    assert model.get_disturbance(0.0, 0.0) == 0.0
    assert model.calc_gravity_z(0.0, 0.0, 100.0) == pytest.approx(base)


def test_vectorized_anomaly_fallback_does_not_query_geoid_outside_map(
    tmp_path: Path,
):
    model = _make_scalar_map(
        tmp_path, quantity="free_air_anomaly", geoid=_LimitedGeoid())
    lat = np.array([70.0, 0.0])
    lon = np.array([10.0, 0.0])
    alt = np.array([100.0, 100.0])

    disturbance = model.get_disturbance_vec(lat, lon)
    gravity = model.calc_gravity_z_vec(lat, lon, alt)

    assert disturbance[0] != 0.0
    assert disturbance[1] == 0.0
    assert gravity[1] == pytest.approx(
        model.base_model.calc_gravity_z(lat[1], lon[1], alt[1]))


def test_headerless_whitespace_scalar_map_loads_directly(tmp_path: Path):
    rows = [
        "1 7.000000 -5.000000 10.0 0.0 100.0",
        "1 7.000000 -4.000000 11.0 0.0 100.0",
        "1 7.000000 -3.000000 12.0 0.0 100.0",
        "1 6.000000 -5.000000 20.0 0.0 100.0",
        "1 6.000000 -4.000000 21.0 0.0 100.0",
        "1 6.000000 -3.000000 22.0 0.0 100.0",
    ]
    (tmp_path / "public.xyz").write_text(
        "\n".join(rows), encoding="utf-8")
    config_file = tmp_path / "delimited.ini"
    config_file.write_text(
        """
[CustomGravityMap]
name = Headerless public map
crs = EPSG:4326
mode = residual
interpolation = linear
outOfBounds = error

[Grid]
format = delimited
file = public.xyz
delimiter = whitespace
header = none
xIndex = 2
yIndex = 1
rowOrder = x_fastest

[Field]
format = delimited
file = public.xyz
delimiter = whitespace
header = none
representation = scalar
valueIndex = 3
units = mGal
quantity = gravity_disturbance
verticalDirection = down
coordinateFrame = grid
gridXIndex = 2
gridYIndex = 1
rowOrder = x_fastest
""",
        encoding="utf-8",
    )

    model = CustomGravityMap(FixedValue(), None, config_file)

    np.testing.assert_allclose(model.map_data.x, [-5.0, -4.0, -3.0])
    np.testing.assert_allclose(model.map_data.y, [6.0, 7.0])
    assert model.get_disturbance(7.0, -5.0) == pytest.approx(10e-5)
    assert model.get_disturbance(6.0, -3.0) == pytest.approx(22e-5)


def test_headed_delimited_map_honours_skip_rows_and_comments(tmp_path: Path):
    (tmp_path / "headed.gdf").write_text(
        "metadata\nlon;lat;value\n"
        "# ignored\n10;50;1\n11;50;2\n10;51;3\n11;51;4\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "headed.ini"
    config_file.write_text(
        """
[CustomGravityMap]
crs = EPSG:4326
mode = residual

[Grid]
format = delimited
file = headed.gdf
delimiter = ;
header = present
skipRows = 1
commentPrefix = #
xColumn = lon
yColumn = lat
rowOrder = x_fastest

[Field]
format = delimited
file = headed.gdf
delimiter = ;
header = present
skipRows = 1
commentPrefix = #
representation = scalar
valueColumn = value
units = mGal
quantity = gravity_disturbance
verticalDirection = down
rowOrder = x_fastest
""",
        encoding="utf-8",
    )

    model = CustomGravityMap(FixedValue(), None, config_file)

    assert model.get_disturbance(51.0, 11.0) == pytest.approx(4e-5)


def test_headerless_delimited_vector_map_uses_component_indexes(
    tmp_path: Path,
):
    (tmp_path / "vector.xyz").write_text(
        "10 50 1 2 3\n11 50 4 5 6\n"
        "10 51 7 8 9\n11 51 10 11 12\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "vector.ini"
    config_file.write_text(
        """
[CustomGravityMap]
crs = EPSG:4326
mode = residual

[Grid]
format = delimited
file = vector.xyz
header = none
xIndex = 0
yIndex = 1
rowOrder = x_fastest

[Field]
format = delimited
file = vector.xyz
header = none
representation = vector
northIndex = 2
eastIndex = 3
downIndex = 4
units = mGal
frame = NED
quantity = residual
rowOrder = x_fastest
""",
        encoding="utf-8",
    )

    model = CustomGravityMap(FixedValue(), None, config_file)

    np.testing.assert_allclose(
        model.get_residual(51.0, 11.0), np.array([10, 11, 12]) * 1e-5)


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


def test_scalar_and_vector_down_match_full_vector_with_wgs84(
    tmp_path: Path,
):
    configured, _, _, _ = _make_map(tmp_path)
    model = CustomGravityMap(WGS84Gravity(), None, configured.config)
    lat = np.array([70.0, 70.1, 70.2])
    lon = np.array([10.0, 10.1, 10.2])
    alt = np.array([0.0, 100.0, 200.0])

    scalar_down = np.array([
        model.calc_gravity_z(a, b, c)
        for a, b, c in zip(lat, lon, alt)
    ])
    vector_down = model.calc_gravity_z_vec(lat, lon, alt)
    full_down = model.calc_gravity_xyz_vec(lat, lon, alt)[:, 2]

    np.testing.assert_allclose(scalar_down, full_down)
    np.testing.assert_allclose(vector_down, full_down)


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


def test_csv_grid_and_field_are_subset_consistently(tmp_path: Path):
    original, _, _, expected = _make_map(tmp_path)
    with original.config.config_file.open("a", encoding="utf-8") as file:
        file.write(
            "\n[Subset]\ncoordinateFrame = grid\n"
            "minX = 10.0\nmaxX = 10.1\n"
            "minY = 70.0\nmaxY = 70.1\n"
            "paddingCells = 0\n"
        )

    model = CustomGravityMap(
        FixedValue(), None, original.config.config_file)

    np.testing.assert_allclose(model.map_data.x, [10.0, 10.1])
    np.testing.assert_allclose(model.map_data.y, [70.0, 70.1])
    np.testing.assert_allclose(
        model.get_residual(70.1, 10.1), expected[1, 1])


def test_projected_crs_boundary_roundoff_remains_in_coverage(
    tmp_path: Path,
):
    model, _, _, expected = _make_map(tmp_path, crs="EPSG:3413")
    inverse = Transformer.from_crs(
        "EPSG:3413", "EPSG:4326", always_xy=True)
    x = np.array([model.map_data.x[0], model.map_data.x[-1]])
    y = np.array([model.map_data.y[0], model.map_data.y[-1]])
    longitude, latitude = inverse.transform(x, y)

    result = model.get_residual_vec(latitude, longitude)

    np.testing.assert_allclose(
        result, np.stack((expected[0, 0], expected[-1, -1])), atol=1e-12)


@pytest.mark.parametrize(
    ("map_longitudes", "query_longitude"),
    (
        ((-180.0, -179.9, -179.8), 180.1),
        ((180.0, 180.1, 180.2), -179.9),
        ((350.0, 350.1, 350.2), -9.9),
    ),
)
def test_geographic_query_uses_map_longitude_convention(
    tmp_path: Path,
    map_longitudes: tuple[float, float, float],
    query_longitude: float,
):
    model, _, _, expected = _make_map(tmp_path)
    coordinates_file = tmp_path / "coordinates.csv"
    coordinates = pd.read_csv(coordinates_file)
    coordinates["x"] = np.tile(map_longitudes, 3)
    coordinates.to_csv(coordinates_file, index=False)
    model = CustomGravityMap(FixedValue(), None, tmp_path / "map.ini")

    result = model.get_residual(70.1, query_longitude)

    np.testing.assert_allclose(result, expected[1, 1])


def test_longitude_normalization_does_not_mask_out_of_bounds_query(
    tmp_path: Path,
):
    model, _, _, _ = _make_map(tmp_path, out_of_bounds="error")

    with pytest.raises(ValueError, match="cannot provide"):
        model.get_residual(70.1, 190.0)


@pytest.mark.parametrize(
    "map_longitudes",
    (
        (-180.0, 0.0, 180.0),
        (0.0, 180.0, 360.0),
        (-200.0, 0.0, 200.0),
    ),
)
def test_rejects_ambiguous_geographic_longitude_span(
    tmp_path: Path,
    map_longitudes: tuple[float, float, float],
):
    _make_map(tmp_path)
    coordinates_file = tmp_path / "coordinates.csv"
    coordinates = pd.read_csv(coordinates_file)
    coordinates["x"] = np.tile(map_longitudes, 3)
    coordinates.to_csv(coordinates_file, index=False)

    with pytest.raises(ValueError, match="less than one revolution"):
        CustomGravityMap(FixedValue(), None, tmp_path / "map.ini")


@pytest.mark.parametrize(
    "map_longitudes",
    (
        (-179.999, 0.0, 179.999),
        (359820.001, 360000.0, 360179.999),
    ),
)
def test_accepts_geographic_axis_just_under_one_revolution(
    tmp_path: Path,
    map_longitudes: tuple[float, float, float],
):
    _make_map(tmp_path)
    coordinates_file = tmp_path / "coordinates.csv"
    coordinates = pd.read_csv(coordinates_file)
    coordinates["x"] = np.tile(map_longitudes, 3)
    coordinates.to_csv(coordinates_file, index=False)

    model = CustomGravityMap(FixedValue(), None, tmp_path / "map.ini")

    assert model.map_data.x[-1] - model.map_data.x[0] < 360.0


@pytest.mark.parametrize(
    ("latitude", "longitude"),
    (
        (70.1, np.nan),
        (70.1, np.inf),
        (70.1, -np.inf),
        (np.nan, 10.1),
        (np.inf, 10.1),
        (-np.inf, 10.1),
    ),
)
def test_nonfinite_query_uses_coverage_policy_without_warning(
    tmp_path: Path,
    latitude: float,
    longitude: float,
):
    model, _, _, _ = _make_map(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = model.get_residual(latitude, longitude)

    np.testing.assert_allclose(result, 0.0)


def test_nonfinite_query_respects_error_policy_without_warning(
    tmp_path: Path,
):
    model, _, _, _ = _make_map(tmp_path, out_of_bounds="error")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match="cannot provide"):
            model.get_residual(np.inf, 10.1)


def test_vectorized_query_isolates_nonfinite_rows(tmp_path: Path):
    model, _, _, expected = _make_map(tmp_path)
    latitude = np.array([70.1, np.nan, 70.1])
    longitude = np.array([10.1, 10.1, np.inf])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = model.get_residual_vec(latitude, longitude)

    np.testing.assert_allclose(result[0], expected[1, 1])
    np.testing.assert_allclose(result[1:], 0.0)


def test_rejects_mismatched_query_shapes(tmp_path: Path):
    model, _, _, _ = _make_map(tmp_path)

    with pytest.raises(ValueError, match="same shape"):
        model.get_residual_vec(
            np.array([70.0, 70.1]), np.array([10.0]))
