from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qnav.gravity.custom import CustomGravityMap
from qnav.gravity import custom_data
from qnav.gravity.simple import FixedValue
from qnav.input.config_handler import NavConfigError


def _write_scalar_config(
    path: Path,
    source_format: str,
    data_file: str,
    grid_mapping: str,
    field_mapping: str,
    crs: str = "EPSG:4326",
    extra: str = "",
    quantity: str = "gravity_disturbance",
) -> Path:
    path.write_text(
        f"""
[CustomGravityMap]
name = Raster custom map
crs = {crs}
mode = residual
interpolation = linear
outOfBounds = base
{extra}

[Grid]
format = {source_format}
file = {data_file}
{grid_mapping}

[Field]
format = {source_format}
file = {data_file}
representation = scalar
units = mGal
quantity = {quantity}
verticalDirection = down
{field_mapping}
""",
        encoding="utf-8",
    )
    return path


def test_geotiff_scalar_uses_pixel_centres_and_nodata(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data_file = tmp_path / "map.tif"
    values = np.array([[1, 2, 3], [4, -9999, 6]], dtype=np.float32)
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10.0, 52.0, 1.0, 1.0),
        nodata=-9999,
    ) as destination:
        destination.write(values, 1)
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "geotiff",
        data_file.name,
        "",
        "valueBand = 1",
    )

    model = CustomGravityMap(FixedValue(), None, config)

    np.testing.assert_allclose(model.map_data.x, [10.5, 11.5, 12.5])
    np.testing.assert_allclose(model.map_data.y, [50.5, 51.5])
    assert model.get_disturbance(51.5, 10.5) == pytest.approx(1e-5)
    assert model.get_disturbance(50.5, 11.5) == 0.0


def test_geotiff_applies_band_scale_and_offset(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data_file = tmp_path / "scaled.tif"
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=from_origin(10.0, 52.0, 1.0, 1.0),
    ) as destination:
        destination.write(np.full((2, 2), 2, dtype=np.int16), 1)
        destination.scales = (0.5,)
        destination.offsets = (10.0,)
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "geotiff",
        data_file.name,
        "",
        "valueBand = 1",
    )

    model = CustomGravityMap(FixedValue(), None, config)

    assert model.get_disturbance(51.5, 10.5) == pytest.approx(11e-5)


def test_geotiff_rejects_crs_mismatch(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data_file = tmp_path / "map.tif"
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10.0, 52.0, 1.0, 1.0),
    ) as destination:
        destination.write(np.ones((2, 2), dtype=np.float32), 1)
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "geotiff",
        data_file.name,
        "",
        "valueBand = 1",
        crs="EPSG:3857",
    )

    with pytest.raises(ValueError, match="Embedded CRS does not match"):
        CustomGravityMap(FixedValue(), None, config)


def test_geotiff_rejects_rotated_grid(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    data_file = tmp_path / "rotated.tif"
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=Affine(1.0, 0.1, 10.0, 0.0, -1.0, 52.0),
    ) as destination:
        destination.write(np.ones((2, 2), dtype=np.float32), 1)
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "geotiff",
        data_file.name,
        "",
        "valueBand = 1",
    )

    with pytest.raises(ValueError, match="non-rotated rectilinear grid"):
        CustomGravityMap(FixedValue(), None, config)


def test_geotiff_vector_maps_bands_through_enu_frame(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data_file = tmp_path / "vector.tif"
    with rasterio.open(
        data_file,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(10.0, 52.0, 1.0, 1.0),
    ) as destination:
        destination.write(np.full((2, 2), 1.0, dtype=np.float32), 1)
        destination.write(np.full((2, 2), 2.0, dtype=np.float32), 2)
        destination.write(np.full((2, 2), 3.0, dtype=np.float32), 3)
    config = tmp_path / "vector.ini"
    config.write_text(
        """
[CustomGravityMap]
crs = EPSG:4326
mode = residual
[Grid]
format = geotiff
file = vector.tif
[Field]
format = geotiff
file = vector.tif
representation = vector
eastBand = 1
northBand = 2
upBand = 3
units = mGal
frame = ENU
quantity = residual
""",
        encoding="utf-8",
    )

    model = CustomGravityMap(FixedValue(), None, config)

    np.testing.assert_allclose(
        model.get_residual(51.5, 10.5), np.array([2, 1, -3]) * 1e-5)


def test_netcdf_scalar_transposes_named_dimensions(tmp_path: Path):
    xarray = pytest.importorskip("xarray")
    from pyproj import CRS

    data_file = tmp_path / "map.nc"
    data = xarray.Dataset(
        data_vars={
            "gravity": (("latitude", "longitude"), [[1, 2, 3], [4, 5, 6]])
        },
        coords={
            "longitude": [10.0, 11.0, 12.0],
            "latitude": [50.0, 51.0],
        },
        attrs={"crs_wkt": CRS.from_epsg(4326).to_wkt()},
    )
    data.to_netcdf(data_file, engine="scipy")
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        data_file.name,
        "xVariable = longitude\nyVariable = latitude",
        "valueVariable = gravity",
    )

    model = CustomGravityMap(FixedValue(), None, config)

    assert model.get_disturbance(50.0, 10.0) == pytest.approx(1e-5)
    assert model.get_disturbance(51.0, 12.0) == pytest.approx(6e-5)


def test_netcdf_decodes_cf_scale_fill_and_descending_axes(tmp_path: Path):
    xarray = pytest.importorskip("xarray")

    data_file = tmp_path / "packed.nc"
    data = xarray.Dataset(
        data_vars={
            "gravity": (
                ("latitude", "longitude"),
                np.array([[11.0, 12.0, 13.0], [21.0, np.nan, 23.0]]),
            )
        },
        coords={
            "longitude": [12.0, 11.0, 10.0],
            "latitude": [51.0, 50.0],
        },
    )
    data["gravity"].encoding = {
        "dtype": "int16",
        "scale_factor": 0.5,
        "add_offset": 10.0,
        "_FillValue": -9999,
    }
    data.to_netcdf(data_file, engine="scipy")
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        data_file.name,
        "xVariable = longitude\nyVariable = latitude",
        "valueVariable = gravity",
    )

    model = CustomGravityMap(FixedValue(), None, config)

    np.testing.assert_allclose(model.map_data.x, [10.0, 11.0, 12.0])
    np.testing.assert_allclose(model.map_data.y, [50.0, 51.0])
    assert model.get_disturbance(51.0, 10.0) == pytest.approx(13e-5)
    assert model.get_disturbance(50.0, 11.0) == 0.0


@pytest.mark.parametrize("configured_epsg", [3413, 4326])
def test_netcdf_validates_pure_cf_grid_mapping(
    tmp_path: Path, configured_epsg: int,
):
    xarray = pytest.importorskip("xarray")
    from pyproj import CRS

    data_file = tmp_path / "cf.nc"
    cf_attributes = CRS.from_epsg(3413).to_cf()
    cf_attributes.pop("crs_wkt", None)
    unrelated_cf = CRS.from_epsg(4326).to_cf()
    unrelated_cf.pop("crs_wkt", None)
    data = xarray.Dataset(
        data_vars={
            "gravity": (("y", "x"), np.ones((2, 2))),
            "polar_mapping": ((), 0, cf_attributes),
            "unrelated": (("y", "x"), np.ones((2, 2))),
            "geographic_mapping": ((), 0, unrelated_cf),
        },
        coords={"x": [0.0, 1.0], "y": [-1.0, 0.0]},
    )
    data["gravity"].attrs["grid_mapping"] = "polar_mapping"
    data["unrelated"].attrs["grid_mapping"] = "geographic_mapping"
    data.to_netcdf(data_file, engine="scipy")
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        data_file.name,
        "xVariable = x\nyVariable = y",
        "valueVariable = gravity",
        crs=f"EPSG:{configured_epsg}",
    )

    if configured_epsg == 3413:
        model = CustomGravityMap(FixedValue(), None, config)
        assert model.map_data.crs.to_epsg() == 3413
    else:
        with pytest.raises(ValueError, match="Embedded CRS does not match"):
            CustomGravityMap(FixedValue(), None, config)


def test_netcdf_rejects_non_singleton_extra_dimension(tmp_path: Path):
    xarray = pytest.importorskip("xarray")

    data_file = tmp_path / "map.nc"
    data = xarray.Dataset(
        data_vars={
            "gravity": (
                ("time", "latitude", "longitude"),
                np.ones((2, 2, 2)),
            )
        },
        coords={
            "time": [0, 1],
            "longitude": [10.0, 11.0],
            "latitude": [50.0, 51.0],
        },
    )
    data.to_netcdf(data_file, engine="scipy")
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        data_file.name,
        "xVariable = longitude\nyVariable = latitude",
        "valueVariable = gravity",
    )

    with pytest.raises(ValueError, match="non-singleton dimension 'time'"):
        CustomGravityMap(FixedValue(), None, config)


def test_netcdf_vector_uses_named_component_variables(tmp_path: Path):
    xarray = pytest.importorskip("xarray")

    data_file = tmp_path / "vector.nc"
    shape = (2, 2)
    data = xarray.Dataset(
        data_vars={
            "north": (("y", "x"), np.full(shape, 1.0)),
            "east": (("y", "x"), np.full(shape, 2.0)),
            "down": (("y", "x"), np.full(shape, 3.0)),
        },
        coords={"x": [10.0, 11.0], "y": [50.0, 51.0]},
    )
    data.to_netcdf(data_file, engine="scipy")
    config = tmp_path / "vector.ini"
    config.write_text(
        """
[CustomGravityMap]
crs = EPSG:4326
mode = residual
[Grid]
format = netcdf
file = vector.nc
xVariable = x
yVariable = y
[Field]
format = netcdf
file = vector.nc
representation = vector
northVariable = north
eastVariable = east
downVariable = down
units = mGal
frame = NED
quantity = residual
""",
        encoding="utf-8",
    )

    model = CustomGravityMap(FixedValue(), None, config)

    np.testing.assert_allclose(
        model.get_residual(50.0, 10.0), np.array([1, 2, 3]) * 1e-5)


def test_subset_and_cell_guard_apply_before_interpolation(tmp_path: Path):
    xarray = pytest.importorskip("xarray")

    data_file = tmp_path / "map.nc"
    data = xarray.Dataset(
        data_vars={"gravity": (("y", "x"), np.arange(36).reshape(6, 6))},
        coords={"x": np.arange(6.0), "y": np.arange(6.0)},
    )
    data.to_netcdf(data_file, engine="scipy")
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        data_file.name,
        "xVariable = x\nyVariable = y",
        "valueVariable = gravity",
        extra="maxCells = 4",
    )

    with pytest.raises(ValueError, match="36 cells; maxCells is 4"):
        CustomGravityMap(FixedValue(), None, config)

    with config.open("a", encoding="utf-8") as file:
        file.write(
            "\n[Subset]\ncoordinateFrame = grid\n"
            "minX = 2\nmaxX = 3\nminY = 2\nmaxY = 3\n"
            "paddingCells = 0\n"
        )
    model = CustomGravityMap(FixedValue(), None, config)

    np.testing.assert_allclose(model.map_data.x, [2.0, 3.0])
    np.testing.assert_allclose(model.map_data.y, [2.0, 3.0])


def test_delimited_subset_streams_shared_grid_and_field(tmp_path: Path):
    data_file = tmp_path / "map.xyz"
    x, y = np.meshgrid(np.arange(6.0), np.arange(5.0), indexing="xy")
    pd.DataFrame({
        "x": x.ravel(),
        "y": y.ravel(),
        "gravity": (10.0 * y + x).ravel(),
    }).to_csv(data_file, sep=" ", index=False)
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "delimited",
        data_file.name,
        "delimiter = whitespace\nheader = present\n"
        "xColumn = x\nyColumn = y",
        "delimiter = whitespace\nheader = present\n"
        "valueColumn = gravity",
        extra="maxCells = 4",
    )
    with config.open("a", encoding="utf-8") as file:
        file.write(
            "\n[Subset]\ncoordinateFrame = grid\n"
            "minX = 2\nmaxX = 3\nminY = 1\nmaxY = 2\n"
            "paddingCells = 0\n"
        )

    model = CustomGravityMap(FixedValue(), None, config)

    np.testing.assert_allclose(model.map_data.x, [2.0, 3.0])
    np.testing.assert_allclose(model.map_data.y, [1.0, 2.0])
    assert model.get_disturbance(2.0, 3.0) == pytest.approx(23e-5)


def test_wgs84_subset_is_transformed_to_projected_map_crs(tmp_path: Path):
    xarray = pytest.importorskip("xarray")
    from pyproj import Transformer

    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:3413", always_xy=True)
    centre_x, centre_y = transformer.transform(-45.0, 75.0)
    axis_x = centre_x + np.arange(-2, 3) * 10_000.0
    axis_y = centre_y + np.arange(-2, 3) * 10_000.0
    data_file = tmp_path / "polar.nc"
    xarray.Dataset(
        data_vars={"gravity": (("y", "x"), np.ones((5, 5)))},
        coords={"x": axis_x, "y": axis_y},
    ).to_netcdf(data_file, engine="scipy")
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        data_file.name,
        "xVariable = x\nyVariable = y",
        "valueVariable = gravity",
        crs="EPSG:3413",
        extra="maxCells = 9",
    )
    with config.open("a", encoding="utf-8") as file:
        file.write(
            "\n[Subset]\ncoordinateFrame = wgs84\n"
            "minLongitude = -45.01\nmaxLongitude = -44.99\n"
            "minLatitude = 74.99\nmaxLatitude = 75.01\n"
            "paddingCells = 1\n"
        )

    model = CustomGravityMap(FixedValue(), None, config)

    assert model.map_data.x.size <= 3
    assert model.map_data.y.size <= 3
    assert model.get_disturbance(75.0, -45.0) == pytest.approx(1e-5)


def test_missing_raster_reader_has_actionable_install_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    data_file = tmp_path / "map.tif"
    data_file.touch()
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "geotiff",
        data_file.name,
        "",
        "valueBand = 1",
    )
    real_import = custom_data.import_module

    def reject_rasterio(name: str):
        if name == "rasterio":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr(custom_data, "import_module", reject_rasterio)

    with pytest.raises(NavConfigError, match=r"qnav\[maps\]"):
        CustomGravityMap(FixedValue(), None, config)


@pytest.mark.parametrize(
    "quantity", ["bouguer_anomaly", "isostatic_anomaly", "gravity_anomaly"])
def test_nonphysical_or_ambiguous_anomaly_is_rejected(
    tmp_path: Path, quantity: str,
):
    config = _write_scalar_config(
        tmp_path / "map.ini",
        "netcdf",
        "unused.nc",
        "xVariable = x\nyVariable = y",
        "valueVariable = gravity",
        quantity=quantity,
    )

    with pytest.raises(
        NavConfigError, match="not a physical gravity correction"
    ):
        CustomGravityMap(FixedValue(), None, config)
