from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qnav.gravity.custom import CustomGravityMap
from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini.gravity_config import (
    get_estimated_gravity_model,
    get_true_gravity_model,
)


def _write_custom_map(parent: Path, name: str, value: float) -> Path:
    map_dir = parent / name
    map_dir.mkdir()

    pd.DataFrame({
        "longitude": [10.0, 10.1, 10.0, 10.1],
        "latitude": [70.0, 70.0, 70.1, 70.1],
    }).to_csv(map_dir / "coordinates.csv", index=False)
    pd.DataFrame({
        "north": np.full(4, value),
        "east": np.full(4, value * 2),
        "down": np.full(4, value * 3),
    }).to_csv(map_dir / "residual.csv", index=False)

    config_file = map_dir / "map.ini"
    config_file.write_text(
        """
[CustomGravityMap]
name = Integration map
crs = EPSG:4326
mode = residual

[Grid]
format = csv
file = coordinates.csv
xColumn = longitude
yColumn = latitude
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
    return config_file


def _write_main_config(
    parent: Path,
    true_map: str,
    estimated_map: str,
) -> Path:
    config_file = parent / "settings.ini"
    config_file.write_text(
        f"""
[Database]
geoidDatabase = unused

[Geoid]
estimatedGeoidModel = none

[Gravity]
autoDownload = no
trueGravityFunction = fixed
trueGravityCorrectionMap = custom
trueGravityMapConfig = {true_map}
estimatedGravityFunction = fixed
estimatedGravityCorrectionMap = custom
estimatedGravityMapConfig = {estimated_map}
""",
        encoding="utf-8",
    )
    return config_file


def test_constructs_role_specific_custom_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    true_config = _write_custom_map(tmp_path, "true_map", 0.01)
    estimated_config = _write_custom_map(
        tmp_path, "estimated_map", 0.02)
    main_config = _write_main_config(
        tmp_path,
        str(true_config.relative_to(tmp_path)),
        str(estimated_config.relative_to(tmp_path)),
    )
    monkeypatch.chdir(tmp_path.parent)
    config = ConfigHandler(main_config)

    true_model = get_true_gravity_model(config)
    estimated_model = get_estimated_gravity_model(config)

    assert isinstance(true_model, CustomGravityMap)
    assert isinstance(estimated_model, CustomGravityMap)
    np.testing.assert_allclose(
        true_model.get_residual(70.05, 10.05),
        [0.01, 0.02, 0.03],
    )
    np.testing.assert_allclose(
        estimated_model.get_residual(70.05, 10.05),
        [0.02, 0.04, 0.06],
    )


def test_custom_map_requires_role_config(tmp_path: Path):
    main_config = tmp_path / "settings.ini"
    main_config.write_text(
        """
[Database]
geoidDatabase = unused

[Geoid]
estimatedGeoidModel = none

[Gravity]
trueGravityFunction = fixed
trueGravityCorrectionMap = custom
""",
        encoding="utf-8",
    )

    with pytest.raises(NavConfigError, match="trueGravityMapConfig"):
        get_true_gravity_model(ConfigHandler(main_config))
