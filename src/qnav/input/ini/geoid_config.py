from pathlib import Path
from typing import Optional
from functools import partial

from qnav.earth.geoid import GeoidPGM, GeoidModel
from qnav.input.config_handler import ConfigHandler
from qnav.input.config_handler import NavConfigError
from qnav.input.ini.database_config import get_geoid_dir

# The shared section name to use
__SECTION_ID: str = "Geoid"

# The default ID for the true geoid model
__DEFAULT_TRUE_GEOID: str = "none"

# The default ID for the estimate geoid model
__DEFAULT_EST_GEOID: str = "none"


def get_true_geoid_model(config: ConfigHandler) -> Optional[GeoidModel]:
    return _geoid_field_to_model(config, "trueGeoidModel")


def get_estimation_geoid_model(config: ConfigHandler) -> Optional[GeoidModel]:
    return _geoid_field_to_model(config, "estimatedGeoidModel")


def _geoid_field_to_model(config: ConfigHandler, field_name: str) -> Optional[GeoidModel]:
    get_str = partial(config.get_str, __SECTION_ID)
    geoid_name = get_str(field_name, __DEFAULT_TRUE_GEOID)
    geoid_dir = get_geoid_dir(config)

    return _get_geoid_model(geoid_name, geoid_dir, field_name)

def _get_geoid_model(model_name: str, geoid_dir: Path, field_name: str) -> Optional[GeoidModel]:

    if model_name is None or model_name == 'none':
        return None

    geoid_path = (geoid_dir / model_name).with_suffix('.pgm')
    if geoid_path.is_file():
        return GeoidPGM(geoid_path)

    raise NavConfigError(__SECTION_ID, field_name, "Unknown model",
           f"Unrecognized geoid model: {model_name}")
