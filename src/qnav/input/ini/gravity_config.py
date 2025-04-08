
import qnav.input.ini.database_config as data_dirs

from functools import partial
from typing import Optional

from qnav.earth.geoid import GeoidModel
from qnav.gravity.base import GravityModel
from qnav.gravity.egm import GeoidCorrection
from qnav.gravity.geosat import Geosat44
from qnav.gravity.ggm_plus import GGMPlusAcc, GGMPlusDist
from qnav.gravity.marine import MarineGravity
from qnav.gravity.nima import WGS84Gravity
from qnav.gravity.simple import FixedValue, SimpleUniform
from qnav.gravity.somigliana import Somigliana
from qnav.gravity.srtm2gravity import SRTM2GravityFull, SRTM2GravityRes
from qnav.input.config_handler import ConfigHandler
from qnav.input.ini.geoid_config import get_estimation_geoid_model

__SECTION_ID = 'Gravity'

__DEFAULT_ALLOW_DOWNLOADS = True

__DEFAULT_GRAVITY_FUNC = 'somigliana'

__DEFAULT_GRAVITY_MAP = 'none'

__DEFAULT_TILE_MARGIN = 1


def get_true_gravity_model(config: ConfigHandler):

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    func_id: str = get_id("trueGravityFunction", __DEFAULT_GRAVITY_FUNC)
    map_id: str = get_id("trueGravityCorrectionMap", __DEFAULT_GRAVITY_MAP)

    # TODO: Include better error handling
    grav_func = _make_gravity_function(func_id)
    geoid_model = get_estimation_geoid_model(config)
    return _make_gravity_map(config, grav_func, geoid_model, map_id)


def get_estimated_gravity_model(config: ConfigHandler):

    # Shorthand function for reading float values from configuration
    get_id = partial(config.get_str_alpha, __SECTION_ID)
    func_id: str = get_id("estimatedGravityFunction", __DEFAULT_GRAVITY_FUNC)
    map_id: str = get_id("estimatedGravityCorrectionMap", __DEFAULT_GRAVITY_MAP)

    # TODO: Include better error handling
    grav_func = _make_gravity_function(func_id)
    geoid_model = get_estimation_geoid_model(config)
    return _make_gravity_map(config, grav_func, geoid_model, map_id)


def get_gradient_grid(config: ConfigHandler):

    get_float = partial(config.get_float, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    num_rings = get_int('gradGridNumRings', 100)
    num_angles = get_int('gradGridNumAngles', 50)
    max_radius = get_float('gradGridMaxRadius', 0.25)
    l_threshold = get_float('gradGridInnerMargin', 1.0)

    return {
        'num_rings': num_rings,
        'num_angles': num_angles,
        'max_radius': max_radius,
        'l_threshold': l_threshold,
    }


def _make_gravity_function(func_id: str) -> GravityModel:

    match (func_id.strip().casefold()):

        case 'fixed':
            return FixedValue()

        case 'uniform':
            return SimpleUniform()

        case 'wgs84':
            return WGS84Gravity()

        case 'somigliana':
            return Somigliana()

        case _:
            raise ValueError(f'unrecognised gravity function {func_id}')


def _make_gravity_map(config: ConfigHandler,
                      gravity_model: GravityModel,
                      geoid_model: Optional[GeoidModel],
                      map_id: str) -> GravityModel:

    # Create short-hand functions
    get_bool = partial(config.get_bool, __SECTION_ID)
    get_int = partial(config.get_int, __SECTION_ID)

    # Acquire other configuration variables related to gravity maps
    can_download = get_bool('autoDownload', __DEFAULT_ALLOW_DOWNLOADS)
    margin_size = get_int('mapTileMargin', __DEFAULT_TILE_MARGIN)

    # Return the requested fusion method:
    match (map_id.strip().casefold()):

        case 'none':
            return gravity_model

        case 'geoid':
            if geoid_model is None:
                raise ValueError(
                    'Geoid gravity correction cannot be applied '
                    'without geoid model. Check \'Geoid\' section.')
            return GeoidCorrection(gravity_model, geoid_model)

        case 'geosat':
            geosat_dir = data_dirs.get_geosat_gravity_dir(config)
            return Geosat44(gravity_model, geoid_model, geosat_dir, can_download)

        case 'marine':
            marine_dir = data_dirs.get_marine_gravity_dir(config)
            return MarineGravity(gravity_model, geoid_model, marine_dir, can_download)

        case 'ggmplusacc':
            ggm_plus_dir = data_dirs.get_ggm_plus_gravity_dir(config)
            return GGMPlusAcc(gravity_model, geoid_model, ggm_plus_dir,
                              can_download, margin_size)

        case 'ggmplusdist':
            ggm_plus_dir = data_dirs.get_ggm_plus_gravity_dir(config)
            return GGMPlusDist(gravity_model, geoid_model, ggm_plus_dir,
                               can_download, margin_size)

        case 'srtm2gravityfs':
            srtm2gravity_dir = data_dirs.get_srtm2gravity_dir(config)
            return SRTM2GravityFull(gravity_model, geoid_model, srtm2gravity_dir,
                                    can_download, margin_size)

        case 'srtm2gravityres':
            srtm2gravity_dir = data_dirs.get_srtm2gravity_dir(config)
            return SRTM2GravityRes(gravity_model, geoid_model, srtm2gravity_dir,
                                   can_download, margin_size)

        case _:
            raise ValueError(f'unrecognised gravity map {map_id}')



