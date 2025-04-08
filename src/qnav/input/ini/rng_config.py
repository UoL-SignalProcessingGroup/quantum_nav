"""
=============
rng_config.py
=============

:summary:
    Functions related to the controlling of random number generation.
    Provides a collection of functions used for reading content under the
    'random' section within the configuration file. These are mostly used
    for defining the initial random number generation seeds. This is
    required so that randomness within simulations can be repeated.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""


from qnav.input.config_handler import ConfigHandler
from numpy.random import default_rng
from numpy.random import Generator
from functools import cache
from random import randint


# The section ID this module is responsible for
__SECTION_ID: str = "Random"

# The minimum integer value to randomly generate
__MIN_SEED_VALUE: int = 0

# The maximum integer value to randomly generate
__MAX_SEED_VALUE: int = 2**31

# The default seed to use for randomness during initialisation
__DEFAULT_INIT_SEED: int = randint(__MIN_SEED_VALUE, __MAX_SEED_VALUE)

# The default seed to use for randomness during waypoint generation
__DEFAULT_WAYPOINT_SEED: int = randint(__MIN_SEED_VALUE, __MAX_SEED_VALUE)

# The default seed to use for randomness for error property generation
__DEFAULT_ERROR_SEED: int = randint(__MIN_SEED_VALUE, __MAX_SEED_VALUE)


def __get_seed(config: ConfigHandler,
               seed_field: str,
               auto_gen_field: str,
               default_seed) -> int:
    """
    A shorthand function for returning the value for given seed.
    On succession returns the corresponding seed value assigned for the given
    section. This is dependent if the configuration file's seed field gives a
    seed value, defines the seed is to be automatically generated or doesn't
    specify anything at all.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param seed_field: The field ID that can be holding the seed value.
    :type seed_field: str

    :param auto_gen_field: The field ID to mark if the seed is to be
        automatically generated instead (overrides any given value).
    :type auto_gen_field: str

    :param default_seed: The default seed value to use if it is to be
        automatically generated.
    :type default_seed: int

    :return: The corresponding random number generation seed value to use.
    :rtype: int
    """

    # If to autogenerate, return pre-generated seed:
    if config.get_bool(__SECTION_ID, auto_gen_field, False):
        return default_seed

    # Otherwise return either user given or default_seed:
    return config.get_int(__SECTION_ID, seed_field, 0)


@cache
def get_init_seed(config: ConfigHandler) -> int:
    """
    Returns the random number generation seed value for initialisation.
    This is the seed used to control randomness during the set-up stages
    of the simulation.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: RNG seed value for initialisation.
    """
    return __get_seed(config, "initialRandomSeed",
                      "autoGenerateInitialSeed",
                      __DEFAULT_INIT_SEED)

@cache
def get_waypoint_seed(config: ConfigHandler) -> int:
    """
    Returns the random number generation seed value for waypoint generation.
    This is the seed used to control randomness during the trajectory
    generation stages. Namely, when including noise and oscillations in the
    (ground truth) vehicle trajectory.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: RNG seed value for waypoint generation.
    """
    return __get_seed(config, "initialRandomSeed",
                      "autoGenerateInitialSeed",
                      __DEFAULT_INIT_SEED)

@cache
def get_error_seed(config: ConfigHandler) -> int:
    """
    Returns the random number generation seed value for measurement errors.
    This is the seed used to control randomness around measurement noise and
    other sensor related errors.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: RNG seed value for measurement errors and noise.
    """
    return __get_seed(config, "errorRandomSeed",
                      "autoGenerateErrorSeed",
                      __DEFAULT_ERROR_SEED)

def get_init_rng(config: ConfigHandler) -> Generator:
    """
    Returns initialised random number generator for simulation initialisation.
    Namely used for locally initialising random values, outside of objects,
    before the start of the simulation.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A random number generator.
    :rtype: Generator
    """
    return default_rng(get_init_seed(config))
