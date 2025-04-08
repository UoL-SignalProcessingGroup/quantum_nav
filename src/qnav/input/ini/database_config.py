"""
===========
database.py
===========

:summary:
    Functions related to reading the database config section.
    A collection of functions used for reading content under the 'database'
    section within the configuration and returning the parsed data. This is
    commonly used for finding the URIs for database related files.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""


from qnav.input.config_handler import ConfigHandler
from pathlib import Path


# The shared section name to use
__SECTION_ID: str = "Database"


def get_dted_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for DTED data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the DTED database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured DTED database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "dtedDatabase")

def get_geoid_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for Geoid data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the Geoid database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured Geoid database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "geoidDatabase")

def get_gps_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for GPS data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the GPS RINEX file database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured GPS database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "gpsDatabase")

def get_geosat_gravity_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for Geosat-44 gravity data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the  Geosat-44 gravity database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured Geosat-44 gravity database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "geosatGravityDatabase")

def get_marine_gravity_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for Marine Gravity map data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the Marine Gravity map database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured Marine Gravity database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "marineGravityDatabase")

def get_ggm_plus_gravity_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for GGM Plus gravity data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the Global Gravity Map Plus database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured GGM Plus gravity database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "ggmPlusGravityDatabase")

def get_srtm2gravity_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for 'SRTM 2 Gravity' data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the SRTM2Gravity database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured SRTM2Gravity database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "srtm2GravityDatabase")

def get_irish_sea_gravity_dir(config: ConfigHandler) -> Path:
    """
    Returns the configured database_dir for bespoke Irish sea gravity data.
    Reads the database database_dir contents from the configuration file and
    returns the path for the Irish sea gravity database database_dir.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The configured Irish sea gravity database database_dir.
    :rtype: Path
    """
    return config.get_value_filepath(__SECTION_ID, "irishSeaGravityDatabase")
