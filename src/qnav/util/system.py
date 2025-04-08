"""
=========
common.py
=========

:summary:
    Provides commonly used project functions.
    This module contains commonly used (non-calculation) functions that are
    called throughout the project. These are namely for providing information
    about the project and the current platform.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module
"""


import platform

from qnav import __version__ as toolbox_version


def get_toolbox_version() -> str:
    """
    A simple function to return the current version number of the toolbox.
    This is useful for reporting information that may dependent on the build
    of the software, such as generated results or reported errors.

    :return: The decimal version number for the toolbox.
    :rtype: str
    """
    return toolbox_version


def get_system_name(return_min: bool = False) -> str:
    """
    Return a short string for identifying the host operating system. This is
    namely used for identifying the operating system in a human-readable
    format. This is useful for identifying the type of system used when
    running an experiment.

    :param return_min: If set to true, only the operating system name
        and major version number will be returned.
    :type return_min: bool

    :return: A short string used for identifying the host operating system.
    :rtype: str
    """

    # Obtain the operating system name.
    os_name = platform.platform(aliased=True, terse=return_min)

    # Provide a default value if nothing is returned.
    if os_name == "":
        os_name = "UNKNOWN"

    # Return the parts of the name.
    return os_name.replace("-", " ")


def get_platform_details() -> str:
    """
    A simple function that returns a description of the platform.
    This function returns a string including the python version, processor
    platform architecture of the host system.

    :return: A string describing the platform.
    :rtype: str
    """
    proc = platform.processor()
    arc, _ = platform.architecture()
    py_ver = platform.python_version()
    return f"{py_ver} {proc} {arc}"


def is_window_platform() -> bool:
    """
    A simple function which will return true if the system's operating system
    is Microsoft Windows based. Otherwise, this function will return false.

    :return: Returns true if the host operating system is Windows based.
    :rtype: bool
    """
    sys_name = platform.system().casefold()
    return "windows" in sys_name or "win32" in sys_name


def is_mac_platform() -> bool:
    """
    A simple function which will return true if the system's operating system
    is Mac-OS based. Otherwise, this function will return false.

    :return: Returns true if the host operating system is Mac-OS based.
    :rtype: bool
    """
    return "darwin" in platform.system().casefold()


def is_linux_platform() -> bool:
    """
    A simple function which will return true if the system's operating system
    is Linux/Unix based. Otherwise, this function will return false.

    :return: Returns true if the host operating system is Linux based.
    :rtype: bool
    """
    return "linux" in platform.system().casefold()

