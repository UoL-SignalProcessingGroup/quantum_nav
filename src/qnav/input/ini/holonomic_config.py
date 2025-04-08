"""
===================
holonomic_config.py
===================

:summary:
    Functions related to initialising holonomic constraints during simulations.
    A collection of functions used for reading content under the 'holonomics'
    section within the configuration and initialising the corresponding
    objects. Specifically, this involves initialising the holonomic
    constraints to apply for specific values.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""

from functools import partial
from typing import Optional

from qnav.input.config_handler import ConfigHandler
from qnav.input.config_handler import NavConfigError
from qnav.measurement.sensor import FusionTrigger
from qnav.simulation.holonomic import HolonomicConstraint
from qnav.simulation.holonomic import HolonomicFusion

import numpy as np


# The shared section name to use:
__SECTION_ID: str = "Holonomics"

# The default for if holonomics are to be used:
__DEFAULT_IS_USED: bool = False

# The default update frequency:
__DEFAULT_FREQ: float = 1

# The default properties to expect:
__DEFAULT_PROPERTIES = [
    "position",
    "velocity",
    "acceleration",
    "attitude",
    "angleRate"
]


def _get_frequency(config: ConfigHandler) -> float:
    """
    Returns the fixed frequency that holonomic constraints are applied at.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: The update frequency (in Hz).
    :rtype: float
    """
    get_float = partial(config.get_float, __SECTION_ID)
    return get_float("correctionFreq", __DEFAULT_FREQ)


def _get_holonomic_constraint(config: ConfigHandler, constraint_id: str) -> Optional[HolonomicConstraint]:
    """
    Generates a holonomic constraint instance for given ID from configuration.
    A helper function that initialises and returns a holonomic constraint
    instance for given property ID, using the values from the configuration.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :param constraint_id: The ID or name of the constraint property. For
        example "position", "velocity", "attitude" etc.
    :type constraint_id: str

    :return: Either an initialised holonomic constraint for given property,
        or None if the holonomic constraint has not been requested.
    :rtype: HolonomicConstraint | None
    """

    get_float = partial(config.get_float, __SECTION_ID)
    get_csv = partial(config.get_csv_numeric, __SECTION_ID)

    gain_field = f"{constraint_id}Gain"
    axis_field = f"{constraint_id}Axis"

    freq = _get_frequency(config)
    gain = get_float(gain_field, 0)
    axis = get_csv(axis_field, np.ones(3))

    if gain == 0:
        return None

    if len(axis) == 1:
        axis = np.ones(3) * axis

    if np.shape(axis) != (3,):
        raise NavConfigError(
            __SECTION_ID, axis_field,"Invalid Size",
            "must contain 1 or 3 values")

    return HolonomicConstraint(freq, constraint_id, gain, axis)


def get_constraints(config: ConfigHandler) -> list[HolonomicConstraint]:
    """
    Obtains all holonomic constraints instances for all typical properties.
    This initialises and returns holonomic constraints instances for the
    position, velocity, acceleration, attitude and angle rate properties,
    subject to their preferences in the user configuration file. Those
    that are not enabled will not be returned.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: A list containing all configured holonomic instances. This can
        be empty if no holonomic constraints are requested/configured.
    :rtype: list[HolonomicConstraint]
    """

    # Get the holonomic constraint instance for each property
    constraints = [_get_holonomic_constraint(config, cid)
                   for cid in __DEFAULT_PROPERTIES]

    # Return all non-None objects in the gathered list
    return [c for c in constraints if c is not None]


def get_holonomic_sf(config: ConfigHandler) -> Optional[HolonomicFusion]:
    """
    Generates and returns a configured holonomic fusion instance.
    Produces a complete holonomic fusion instance, subject to the
    configuration file. If disabled, None will be returned instead.

    :param config: The config handler instance to use for parameter values.
    :type config: ConfigHandler

    :return: Either a configured holonomic fusion instance, or None.
    :rtype: HolonomicFusion | None
    """

    get_bool = partial(config.get_bool, __SECTION_ID)
    if not get_bool("useHolonomics", __DEFAULT_IS_USED):
        return None

    constraints = get_constraints(config)
    if len(constraints) == 0:
        return None

    return HolonomicFusion(FusionTrigger.ALL, *constraints)
