"""
=========
simple.py
=========

:summary:
    Implementation of simple gravity modelling classes and functions.
    This class contains very simple gravity modelling functions that are
    namely used for testing, replacing gravity with values that are fixed
    or vary very little. Primarily used for cancelling out any impact of
    gravity during experiments.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of this module.
"""

import math
import numpy as np
import qnav.util.constants as const

from qnav.gravity.base import GravityModel


class FixedValue(GravityModel):
    """
    Sets gravity equal to fixed constant regardless of location.
    Using this model, gravity will be fixed and the same constant
    value will be returned.
    """

    # The value for vertical gravity acceleration to use.
    __G: float = const.G

    def __str__(self) -> str:
        return "Fixed Value Gravity"

    def calc_gravity_z(self, lat: float = None, lon: float = None, alt: float = None) -> float:
        return self.__G

    def calc_gravity_z_vec(self, lat: np.ndarray, lon: np.ndarray = None, alt: np.ndarray = None) -> np.ndarray:
        return np.full(lat.shape, self.__G)


class SimpleUniform(GravityModel):
    """
    Calculates gravity using uniformly across the Earth.
    Produces values using weighted average of the gravity between the poles
    and equator, meaning only latitude affected the calculations produced.
    Changes in longitude or altitude will have no impact.
    """

    # Normal gravity at poles.
    __G_P: float = const.G_P

    # Normal gravity at Equator.
    __G_E: float = const.G_E

    # Average gravity between Poles and Equator.
    __G_45 = 0.5 * (const.G_P + const.G_E)

    def __str__(self) -> str:
        return "Simple Uniform Gravity"

    def calc_gravity_z(self, lat: float, lon: float = None, alt: float= None) -> float:
        return (self.__G_45 - 0.5 * (self.__G_P - self.__G_E)
                * math.cos(2 * math.radians(lat)))

    def calc_gravity_z_vec(self, lat: np.ndarray, lon: np.ndarray = None, alt: np.ndarray = None) -> float:
        return (self.__G_45 - 0.5 * (self.__G_P - self.__G_E)
                * np.cos(2 * np.radians(lat)))


if __name__ == '__main__':
    fv = FixedValue()
    su = SimpleUniform()
