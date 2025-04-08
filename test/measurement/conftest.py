
import pytest
import numpy as np

from qnav.waypoints.trajectory import GroundTruth
from numpy.random import rand


def assert_array_equal(array_1: float | np.ndarray,
                       array_2: float | np.ndarray):
    np.testing.assert_allclose(array_1, array_2, rtol=1e-12)

