
import pytest
import numpy as np

from qnav.waypoints.trajectory import GroundTruth
from numpy.random import rand


@pytest.fixture
def random_truth() -> GroundTruth:
    """
    Returns a randomly generated ground truth.
    """
    return GroundTruth(
        timestamp=(rand() * 1000),
        position=(rand(3) * np.array([180, 360, 10000]) - np.array([90, 180, 0])),
        velocity=(rand(3) * np.array([200, 200, 20]) - np.array([100, 100, 10])),
        acceleration=(rand(3) * np.array([20, 20, 20]) - np.array([10, 10, 10])),
        attitude=(rand(3) * 180 - 90),
        angle_rates=(rand(3) * 10 - 5),
    )


@pytest.fixture
def another_truth():
    """
    Returns another random ground truth.
    """
    return GroundTruth(
        timestamp=(rand() * 1000),
        position=(rand(3) * np.array([180, 360, 10000]) - np.array([90, 180, 0])),
        velocity=(rand(3) * np.array([200, 200, 20]) - np.array([100, 100, 10])),
        acceleration=(rand(3) * np.array([20, 20, 20]) - np.array([10, 10, 10])),
        attitude=(rand(3) * 180 - 90),
        angle_rates=(rand(3) * 10 - 5),
    )