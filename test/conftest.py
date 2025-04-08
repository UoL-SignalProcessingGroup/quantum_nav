from pathlib import Path

import pytest
import numpy as np

from typing import Optional, override
from numpy.random import Generator, rand
from numpy.random import MT19937

from qnav.waypoints.trajectory import GroundTruth


class FakeRandom(Generator):
    """
    A fake random number generator used for testing.
    The purpose of this "Generator" is to override what would be randomly
    generated numbers with fixed pre-determined values. This is used to
    cancel out any internal random number generation during testing to
    make methods deterministic.
    """

    def __init__(self, seed: int = None):
        """
        Constructs a fake random number generator.
        All random values that will be produced will be the same fixed value.

        :param seed: Optionally used for verifying it is being set correctly.
        :type seed: int
        """
        self.__seed_used = seed if seed is not None else 1
        super().__init__(MT19937(seed))

    @property
    def seed_used(self) -> Optional[int]:
        """
        Returns the set value set (1 by default).

        :return: The seed value being used.
        :rtype: Optional[int]
        """
        return self.__seed_used

    @override
    def normal(self, loc=0.0, scale=1.0, size=1):
        """
        Overrides the drawing of random samples from a normal (Gaussian)
        distribution. Instead, all values will be that of the seed.

        :param loc: [Unused] Mean ("centre") of the distribution.
        :type loc: float or array_like of floats

        :param scale: [Unused] Standard deviation (spread or
            "width") of the distribution
        :type scale: float or array_like of floats

        :param size: The shape of the array to produce. By default,
            only a single value will be returned.
        :type size: int or tuple of ints, optional

        :return: An array of fixed values for given size.
        :rtype: float or array_like
        """
        # return np.ones(size) * self.__seed_used
        if size == 1 or np.shape(size) == (1,):
            return self.__seed_used
        return np.ones(size) * self.__seed_used

    @override
    def random(self, size=1, dtype=None, out=None):
        """
        Overrides the generation of random floats in the half-open interval
        [0.0, 1.0). Instead, all values will be that of the seed.

        :param size: The shape of the array to produce. By default,
            only a single value will be returned.
        :type size: int or tuple of ints, optional

        :param dtype: Desired dtype of the result, only `float64` and
            `float32` are supported. Byteorder must be native. The
             default value is np.float64.
        :type dtype: np.dtype

        :param out: Alternative output array in which to place the result.
        :type out: np.ndarray, optional

        :return: An array of fixed values for given size.
        :rtype: float or array_like
        """
        to_return = np.ones(size, dtype=dtype) * self.__seed_used

        if size == 1 or np.shape(size) == (1,):
            to_return = to_return[0]

        if out is not None:
            out[:] = to_return
        return to_return


@pytest.fixture
def mock_rng(monkeypatch):
    """
    A fixture for overriding 'np.random.default_rng' random number generators.
    Used for controlling internal random number generation for testing.

    :param monkeypatch: The pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    def mock_get(*args, **kwargs):
        return FakeRandom(*args, **kwargs)
    monkeypatch.setattr(np.random, "default_rng", mock_get)


@pytest.fixture(scope='function')
def random_truth() -> GroundTruth:
    return GroundTruth(
        timestamp=(rand() * 1000),
        position=(rand(3) * np.array([180, 360, 10000]) - np.array([90, 180, 0])),
        velocity=(rand(3) * np.array([200, 200, 20]) - np.array([100, 100, 10])),
        acceleration=(rand(3) * np.array([20, 20, 20]) - np.array([10, 10, 10])),
        attitude=(rand(3) * 180 - 90),
        angle_rates=(rand(3) * 10 - 5),
    )

@pytest.fixture(scope='function')
def empty_truth() -> GroundTruth:
    return GroundTruth(
        timestamp=0.0,
        position=np.zeros(3),
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        attitude=np.zeros(3),
        angle_rates=np.zeros(3),
    )

@pytest.fixture(scope='module')
def project_dir():
    return Path(__file__).parent.parent

# def test_mock_rng(mock_rng):
#     rng = np.random.default_rng()
#     print(rng.random(size=2))