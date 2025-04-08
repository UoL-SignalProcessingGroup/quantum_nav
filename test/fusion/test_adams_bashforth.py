
import pytest

from fusion.test_ins import TestSimpleNumerical
from qnav.fusion.ins import AdamsBashforth
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope


class TestAdamsBashforth(TestSimpleNumerical):

    @pytest.fixture
    def perfect_ins(self, frequency: float):
        """
        Returns a perfect clean INS instance.
        """
        error_profile = ErrorProperties()
        accelerometer = Accelerometer(frequency, error_profile)
        gyroscope = Gyroscope(frequency, error_profile)
        return AdamsBashforth(accelerometer, gyroscope)
