"""
===========
platform.py
===========

:summary:
    Provides conversion to and from sensor reference frame.
    This module is responsible for representing the orientation and location
    of sensor's on a platform body, providing means for common conversion
    between the reference frames.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import qnav.util.transformations as trans
import numpy.typing as npt
import numpy as np


class SensorAxis:
    """
    Specifies the orientation and offset of sensors relative to platform.
    For a corresponding sensor, this class specifies the angle orientation
    and offset location of the sensor on the platform body. This supplies
    a consistent practice for local conversion between sensor and
    platform reference frames.
    """

    # Fixed class properties
    __slots__ = [
        '__sensor_angles_deg',
        '__sensor_angles_rad',
        '__lever_arm_xyz',
        '__body2sensor_mat'
    ]

    def __init__(self,
                 sensor_angles: npt.ArrayLike = np.zeros(3),
                 lever_arm: npt.ArrayLike = np.zeros(3)):
        """
        Define sensor orientation and location relative to platform.
        Provided with the local orientations and offset location of a sensor
        relative to platform body, this initialises an instance responsible
        for handling conversions to and from sensor and platform body.

        :param sensor_angles: The yaw, pitch and roll angle orientations
            (in degrees) of the sensor relative to the platform body.
        :type sensor_angles: npt.ArrayLike

        :param lever_arm: The x, y and z position offsets (in metres)
            of the sensor relative to the platform body.
        :type sensor_angles: npt.ArrayLike
        """

        # Record the sensor angles and lever arm offset
        self.__sensor_angles_deg = np.array(sensor_angles)
        self.__lever_arm_xyz = np.array(lever_arm)

        # Ensure the sensor angles are valid:
        if self.sensor_angles_deg.shape != (3,) or not np.isfinite(self.sensor_angles_deg).all():
            raise ValueError('Sensor angles must be a valid 3-element array')

        # Ensure the lever position is valid:
        if self.lever_arm.shape != (3,) or not np.isfinite(self.lever_arm).all():
            raise ValueError('Lever arm offset must be a valid 3-element array')

        # Calculate the transformation matrix for body to sensor.
        self.__sensor_angles_rad = np.radians(self.sensor_angles_deg)
        self.__body2sensor_mat = trans.rotate_3d(*self.sensor_angles_rad)

    @property
    def sensor_angles_deg(self) -> np.ndarray:
        """
        Returns the sensor orientation angles in degrees.

        :return: Sensor angles (in degrees).
        :rtype: np.ndarray (3-elements)
        """
        return self.__sensor_angles_deg

    @property
    def sensor_angles_rad(self) -> np.ndarray:
        """
        Returns the sensor orientation angles in radians.

        :return: Sensor angles (in radians).
        :rtype: np.ndarray (3-elements)
        """
        return self.__sensor_angles_rad

    @property
    def lever_arm(self) -> np.ndarray:
        """
        The lever arm position offset in metres.

        :return: Lever arm offset (in metres).
        :rtype: np.ndarray (3-elements)
        """
        return self.__lever_arm_xyz

    @property
    def body2sensor_mat(self) -> np.ndarray:
        """
        The rotation matrix for body to sensor conversion.

        :return: The rotation matrix for reference frame conversion.
        :rtype: np.ndarray (3-by-3 elements)
        """
        return self.__body2sensor_mat

    def __str__(self) -> str:
        """
        Prints a descriptive string of the orientation and offset.
        Used to cleanly display the sensor angles and lever arm values.

        :return: A string stating the orientation angles and offset position.
        :rtype: str
        """
        x, y, z = self.__lever_arm_xyz
        p, q, r = self.__sensor_angles_deg
        return (f"Angles: (p:{p} q:{q} r:{r}) deg\t"
                f"Offset: (x:{x} y:{y} z:{z}) m")


if __name__ == '__main__':
    sensor = [0, 0, 0]
    lever = np.array([0, 1, 0])
    sa = SensorAxis(sensor, lever)
