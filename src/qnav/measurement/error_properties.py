"""
===================
error_properties.py
===================

:summary:
    Contains representations for sensor error profiles.
    Module contains classes that are used for profiling the common error
    types represented by various sensor types. Namely used to reduce
    repeated common class arguments and methods.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""

import numpy as np
import numpy.typing as npt


class ErrorProperties:
    """
    Defines common error properties used for 3D sensors.
    This data class contains the errors associated with 3D sensors;
    specifically, the bias error, bias drift rate, scaling errors,
    non-orthogonality and measurement noise properties. These are commonly
    used when defining sensors such as accelerometer and gyroscope, along
    with their quantum counterparts.
    """

    def __init__(self,
                 bias_error: float | npt.ArrayLike = 0,
                 bias_drift_rate: float | npt.ArrayLike = 0,
                 scale_error: float | npt.ArrayLike = 0,
                 non_orth_error: float | npt.ArrayLike = 0,
                 avg_meas_noise: float | npt.ArrayLike = 0):
        """
        Defines the error properties used for a 3D sensor.
        On creation, validates given error values and returns a data class
        with read-only properties corresponding to the error profile given.

        :param bias_error: The initial bias errors (in micro-g).
        :type bias_error: scalar or array-like (3-elements)

        :param bias_drift_rate: The bias drift rate (in micro-g/root sec).
        :type bias_drift_rate: scalar or array-like (3-elements)

        :param scale_error: The scaling errors (in ppm).
        :type scale_error: scalar or array-like (3-elements)

        :param non_orth_error: The non-orthogonal errors (in micro-g).
        :type non_orth_error: scalar or array-like (6-elements)

        :param avg_meas_noise: The average measurement error (in
            micro-g.root Hz).
        :type avg_meas_noise: scalar or array-like (3-elements)`
        """

        # TODO: FIX UNITS FOR GYROSCOPES!
        # TODO: Replace validation with pydantic!
        # TODO: Implement as dataclass

        __req_size_3 = ((1,), (3,))
        __req_size_6 = ((1,), (6,))

        # if is_invalid(bias_error, 3):
        if not (np.isscalar(bias_error) or (
                np.shape(bias_error) in __req_size_3)):
            raise ValueError("bias_error must have 1 or 3 elements")

        # if is_invalid(bias_drift_rate, 3):
        if not (np.isscalar(bias_drift_rate) or (
                np.shape(bias_drift_rate) in __req_size_3)):
            raise ValueError("bias_drift_rate must have 1 or 3 elements")

        # if is_invalid(scale_error, 3):
        if not (np.isscalar(scale_error) or (
                np.shape(scale_error) in __req_size_3)):
            raise ValueError("scale_error must have 1 or 3 elements")

        # if is_invalid(non_orth_error, 6):
        if not (np.isscalar(non_orth_error) or (
                np.shape(non_orth_error)in __req_size_6)):
            raise ValueError("non_ort_error must have 1 or 6 elements")

        # if is_invalid(avg_meas_noise, 3):
        if not (np.isscalar(avg_meas_noise) or (
                np.shape(avg_meas_noise) in __req_size_3)):
            raise ValueError("avg_meas_noise must have 1 or 3 elements")

        # Ensure all variables are in the form of numpy arrays:
        self.__bias_error = bias_error * np.ones(3)
        self.__bias_drift_rate = bias_drift_rate * np.ones(3)
        self.__scale_error = scale_error * np.ones(3)
        self.__non_orth_error = non_orth_error * np.ones(6)
        self.__avg_meas_noise = avg_meas_noise * np.ones(3)

        # Initialise the error matrix
        di = np.eye(3, dtype=bool)
        self.__error_matrix = np.zeros((3, 3))
        self.__error_matrix[di] = self.__scale_error
        self.__error_matrix[~di] = self.__non_orth_error

        # Correct error matrix units
        self.__error_matrix *= 1.0e-6
        self.__error_matrix += np.eye(3)

    @property
    def bias_error(self) -> np.ndarray:
        """
        The bias errors for each axis (in micro-g).

        :return: Sensor bias errors (in micro-g).
        :rtype: np.ndarray (3-elements)
        """
        return self.__bias_error

    @property
    def bias_drift_rate(self) -> np.ndarray:
        """
        The bias drift rate for each axis (in micro-g/root sec).

        :return: Bias drift rate (in micro-g/root sec).
        :rtype: np.ndarray (3-elements)
        """
        return self.__bias_drift_rate

    @property
    def scale_error(self) -> np.ndarray:
        """
        The scaling errors for each axis (in ppm).

        :return: Scaling errors (in ppm).
        :rtype: np.ndarray (3-elements)
        """
        return self.__scale_error

    @property
    def non_orth_error(self) -> np.ndarray:
        """
        The non-orthogonal "cross-coupling" errors (in micro-g).
        Specifically the alignment errors for axis xy, xz, yx, yz,
        zx and zy respectively.

        :return: non-orthogonal errors (in micro-g).
        :rtype: np.ndarray (6-elements)
        """
        return self.__non_orth_error

    @property
    def avg_meas_noise(self) -> np.ndarray:
        """
        The average measurement error for each axis (in micro-g.root Hz).

        :return: average measurement error (in micro-g.root Hz).
        :rtype: np.ndarray (3-elements)
        """
        return self.__avg_meas_noise

    @property
    def error_matrix(self) -> np.ndarray:
        """
        A matrix containing the scale factor and cross-coupling errors for a
        nominally orthogonal sensor. This is a commonly used representation.
        For scale errors S and non-orthogonal errors M, this is formatted as:
        [[ Sx, Mxy, Mxz ],
         [ Myx, Sy, Myz ],
         [ Mzx, Mzy, Sz ]]

        :return: Error matrix for scale factor and cross-coupling errors.
        :rtype: np.ndarray (3-by-3 elements)
        """
        return self.__error_matrix



