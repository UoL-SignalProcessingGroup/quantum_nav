"""
===========
altitude.py
===========

:summary:
    Supplied fusion method for altitude measurements.
    This module contains methods that can be used for acquiring altimeter
    measurements during simulation and fusing them with the current
    estimated altitude. Additional methods may be added in future updates.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of module.
"""


from qnav.estimation.state import EstimatedState
from qnav.measurement.altimeter import Altimeter
from qnav.measurement.sensor import FusionTrigger
from qnav.measurement.sensor import SensorFusion

import qnav.util.transformations as trans
import numpy as np


class FixedGainAltimeter(SensorFusion):
    """
    Fuses altimeter altitude measurements using fixed gain fusion.
    Updates the estimated altitude by using a weighted average of the
    measurement altitude and current estimated altitude. This gain
    amount can be set and updated at any time.
    """

    # Fixed properties used by the class
    __slots__ = '__altimeter', '__gain_amount'

    def __init__(self, altimeter: Altimeter, gain_amount: float = 0.5):
        """
        Creates fixed gain fusion for given altimeter and gain amounts.
        When initialised can be used to fuse successful altimeter measurements
        with the current estimated state.

        :param altimeter: The altimeter to acquire measurements from.
        :type altimeter: Altimeter

        :param gain_amount: The gain amount to use (between 0 and 1).
        :type gain_amount: float
        """
        super().__init__(FusionTrigger.ALL, altimeter)
        self.__altimeter = altimeter
        self.__gain_amount = gain_amount

    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Performs the fixed gained fusion with last altimeter measurement.
        Simply fuses the last altimeter measurement with previous measurement
        using a fixed-gain (weighted average) update.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState
        """

        # Unpack measurements and current estimates:
        measured_alt = self.__altimeter.last_measurement
        est_position = estimated_state.position

        # Skip fusion if measurement unsuccessful:
        if measured_alt is None:
            return

        # Update the current estimated position.
        est_position[2] = ((1 - self.__gain_amount) * est_position[2]
                     + self.__gain_amount * measured_alt)

        # Write update to the shared estimated state.
        estimated_state.update_estimates(position=est_position)

    @property
    def fixed_gain(self) -> float:
        """
        Property getter method for fixed gain amount.

        :return: The current gain amount for the altimeter measurements.
        :rtype: float
        """
        return self.__gain_amount

    @fixed_gain.setter
    def fixed_gain(self, gain: float):
        """
        Property setter method for fixed gain amount.

        :param gain: The new gain amount for the altimeter measurements.
        :type gain: float
        """
        if gain < 0 or gain > 1:
            raise ValueError('gain must be between 0 and 1')
        self.__gain_amount = gain


class AlphaBetaAltimeter(SensorFusion):
    """
    Fuses altimeter altitude measurements using alpha-beta filtering.
    Updates the estimated altitude by using an alpha-beta filter, where alpha
    relates to the expected accuracy of the altimeter and beta relates to the
    precision of the vertical velocity estimate. This method updates both
    estimated altitude and the vertical velocity.
    """

    # Fixed properties used by the class
    __slots__ = '__altimeter', '__alpha', '__beta'

    def __init__(self, altimeter: Altimeter, alpha: float = 0.9, beta: float = None):
        """
        Creates alpha-beta fusion for given altimeter
        When initialised can be used to fuse successful altimeter measurements
        with the current estimated state using alpha-beta fusion. This updates
        both the estimated altitude and vertical velocity.

        :param alpha: The value to use for alpha (between 0 and 1). This
            corresponds to the precision of the altimeter.
        :type alpha: float

        :param beta: The value to use for beta (between 0 and 1). This
            corresponds to the accuracy of the vertical velocity.
        :type beta: float

        :param altimeter: The altimeter to acquire measurements from.
        :type altimeter: Altimeter
        """
        super().__init__(FusionTrigger.ALL, altimeter)
        self.__altimeter = altimeter
        self.__alpha = alpha
        self.__beta = (alpha * alpha) / 2 if beta is None else beta

    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Performs alpha-beta fusion with latest altimeter measurement.
        Updates both the estimated altitude and the vertical velocity using
        the latest measurement from the altimeter. From this both the
        vertical position and vertical velocity are updated.

        :param estimated_state: The current estimated state.
        :type estimated_state: EstimatedState
        """

        alt = self.__altimeter.last_measurement
        if alt is None:
            return

        # Convert position to local NED co-ordinates
        position_e0 = estimated_state.position
        # position_e0 = trans.lla2ned(ref_lla, ref_lla)

        # Obtain the rotation from Earth to body axis
        attitude_0 = np.radians(estimated_state.attitude)
        rot_earth2body = trans.rotate_3d(*attitude_0)

        # Convert velocity from body axes to Earth Axes
        velocity_b0 = estimated_state.velocity
        velocity_e0 = np.linalg.solve(rot_earth2body, velocity_b0)

        time_step = self.__altimeter.time_step

        alpha = self.__alpha
        beta = self.__beta  # (alpha * alpha) / 2

        z = position_e0[2]
        v_z = velocity_e0[2]

        z_prime = (1 - alpha) * z + alpha * alt
        v_z_prime = v_z - (beta / time_step) * (alt - z)

        position_e1 = np.array([position_e0[0], position_e0[1], z_prime])
        velocity_e1 = np.array([velocity_e0[0], velocity_e0[1], v_z_prime])
        velocity_b1 = rot_earth2body.dot(velocity_e1)

        estimated_state.update_estimates(
            position=position_e1,
            velocity=velocity_b1
        )

    @property
    def alpha(self) -> float:
        """
        Property getter method for the alpha value.

        :return: The current alpha value to use for filtering.
        :rtype: float
        """
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha_value: float):
        """
        Property setter method for the alpha value.

        :param alpha_value: The updated alpha value to use for filtering.
        :type alpha_value: float
        """
        if alpha_value < 0 or alpha_value > 1:
            raise ValueError('Alpha must be between 0 and 1')
        self.__alpha = alpha_value

    @property
    def beta(self) -> float:
        """
        Property getter method for the beta value.

        :return: The current beta value to use for filtering.
        :rtype: float
        """
        return self.__beta

    @beta.setter
    def beta(self, beta_value: float):
        """
        Property setter method for the beta value.

        :param beta_value: The updated beta value to use for filtering.
        :type beta_value: float
        """
        if beta_value < 0 or beta_value > 1:
            raise ValueError('Beta must be between 0 and 1')
        self.__beta = beta_value

