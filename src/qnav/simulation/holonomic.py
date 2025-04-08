"""
============
holonomic.py
============

:summary:
    Supplies means of applying configured holonomic constraints.
    Virtual simulations are vulnerable to numerical instabilities, causing
    unintended drift between the ground truth and estimated state over time.
    Unlike with artificial noise that is purposely added through configured
    sensor error properties, this is undesirable and can cause unintended
    behavior. To combat this, this holonomic module can be used to apply
    a variety of constraints to restrict the estimated state of the system
    to values within the expected behaviour. In other words, this is used
    to slightly nudge the estimated state toward that of the ground truth
    at a scale to overcome numerical instabilities (such as rounding errors)
    but not to affect simulated errors.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation.
"""

import numpy as np

from qnav.measurement.sensor import Sensor
from qnav.measurement.sensor import SensorFusion
from qnav.measurement.sensor import FusionTrigger
from qnav.waypoints.trajectory import GroundTruth
from qnav.estimation.state import EstimatedState


class HolonomicConstraint(Sensor):
    """
    Obtains holonomic correction values during simulation.
    For a given ground truth record, gently shifts the estimated state towards
    the ground truth state at a rate to overcome unintended drift, but not
    affect simulation accuracy (i.e. intended behaviour from
    artificial/configured errors).
    """

    # TODO: Why are fixed gain and axis filter separate variables?
    # TODO: Does not support non-inertia measurements
    def __init__(self, frequency: float, gt_filter: str, fixed_gain: float,
                 axis_filter: np.ndarray = None):
        """
        Initialises a holonomic constraint to apply for given ground truth.
        Requires an update frequency, ground truth record filter and fixed
        gain amount (along with optional record axis to apply). On succession
        creates a sensor wrapper that applies the holonomic constraint at
        fixed frequency for the specified ground truth record during
        simulation.

        :param frequency: The frequency to apply the correction (in Hz).
        :type frequency: float

        :param fixed_gain: The percentage gain (between 0.0 and 1.0)
            of correction to apply.
        :type fixed_gain: float

        :param gt_filter: The ground truth record name to apply the correction
            for. This is case-sensitive and must match the property name of
            the record in the ground truth and estimated state.
        :type gt_filter: str

        :param axis_filter: (Optional) The filter percentages (or toggle
            flags) for each axis of the ground truth record. Can also be
            used to disable corrections for certain axis.
        :type axis_filter: np.ndarray[float]
        """
        super().__init__(frequency)
        self._gt_filter = gt_filter
        self._fixed_gain = fixed_gain

        if axis_filter is not None:
            self._fixed_gain *= axis_filter
            if axis_filter.shape != (3,):
                raise ValueError('axis_filter must be of shape (3,)')

        self._last_measurement = None

    @property
    def filter_id(self) -> str:
        """
        Displays the tag used for filtering the ground truth.
        Simply used for checking filter doesn't already exist.

        :return: The tag ID for the ground truth record to filter by.
        :rtype: str
        """
        return self._gt_filter

    @property
    def filter_gain(self) -> float | np.ndarray:
        """
        Displays the gain amount (percentage) being applied.
        Simply used for checking the filter's value.

        :return: The gain amount to apply to each axis
        :rtype: float | np.ndarray
        """
        return self._fixed_gain

    def take_measurement(self, est_time: float, ground_truth: GroundTruth) :
        """
        Extracts data from ground truth to obtain correction amount.
        When called during simulation loop, this checks the relevant data can
        be extracted from the ground truth and uses it to update the
        correction value to apply at that time.

        :param est_time: The estimate time of the measurement (not used).
        :type est_time: float

        :param ground_truth: The ground truth record at the given time.
        :type ground_truth: GroundTruth
        """
        if hasattr(ground_truth, self._gt_filter):
            value = getattr(ground_truth, self._gt_filter)
            # self._last_measurement = value * self._fixed_gain
            self._last_measurement = value

    def clear_measurement(self):
        """
        Clears the last measurement value obtained.
        Useful in preventing the accidental reuse of measurements when using
        multiple holonomic constraints asynchronously.
        """
        self._last_measurement = None

    @property
    def last_measurement(self) -> any:
        """
        Returns the last measurement captured (i.e. correction to apply).
        Upon error this will return None.

        :return: The holonomic correction to apply.
        :rtype: float | np.ndarray
        """
        return self._last_measurement


class HolonomicFusion(SensorFusion):
    """
    Applies the holonomic constraint corrections during simulation.
    Manages a correction of holonomic constraints and applying them
    to correct the current estimated state.
    """

    def __init__(self, trigger: FusionTrigger, *holonomics: HolonomicConstraint):
        """
        Defines holonomic fusion to apply with list of holonomic constraints.

        :param trigger: The trigger mode for deciding when to apply fusion.
        :type trigger: FusionTrigger

        :param sensors: A series of configured holonomic constraints to use.
        :type sensors: Iterable[HolonomicConstraint]
        """
        super().__init__(trigger, *holonomics)
        self._holonomics = holonomics

        all_ids = set()
        for holonomic in self._holonomics:

            # Check that the holonomic instances are valid:
            if not isinstance(holonomic, HolonomicConstraint):
                raise ValueError("sensor type must be HolonomicConstraint")

            # Check that the constraint isn't already covered:
            filter_id = holonomic.filter_id
            if filter_id in all_ids:
                raise ValueError(f"Duplicate filter ID: {filter_id}")
            all_ids.add(filter_id)

    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Apply the holonomic constraints and corrections during simulation.
        When called, performs fusion by overwriting current estimates.

        :param estimated_state: The current estimated states to update.
        :type estimated_state: EstimatedState
        """

        updated_states = {}

        # Obtain a dictionary of all updates
        for holonomic in self._holonomics:

            # Get the current property ID
            key = holonomic.filter_id

            # Get the estimated and true values
            est_value = getattr(estimated_state, key, None)
            true_value = holonomic.last_measurement
            gain = holonomic.filter_gain

            # Stop if either of them are none
            if est_value is None or true_value is None:
                continue

            updated_states[key] = est_value + gain * (
                    true_value - est_value)

        # Apply updates by overwriting current estimates
        estimated_state.update_estimates(**updated_states)


# if __name__ == '__main__':
#     tmp = np.array([1,2,3])
#     gt = GroundTruth(1, tmp, tmp, tmp, tmp, tmp)




