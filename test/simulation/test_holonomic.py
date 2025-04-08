
import pytest
import numpy as np

from qnav.estimation.state import EstimatedState
from qnav.measurement.sensor import FusionTrigger
from qnav.simulation.holonomic import HolonomicConstraint, HolonomicFusion

_GT_FILTERS = ['position', 'velocity', 'acceleration',
               'attitude', 'angle_rates']


class TestHolonomics:

    @staticmethod
    @pytest.fixture(params=(0.1, 10.0, 100.0))
    def frequency(request):
        """
        The frequency for applying the holonomic constraint (in Hz).
        """
        return request.param

    @staticmethod
    @pytest.fixture(params=_GT_FILTERS)
    def filter_name(request):
        """
        The ground truth property filter name.
        """
        return request.param

    @staticmethod
    @pytest.fixture(params=(0.0, 0.1, 0.25, 1.0))
    def filter_gain(request):
        """
        The gain amount to use.
        """
        return request.param

    @staticmethod
    @pytest.fixture
    def random_axis() -> np.ndarray:
        """
        Example random axis filter to use.
        """
        return np.random.rand(3)

    def test_update(self, frequency):
        dt = 1 / frequency
        hc = HolonomicConstraint(frequency, _GT_FILTERS[0], 1.0)
        assert hc.next_update == dt
        hc.update()
        assert hc.next_update == 2 * dt

    def test_frequency(self, frequency):
        hc = HolonomicConstraint(frequency, _GT_FILTERS[0], 1.0)
        assert hc.frequency == frequency

    def test_filter_id(self, filter_name):
        hc = HolonomicConstraint(1.0, filter_name, 1.0)
        assert hc.filter_id == filter_name

    def test_fixed_gain(self, filter_gain):
        hc = HolonomicConstraint(1.0, _GT_FILTERS[0], filter_gain)
        assert hc.filter_gain == filter_gain

    def test_axis_gain(self, filter_gain, random_axis):
        hc = HolonomicConstraint(1.0, _GT_FILTERS[0], filter_gain, random_axis)
        expected = filter_gain * random_axis
        np.testing.assert_array_equal(hc.filter_gain, expected)

    def test_valid_measurement(self, random_truth, filter_name):
        hc = HolonomicConstraint(1.0, filter_name, 0.01)
        assert hc.last_measurement is None
        hc.take_measurement(0, random_truth)
        expected = getattr(random_truth, filter_name)
        np.testing.assert_array_equal(hc.last_measurement, expected)

    def test_invalid_measurement(self, random_truth):
        hc = HolonomicConstraint(1.0, "unknown", 0.01)
        assert hc.last_measurement is None
        hc.take_measurement(0, random_truth)
        assert hc.last_measurement is None

    def test_valid_fusion(self, frequency, filter_name, filter_gain, random_truth, empty_truth):

        hc = HolonomicConstraint(frequency, filter_name, filter_gain)
        hf = HolonomicFusion(FusionTrigger.ALL, hc)
        estimated_state = EstimatedState(empty_truth)
        original_estimate = estimated_state.clone()

        hc.take_measurement(0, random_truth)
        hf.perform_fusion(estimated_state)
        gain = hc.filter_gain

        for filter_id in _GT_FILTERS:

            old_value = getattr(original_estimate, filter_id)
            new_value = getattr(estimated_state, filter_id)
            gt_value = getattr(random_truth, filter_id)

            if filter_id == filter_name:
                expected = old_value + gain * (gt_value - old_value)
                np.testing.assert_array_equal(expected, new_value)
            else:
                np.testing.assert_array_equal(old_value, new_value)

    def test_invalid_fusion(self, frequency, filter_gain, random_truth, empty_truth):

        hc = HolonomicConstraint(frequency, 'invalid', filter_gain)
        hf = HolonomicFusion(FusionTrigger.ALL, hc)
        estimated_state = EstimatedState(empty_truth)
        original_estimate = estimated_state.clone()

        hc.take_measurement(0, random_truth)
        hf.perform_fusion(estimated_state)

        for filter_id in _GT_FILTERS:
            old_value = getattr(original_estimate, filter_id)
            new_value = getattr(estimated_state, filter_id)
            np.testing.assert_array_equal(old_value, new_value)

    def test_axis_fusion(self, frequency, filter_name, filter_gain,
                         random_axis, random_truth, empty_truth):

        hc = HolonomicConstraint(frequency, filter_name, filter_gain, random_axis)
        hf = HolonomicFusion(FusionTrigger.ALL, hc)
        estimated_state = EstimatedState(empty_truth)
        original_estimate = estimated_state.clone()

        hc.take_measurement(0, random_truth)
        hf.perform_fusion(estimated_state)
        gain = hc.filter_gain

        for filter_id in _GT_FILTERS:

            old_value = getattr(original_estimate, filter_id)
            new_value = getattr(estimated_state, filter_id)
            gt_value = getattr(random_truth, filter_id)

            if filter_id == filter_name:
                expected = old_value + gain * (gt_value - old_value)
                np.testing.assert_array_equal(expected, new_value)
            else:
                np.testing.assert_array_equal(old_value, new_value)


    @pytest.mark.parametrize('test_pos', (True, False))
    @pytest.mark.parametrize('test_vec', (True, False))
    @pytest.mark.parametrize('test_acc', (True, False))
    @pytest.mark.parametrize('test_att', (True, False))
    @pytest.mark.parametrize('test_ang', (True, False))
    def test_multi_fusion(self, frequency, filter_gain, random_truth, empty_truth,
                          test_pos: bool, test_vec: bool, test_acc: bool,
                          test_att: bool, test_ang: bool):

        # Create a constrain for each of the supported properties
        is_enabled = [test_pos, test_vec, test_acc, test_att, test_ang]
        constraints = [HolonomicConstraint(frequency, name, filter_gain)
                       for name, to_use in zip(_GT_FILTERS, is_enabled) if to_use]

        # Skip cases where no constraints are used:
        if len(constraints) == 0:
            return

        # Create the state and make a clone copy.
        estimated_state = EstimatedState(empty_truth)
        original_estimate = estimated_state.clone()

        # Apply fusion for each of the enabled constraints
        hf = HolonomicFusion(FusionTrigger.ALL, *constraints)
        [c.take_measurement(0, random_truth) for c in constraints]
        hf.perform_fusion(estimated_state)

        # For each of the possible filters
        for is_enabled, filter_id in zip(is_enabled, _GT_FILTERS):

            # Observe the original and update estimates and truth
            old_value = getattr(original_estimate, filter_id)
            new_value = getattr(estimated_state, filter_id)
            gt_value = getattr(random_truth, filter_id)

            if is_enabled:
                expected = old_value + filter_gain * (gt_value - old_value)
                np.testing.assert_array_equal(expected, new_value)
            else:
                np.testing.assert_array_equal(old_value, new_value)









