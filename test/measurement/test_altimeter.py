
import pytest
import math



from .conftest import assert_array_equal
from qnav.measurement.altimeter import Altimeter


@pytest.mark.parametrize("args",(
        {'freq': 1},
        {'freq': 100},
        {'freq': 200},
))
def test_altimeter_timer(args):

    # The number of iterations to attempt.
    num_iter = 5

    freq = args['freq']
    altimeter = Altimeter(freq)

    dt = 1 / freq
    expected = dt

    for _ in range(num_iter):
        assert altimeter.next_update == expected
        altimeter.update()
        expected += dt

@pytest.mark.parametrize("args", ((
    {'freq': 1.0, 'bias_error': 0, 'drift_rate': 0, 'seed': 1},
    {'freq': 1.0, 'bias_error': 1, 'drift_rate': 0, 'seed': 2},
    {'freq': 1.0, 'bias_error': 0, 'drift_rate': 1, 'seed': 3},
    {'freq': 1.0, 'bias_error': 1, 'drift_rate': 1, 'seed': 4},
    {'freq': 2.5, 'bias_error': 0.5, 'drift_rate': 0.5, 'seed': 5},
)))
def test_altimeter_errors(mock_rng, random_truth, args: dict):

    freq = args['freq']
    bias_error = args['bias_error']
    drift_rate = args['drift_rate']
    seed = args['seed']

    altimeter = Altimeter(freq, bias_error, drift_rate, rand_seed=seed)
    altimeter.take_measurement(0, random_truth)
    expected = random_truth.position[2] + bias_error
    dt = math.sqrt(1 / freq)

    for i in range(5):
        assert_array_equal(altimeter.last_measurement, expected)

        altimeter.update()
        altimeter.take_measurement(0, random_truth)
        expected += drift_rate * dt * seed



# @pytest.fixture
# def num_iterations() -> int:
#     return 5
#
# @pytest.fixture
# def perfect_altimeter() -> Altimeter:
#     return Altimeter(1, 0,0, 0)
#
# @pytest.fixture
# def random_altimeter() -> Altimeter:
#     return Altimeter(rand() + 0.1, rand() * 10 + 0.01, rand() * 10 + 0.01)
#
# @pytest.fixture
# def fixed_bias_altimeter() -> Altimeter:
#     return Altimeter(rand() + 0.1, rand() * 10 + 0.01,0, rand() * 10)
#
# @pytest.fixture
# def fixed_drift_altimeter() -> Altimeter:
#     return Altimeter(rand() + 0.1, 0, rand() * 10 + 0.01, rand() * 10)
#
# def test_perfect_altitude(perfect_altimeter: Altimeter, random_truth: GroundTruth):
#     perfect_altimeter.take_measurement(0, random_truth)
#     measurement = perfect_altimeter.last_measurement
#     assert measurement == random_truth.position[2]
#
# def test_perfect_no_drift(perfect_altimeter: Altimeter, random_truth: GroundTruth, num_iterations: int):
#     assess_drift(perfect_altimeter, random_truth, True, num_iterations)
#
# def test_fixed_bias_no_drift(fixed_bias_altimeter: Altimeter, random_truth: GroundTruth, num_iterations: int):
#     assess_drift(fixed_bias_altimeter, random_truth, True, num_iterations)
#
# def test_expected_drift(random_altimeter: Altimeter, random_truth: GroundTruth, num_iterations: int):
#     assess_drift(random_altimeter, random_truth, False, num_iterations)
#
# def test_no_bias_drift(fixed_drift_altimeter: Altimeter, random_truth: GroundTruth, num_iterations: int):
#     assess_drift(fixed_drift_altimeter, random_truth, False, num_iterations)
#
#
# @pytest.mark.parametrize("time_step", (1, 2, 4, 8, 16))
# def test_variable_frequency(perfect_altimeter: Altimeter, time_step: float):
#     update_1 = perfect_altimeter.next_update
#     perfect_altimeter.update(time_step)
#     update_2 = perfect_altimeter.next_update
#     assert time_step == (update_2 - update_1)
#
# def test_fixed_frequency(perfect_altimeter: Altimeter):
#     freq = perfect_altimeter.frequency
#     update_1 = perfect_altimeter.next_update
#     perfect_altimeter.update()
#     update_2 = perfect_altimeter.next_update
#     assert (1 / freq) == (update_2 - update_1)
#
# def assess_drift(altimeter: Altimeter, truth: GroundTruth,
#                  drift_expected: bool, num_attempts: int):
#
#     altimeter.take_measurement(0, truth)
#     alt_1 = altimeter.last_measurement
#
#     for i in range(num_attempts):
#         altimeter.update()
#         altimeter.take_measurement(0, truth)
#         alt_2 = altimeter.last_measurement
#         assert (alt_1 != alt_2) != drift_expected