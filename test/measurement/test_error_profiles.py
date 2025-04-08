import pytest

from qnav.measurement.error_properties import ErrorProperties

import numpy as np

@pytest.fixture
def random_error_prop():
    return ErrorProperties(
        bias_error=np.random.rand(3),
        bias_drift_rate=np.random.rand(3),
        scale_error=np.random.rand(3),
        non_orth_error=np.random.rand(6),
        avg_meas_noise=np.random.rand(3)
    )

def test_profile_matrix(random_error_prop):

    scale_errors = random_error_prop.scale_error
    non_orth_errors = random_error_prop.non_orth_error
    error_matrix = random_error_prop.error_matrix

    scale_indices = ((0, 0), (1, 1), (2, 2))
    non_orth_indices = ((0, 1), (0, 2), (1, 0),
                        (1, 2), (2, 0), (2, 1))

    # Ensure scale indices are in micro-units + 1
    for i, (j1, j2) in enumerate(scale_indices):
        expected = (scale_errors[i] * 1e-6) + 1
        assert error_matrix[j1, j2] == expected

    # Ensure non-orth indices are in micro-units
    for i, (j1, j2) in enumerate(non_orth_indices):
        expected = (non_orth_errors[i] * 1e-6)
        assert error_matrix[j1, j2] == expected


def test_profile_fields():

    bias_error = np.random.randn(3)
    bias_drift_rate = np.random.randn(3)
    scale_error = np.random.randn(3)
    non_orth_error = np.random.randn(6)
    avg_meas_noise = np.random.randn(3)

    error_props = ErrorProperties(
        bias_error=bias_error,
        bias_drift_rate=bias_drift_rate,
        scale_error=scale_error,
        non_orth_error=non_orth_error,
        avg_meas_noise=avg_meas_noise
    )

    assert np.all(error_props.bias_error == bias_error)
    assert np.all(error_props.bias_drift_rate == bias_drift_rate)
    assert np.all(error_props.scale_error == scale_error)
    assert np.all(error_props.non_orth_error == non_orth_error)
    assert np.all(error_props.avg_meas_noise == avg_meas_noise)


def create_with_random_fields(sizes: list[int]):

    bias_error = np.random.randn(sizes[0])
    bias_drift_rate = np.random.randn(sizes[1])
    scale_error = np.random.randn(sizes[2])
    non_orth_error = np.random.randn(sizes[3])
    avg_meas_noise = np.random.randn(sizes[4])

    return ErrorProperties(
        bias_error=bias_error,
        bias_drift_rate=bias_drift_rate,
        scale_error=scale_error,
        non_orth_error=non_orth_error,
        avg_meas_noise=avg_meas_noise
    )

@pytest.mark.parametrize('field_sizes', (
    [1, 1, 1, 1, 1],
    [3, 3, 3, 6, 3],
))
def test_fields_with_valid_sizes(field_sizes: list[int]):
    error_properties = create_with_random_fields(field_sizes)
    assert len(error_properties.bias_error) == 3
    assert len(error_properties.bias_drift_rate) == 3
    assert len(error_properties.scale_error) == 3
    assert len(error_properties.non_orth_error) == 6
    assert len(error_properties.avg_meas_noise) == 3

@pytest.mark.parametrize('field_sizes', (
    [2, 3, 3, 6, 3],
    [3, 4, 3, 6, 3],
    [3, 3, 5, 6, 3],
    [3, 3, 3, 7, 3],
    [3, 3, 3, 6, 8]
))
def test_fields_with_invalid_sizes(field_sizes: list[int]):
    with pytest.raises(ValueError, match=r".* must have .*"):
        create_with_random_fields(field_sizes)

