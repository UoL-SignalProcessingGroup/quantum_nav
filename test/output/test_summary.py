from pathlib import Path

import numpy as np
import pytest

from qnav.output.data import ResultsTable
from qnav.output.figures import plot_summary
from qnav.output.summary import get_record_info


def _table(path: Path, times: np.ndarray, **fields: np.ndarray) -> ResultsTable:
    table = ResultsTable(path)
    table.write("Test", times, **fields)
    return table


def test_errors_use_the_same_complete_finite_rows(tmp_path: Path):
    times = np.arange(4.0)
    estimates = _table(
        tmp_path / "estimates.qnr",
        times,
        velocity=np.array([
            [10.0, 10.0, 10.0],
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, 9.0],
        ]),
    )
    ground_truth = _table(
        tmp_path / "ground_truth.qnr",
        times,
        velocity=np.array([
            [np.nan, np.nan, np.nan],
            [3.0, 5.0, 7.0],
            [8.0, 9.0, 10.0],
            [11.0, 13.0, 15.0],
        ]),
    )

    errors = get_record_info(estimates, ground_truth)["errors"]

    assert errors["initial"]["velocity"] == [2.0, 3.0, 4.0]
    assert errors["final"]["velocity"] == [4.0, 5.0, 6.0]


def test_errors_are_none_without_a_paired_finite_row(tmp_path: Path):
    times = np.arange(2.0)
    estimates = _table(
        tmp_path / "estimates.qnr",
        times,
        velocity=np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]),
    )
    ground_truth = _table(
        tmp_path / "ground_truth.qnr",
        times,
        velocity=np.array([[np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]]),
    )

    errors = get_record_info(estimates, ground_truth)["errors"]

    assert errors["initial"]["velocity"] is None
    assert errors["final"]["velocity"] is None


def test_errors_require_exactly_aligned_timestamps(tmp_path: Path):
    estimates = _table(
        tmp_path / "estimates.qnr",
        np.array([0.0, 1.0]),
        velocity=np.zeros((2, 3)),
    )
    ground_truth = _table(
        tmp_path / "ground_truth.qnr",
        np.array([0.0, 1.001]),
        velocity=np.zeros((2, 3)),
    )

    with pytest.raises(ValueError, match="timestamps must be exactly aligned"):
        get_record_info(estimates, ground_truth)


def test_attitude_errors_use_shortest_signed_angle(tmp_path: Path):
    times = np.arange(2.0)
    estimates = _table(
        tmp_path / "estimates.qnr",
        times,
        attitude=np.array([[359.0, 1.0, 180.0], [1.0, 359.0, -180.0]]),
    )
    ground_truth = _table(
        tmp_path / "ground_truth.qnr",
        times,
        attitude=np.array([[1.0, 359.0, 0.0], [359.0, 1.0, 0.0]]),
    )

    errors = get_record_info(estimates, ground_truth)["errors"]

    assert errors["initial"]["attitude"] == [2.0, -2.0, -180.0]
    assert errors["final"]["attitude"] == [-2.0, 2.0, -180.0]


def test_summary_figure_wraps_attitude_errors(tmp_path: Path):
    times = np.arange(2.0)
    common = {
        "position": np.array([[52.0, -2.0, 100.0]] * 2),
        "velocity": np.zeros((2, 3)),
        "acceleration": np.zeros((2, 3)),
        "angle_rates": np.zeros((2, 3)),
    }
    estimates = _table(
        tmp_path / "estimates.qnr",
        times,
        **common,
        attitude=np.array([[359.0, 1.0, 180.0], [1.0, 359.0, -180.0]]),
    )
    ground_truth = _table(
        tmp_path / "ground_truth.qnr",
        times,
        **common,
        attitude=np.array([[1.0, 359.0, 0.0], [359.0, 1.0, 0.0]]),
    )

    figure = plot_summary(estimates, ground_truth)
    attitude_table = np.asarray(figure.data[3].cells.values)

    assert attitude_table[1].tolist() == [
        "2.000000", "-2.000000", "-180.000000"]
    assert attitude_table[2].tolist() == [
        "2.000000", "2.000000", "180.000000"]
    assert attitude_table[5].tolist() == [
        "-2.000000", "2.000000", "-180.000000"]
