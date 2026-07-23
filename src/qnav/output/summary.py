"""
==========
summary.py
==========

:summary:
    Module for collecting summary statistics for simulations.
    Provides functions for gathering summary information, given simulation
    data. These can be exported along with the simulation results.

:version:
    1.0.0 - First full implemented version.
"""

import numpy as np
import json

from qnav.util.transformations import lla2ned
from qnav.input.config_handler import ConfigHandler
from qnav.output.data import ResultsTable
from qnav import __version__ as qnav_version
from pathlib import Path


# Shared field names for records we are interested in
__FIELD_NAMES = ["position", "velocity", "acceleration", "attitude", "angle_rates"]

def write_summary_file(file_path: Path,
                       config: ConfigHandler,
                       estimates_table: ResultsTable,
                       ground_truth_table: ResultsTable = None):
    """
    Generates and writes summary information to given json file.
    Allows the exporting of simulation summary data.

    :param file_path: The JSON file to write summary data to.
    :type file_path: pathlib.Path

    :param config: The configuration instance being used for.
    :type config: ConfigHandler

    :param estimates_table: Results table holding estimation records.
    :type estimates_table: ResultsTable

    :param ground_truth_table: Results table holding ground truth records.
    :type ground_truth_table: ResultsTable
    """
    summary = get_summary(config, estimates_table, ground_truth_table)
    with open(file_path.with_suffix('.json'), "w") as file:
        json.dump(summary, file, indent=4)

def read_summary_file(file_path: Path):
    """
    Reads summary information to previously generated json file.
    Allows the importing and reading of simulation summary data.

    :param file_path: The JSON file to read summary data from.
    :type file_path: pathlib.Path
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def get_summary(config: ConfigHandler,
                estimates_table: ResultsTable,
                ground_truth_table: ResultsTable = None) -> dict[str, dict]:
    """
    Generate dictionary holding simulation summary information.
    This should give a light overview of the results.

    :param config: The configuration instance being used for.
    :type config: ConfigHandler

    :param estimates_table: Results table holding estimation records.
    :type estimates_table: ResultsTable

    :param ground_truth_table: Results table holding ground truth records.
    :type ground_truth_table: ResultsTable
    """

    return {
        'config': get_config_info(config),
        'summary': get_record_info(estimates_table, ground_truth_table)
    }

def get_config_info(config: ConfigHandler) -> dict:
    """
    Collection general configuration information to export.
    These are details that should be included in a summary report.

    :param config: The configuration instance being used for.
    :type config: ConfigHandler
    """

    software_version = qnav_version
    config_name = config.get_config_name()
    config_hash = hash(config)

    return {
        "software_version": software_version,
        "config_name": config_name,
        "config_hash": config_hash,
    }

def get_record_info(estimates_table: ResultsTable,
                    ground_truth_table: ResultsTable = None) -> dict:
    """
    Collection general record information to export.
    These are details that should be included in a summary report.

    :param estimates_table: Results table holding estimation records.
    :type estimates_table: ResultsTable

    :param ground_truth_table: Results table holding ground truth records.
    :type ground_truth_table: ResultsTable
    """

    # Ensure all results are loaded:
    if not estimates_table.is_data_loaded: estimates_table.read()
    if ground_truth_table is not None and not ground_truth_table.is_data_loaded:
        ground_truth_table.read()

    # Acquire all the results data:
    estimated_data = estimates_table.get_data()
    result = {'estimated': _get_stats(estimated_data)}
    if ground_truth_table is not None and ground_truth_table.is_data_loaded:
        ground_truth_data = ground_truth_table.get_data()
        result['ground_truth'] = _get_stats(ground_truth_data)
        result['errors'] = _get_errors(estimated_data, ground_truth_data)
    return result

def _get_stats(table: dict) -> dict:
    stats = {'initial': {}, 'final': {}}
    for field in __FIELD_NAMES:
        if field not in table['data']:
            continue
        stats['initial'][field] = _get_initial_record(table, field)
        stats['final'][field] = _get_final_record(table, field)
    return stats

def _get_errors(table_a: dict, table_b: dict) -> dict:
    _require_aligned_times(table_a, table_b)
    stats = {'initial': {}, 'final': {}}
    for field in __FIELD_NAMES:
        if field not in table_a['data'] or field not in table_b['data']:
            continue
        stats['initial'][field] = _get_initial_error(table_a, table_b, field)
        stats['final'][field] = _get_final_error(table_a, table_b, field)
    return stats

def _get_initial_record(table: dict, field_name: str) -> np.ndarray | float:
    records = table['data'][field_name]
    index = _first_valid_index(records)
    return None if index is None else _array_to_python(records[index])

def _get_final_record(table: dict, field_name: str) -> np.ndarray | float:
    records = table['data'][field_name]
    last_ind = _last_valid_index(records)
    return None if last_ind is None else _array_to_python(records[last_ind])

def _get_initial_error(table_a: dict, table_b: dict, field_name: str) -> np.ndarray | float:
    index = _first_paired_valid_index(
        table_a['data'][field_name], table_b['data'][field_name])
    if index is None:
        return None
    record_a = table_a['data'][field_name][index]
    record_b = table_b['data'][field_name][index]
    error = _calculate_error(record_a, record_b, field_name)
    return _array_to_python(error)

def _get_final_error(table_a: dict, table_b: dict, field_name: str) -> np.ndarray | float:
    records_a = table_a['data'][field_name]
    records_b = table_b['data'][field_name]
    index = _last_paired_valid_index(records_a, records_b)
    if index is None:
        return None
    error = _calculate_error(records_a[index], records_b[index], field_name)
    return _array_to_python(error)


def _require_aligned_times(table_a: dict, table_b: dict) -> None:
    times_a = np.asarray(table_a['time_steps'])
    times_b = np.asarray(table_b['time_steps'])
    if times_a.shape != times_b.shape or not np.array_equal(times_a, times_b):
        raise ValueError(
            "estimate and ground-truth timestamps must be exactly aligned")


def _complete_rows(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    finite = np.isfinite(values)
    return finite if values.ndim == 1 else np.all(finite, axis=1)


def _paired_valid_rows(data_a: np.ndarray, data_b: np.ndarray) -> np.ndarray:
    if np.shape(data_a) != np.shape(data_b):
        raise ValueError(
            "estimate and ground-truth fields must have matching shapes")
    valid_a = _complete_rows(data_a)
    valid_b = _complete_rows(data_b)
    return valid_a & valid_b


def _first_paired_valid_index(data_a: np.ndarray,
                              data_b: np.ndarray) -> int | None:
    indices = np.flatnonzero(_paired_valid_rows(data_a, data_b))
    return None if len(indices) == 0 else int(indices[0])


def _last_paired_valid_index(data_a: np.ndarray,
                             data_b: np.ndarray) -> int | None:
    indices = np.flatnonzero(_paired_valid_rows(data_a, data_b))
    return None if len(indices) == 0 else int(indices[-1])

def _valid_rows(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    finite = ~np.isnan(values)
    return finite if values.ndim == 1 else np.any(finite, axis=1)


def _first_valid_index(data: np.ndarray) -> int | None:
    indices = np.flatnonzero(_valid_rows(data))
    return None if len(indices) == 0 else int(indices[0])


def _last_valid_index(data: np.ndarray) -> int | None:
    indices = np.flatnonzero(_valid_rows(data))
    return None if len(indices) == 0 else int(indices[-1])

def _calculate_error(record_a, record_b, field_name) -> float | np.ndarray:
    if field_name == "position":
        return lla2ned(record_a, record_b)
    error = record_b - record_a
    if field_name == "attitude":
        error = (error + 180.0) % 360.0 - 180.0
    return error

def _array_to_python(data: np.ndarray) -> list | float:
    return data.squeeze().tolist()
