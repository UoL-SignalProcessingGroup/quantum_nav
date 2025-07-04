"""
===================
run_all_examples.py
===================

:summary:
    A helper script to automatically execute all example configurations.
    This script will gather all example configurations and execute
    them sequentially, collecting each of their results. Summary
    results for final estimated position and error will produced.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from datetime import datetime
import qnav.input.config as cfg
import qnav.simulation.simulation as sim
from run import read_config, save_results
from multiprocessing import Pool, cpu_count
from qnav.output.summary import read_summary_file
from qnav.simulation.display import DisplayStatus, MPDisplay

# Dictionary of shared memory between processes
_shared_memory = {}


def get_all_configs(configs_dir: Path) -> list[Path]:
    """
    Obtains list of the paths for all the example configuration files.
    Recursively searches for .ini configurations within the examples folder
    and returns a sorted list of paths.

    :param configs_dir: Path to the project directory.
    :type configs_dir: Path

    :return: A sorted list of paths.
    :rtype: list[Path]
    """
    assert configs_dir.is_dir(), f'Directory {configs_dir} does not exist.'
    examples = configs_dir.rglob('*.ini')
    return sorted(list(examples))


def run_all_configs(config_files: list[Path],
                    config_overrides: dict[tuple[str, str], str] = None,
                    num_threads: int = None,
                    verbose_freq: float = 1.0) -> list[dict]:
    """
    Executes simulations for all the given configurations files.
    Runs multiple simulations simultaneously, using list of given
    configuration files, override settings and thread count. A
    separate thread for printing progress is automatically used.

    :param config_files: Paths to the configuration files.
    :type config_files: list[Path]

    :param config_overrides: Optional overrides for configuration files.
    :type config_overrides: dict[tuple[str, str], str]

    :param num_threads: Number of CPU threads to use.
    :type num_threads: int

    :param verbose_freq: Console printing frequency in seconds.
    :type verbose_freq: float

    :return: A list containing dictionaries of the execution success status,
        any exception encountered, and path of output dir.
    :rtype: dict[str, any]
    """

    # Obtain collection of configurations and settings to use per simulation.
    configs_and_settings = [(config, config_overrides) for config in config_files]

    # Collect arguments for shared memory
    num_configs = len(config_files)
    shared_progress = mp.RawArray('d', np.zeros(num_configs))
    shared_status = mp.RawArray('d', np.ones(num_configs))
    init_args = (config_files, shared_progress, shared_status)

    # Obtain optimal number of threads:
    if num_threads is None:
        num_threads = cpu_count()

    # Print running information in output console:
    num_threads = max(min(num_threads, num_configs), 1)
    print(f'Running {num_configs} simulations on {num_threads} threads:')
    num_threads += 1

    # Execute a thread pool for running each of the configurations.
    with Pool(processes=num_threads, initializer=_thread_init, initargs=init_args) as pool:
        pool.apply_async(_print_progress, args=(verbose_freq,))
        results = pool.starmap(_run_configuration, configs_and_settings)
        time.sleep(verbose_freq)
        pool.close()
        return results


def _thread_init(config_files: list[Path], progress: mp.Array, status: mp.Array):
    """
    Used to initialise threads with shared global memory.
    Specifically, this takes the shared arrays for
    configuration progress percentage and status.

    :param config_files: Paths to the configuration files.
    :type config_files: list[Path]

    :param progress: Shared array for all configurations.
    :type progress: mp.Array

    :param status: Shared array for all configurations.
    :type status: mp.Array
    """
    global _shared_memory
    _shared_memory['config_files'] = config_files
    _shared_memory['progress'] = progress
    _shared_memory['status'] = status


def _mute_thread_output():
    """
    Disables standard out and error printing.
    Directs all standard and error output to dev-null.
    """
    dev_null = open(os.devnull, 'w')
    sys.stdout = dev_null
    sys.stderr = dev_null


def _run_configuration(config_file: Path,
                       overrides: dict[tuple[str, str], str] = None) -> dict:
    """
    Execute simulation for individual configuration file.
    For a given configuration file, this function executes a standard
    simulation with some settings controlling the output overridden. General
    output is collected and returned; including a status flag for successful
    execution, any exceptions encountered and the path for the produced output.

    :param config_file: Path for a given configuration file.
    :type config_file: Path

    :param overrides: Optional dictionary of config settings to override.
        This is a dictionary with tuple keys of config section and settings
        with overriding values as the value.
    :type overrides: dict[tuple[str, str], str]

    :return: A dictionary containing the execution success status,
        any exception encountered, and path of output dir.
    :rtype: dict[str, any]
    """

    # Default returns
    error = None
    success = False
    output_dir = None
    displayer = None

    # Extract shared memory
    global _shared_memory
    is_mp = 'progress' in _shared_memory and 'status' in _shared_memory

    try:

        # Hide standard output if multi-process
        if is_mp:
            _mute_thread_output()

        # Assert that the file exists:
        if not config_file.is_file():
            raise FileNotFoundError('Configuration file not found')

        # Read configuration
        config = read_config(config_file)

        # Apply any configuration overrides
        if overrides:
            for key, value in overrides.items():
                config.overwrite_value(key[0], key[1], value)

        # If multi-process
        if is_mp:

            # Unpack shared items
            all_configs = _shared_memory['config_files']
            progress = _shared_memory['progress']
            status = _shared_memory['status']
            index = all_configs.index(config_file)

            # Use a Multi-Process displayer
            displayer = MPDisplay(1.0, index, progress, status)
            displayer.status = DisplayStatus.STARTING

        else:

            # Otherwise use the default displayer
            displayer = cfg.get_displayer(config)

        # Attempt to run normal simulation with configuration
        trajectory = cfg.get_trajectory(config)
        estimates = cfg.get_estimation(config, trajectory)
        hardware = cfg.get_platform_hardware(config, estimates)
        clock = cfg.get_platform_clock(config)
        results = cfg.get_collector(config)

        # Execute the simulation
        sim.run_simulation_loop(
            trajectory, estimates, hardware,
            clock, results, displayer)

        # Collect output dir and mark execution as successful
        output_dir = save_results(config, results)
        success = True

    except Exception as e:
        if hasattr(displayer, 'status'):
            displayer.status = DisplayStatus.FAILURE
        error = e

    # Finally, return success status, exceptions encounter and output dir.
    return {
        'config': config_file,
        'success': success,
        'error': error,
        'dir': output_dir
    }


def _print_progress(update_interval: float = 1.0):
    """
    Continually prints simulation progress to the stream.
    A function to be executed by a thread that will repeatedly display
    shared simulation progress to the output console.

    :param update_interval: (Optional) How often to print updates in seconds.
        By default, updates are given roughly every second.
    :type update_interval: float
    """

    global _shared_memory
    required_fields = ['config_files', 'progress', 'status']
    assert all([f in _shared_memory for f in required_fields]), \
        "Global shared memory not initialised."

    config_files = _shared_memory['config_files']
    shared_percentages = _shared_memory['progress']
    shared_statuses = _shared_memory['status']

    # Sanity check input sizes:
    num_lines = len(config_files)
    assert num_lines == len(shared_percentages) == len(shared_statuses), \
        "Input arrays must have the same length!"

    display_names = [str(c) for c in config_files]
    max_name_len = max([len(n) for n in display_names])
    max_bar_len = 30

    while True:
        for i in range(num_lines):

            # Get details for current index
            progress = shared_percentages[i]
            status_id = shared_statuses[i]
            config_name = display_names[i]

            name = f'{str(config_name):{max_name_len + 1}} '
            bar = f'{'#' * int(progress * max_bar_len)}'
            bar = f'{bar:{'_'}<{max_bar_len}}'

            match status_id:

                case DisplayStatus.NOT_STARTED:
                    status = 'Waiting'

                case DisplayStatus.STARTING:
                    status = 'Running'

                case DisplayStatus.RUNNING:
                    status = f'{progress * 100:>.2f}%'
                    status = f'{status:>7}'

                case DisplayStatus.FINISHED:
                    status = 'Complete'

                case DisplayStatus.FAILURE:
                    status = 'Failure'

                case _:
                    status = 'Unknown'

            print(f'{name} [{bar}] ({status})')

        # Delay for update interval time
        print('', flush=True)
        time.sleep(update_interval)

        # Either clear or add border before re-printing
        for _ in range(num_lines + 1):
            print('', end='\033[1A')


def read_summary_results(results: list[dict]):
    """
    Reads and appends produced summary results to a list.
    For a list of given result dictionaries, this function will read the
    summary results for the successful runs and append.

    :param results: List of produced simulation results.
    :type results: list[dict]
    """
    for result in results:
        if result['success']:
            summary_file = result['dir'] / 'summary.json'
            result['results'] = read_summary_file(summary_file)


def get_summary_data_frame(results: list[dict]):
    """
    Produces a Pandas data frame containing summary results.
    Generates a simple table listing the final actual, estimated
    and error values for positions.

    :param results: List of produced simulation results.
    :type results: list[dict]

    :return: A Pandas data frame containing summary results.
    :rtype: pandas.DataFrame
    """

    num_results = len(results)

    data = {
        'configs': [''] * num_results,
        'actual_lat': np.full(num_results, np.nan),
        'actual_lon': np.full(num_results, np.nan),
        'actual_alt': np.full(num_results, np.nan),
        'estimated_lat': np.full(num_results, np.nan),
        'estimated_lon': np.full(num_results, np.nan),
        'estimated_alt': np.full(num_results, np.nan),
        'error_north': np.full(num_results, np.nan),
        'error_east': np.full(num_results, np.nan),
        'error_down': np.full(num_results, np.nan),
    }

    for i, result in enumerate(results):
        data['configs'][i] = result['config']

        if result['success']:

            r = result['results']['summary']
            true_pos = r['ground_truth']['final']['position']
            est_pos = r['estimated']['final']['position']
            error_pos = r['errors']['final']['position']

            data['actual_lat'][i] = true_pos[0]
            data['actual_lon'][i] = true_pos[1]
            data['actual_alt'][i] = true_pos[2]
            data['estimated_lat'][i] = est_pos[0]
            data['estimated_lon'][i] = est_pos[1]
            data['estimated_alt'][i] = est_pos[2]
            data['error_north'][i] = error_pos[0]
            data['error_east'][i] = error_pos[1]
            data['error_down'][i] = error_pos[2]

    return pd.DataFrame(data)


if __name__ == '__main__':
    """
    Called when the script is executed directly.
    This is the entry point when using the software from the command-line.
    On execution, reads the configuration file given in the command-line 
    arguments (or uses the default) and completes a typical simulation.
    """

    # Assert a file path is given
    assert len(sys.argv) >= 2, \
        'Please provide file path as first argument'

    # Gather path for all configurations.
    examples_dir = Path(sys.argv[1])   # Path('examples')
    configs = get_all_configs(examples_dir)

    # Specify max number of threads to use
    max_threads = cpu_count()  # Use None for auto

    # Use command line argument if provided:
    if len(sys.argv) >= 3:  # TODO: Use flag for this
        max_threads = max(int(sys.argv[2]), 1)

    # Specify configuration settings to override
    override_settings = {
        ('Output', 'showFigures'): 'no',
        ('Output', 'saveFigures'): 'no',
        ('Output', 'saveSummary'): 'yes'
    }

    # Execute all configurations concurrently
    print_freq = 0.66  # In seconds
    outputs = run_all_configs(configs[:-4], override_settings, max_threads, print_freq)
    # output = _run_configuration(configs[0], override_settings)

    # Collect summary results for all:
    read_summary_results(outputs)
    output_frame = get_summary_data_frame(outputs)

    # Print the entire output frame
    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None):
    #     print(output_frame)

    # Export the results
    # TODO: Allow CLI configuration of output dir
    date_now = datetime.now()
    date_str = date_now.strftime('%d %b %Y %H_%M_%S')
    output_name = f'All Results  {date_str}.csv'
    output_frame.to_csv(output_name, sep=',')
