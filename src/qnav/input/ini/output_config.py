"""
================
output_config.py
================

:summary:
    Functions related to handling requested simulation output.
    A collection of functions used for reading content under the 'output'
    section within the configuration and initialising the corresponding
    objects. This mostly manages the configuring of what the simulation
    is to output and display.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""

from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional
from math import isinf

from qnav.input.config_handler import ConfigHandler, NavConfigError
from qnav.input.ini.estimation_config import use_virtual_memory
from qnav.input.ini.measurement_config import get_imu_frequency
from qnav.output.formats import SaveFormat
from qnav.simulation.display import Display, ConsoleDisplayPlus, ConsoleDisplay, MultilineDisplay
from qnav.simulation.results import ResultsCapture, CaptureAtFixedRate

# The shared section name to use
__SECTION_ID: str = "Output"



def get_verbose_enabled(config: ConfigHandler) -> bool:

    get_bool = partial(config.get_bool, __SECTION_ID)
    return get_bool("showVerbose", False)




def get_results_capture(config: ConfigHandler) -> ResultsCapture:

    get_int = partial(config.get_int, __SECTION_ID)

    base_freq = get_imu_frequency(config)
    down_sample =  get_int("resultsDownSampleRate", 1)

    if down_sample < 1 or isinf(down_sample):
        raise NavConfigError(__SECTION_ID, "outputDownSampleRate",
                             "Must Be Positive", "value must be positive integer")

    freq = base_freq / down_sample
    use_vm = use_virtual_memory(config)
    return CaptureAtFixedRate(freq, use_virtual_memory=use_vm)


def get_displayer(config: ConfigHandler) -> Display:

    get_float = partial(config.get_float, __SECTION_ID)
    get_id = partial(config.get_str_alpha, __SECTION_ID)

    display_mode = get_id("displayInfo", "time")
    update_interval = get_float("displayInterval", 5.0)
    auto_stop_error = get_float("autoStopMaxEstimationError", float('inf'))

    # get_bool = partial(config.get_bool, __SECTION_ID)
    # calculate_errors = get_bool("calculateErrors", False)
    #
    # if calculate_errors:
    #     return ConsoleDisplayPlus(update_interval, auto_stop_error)
    # else:
    #     return ConsoleDisplay(update_interval)

    # Return the requested fusion method:
    match display_mode:

        case 'time':
            return ConsoleDisplay(update_interval)

        case 'error':
            return ConsoleDisplayPlus(update_interval, auto_stop_error)

        case 'all':
            return MultilineDisplay(update_interval, auto_stop_error)

        case _:
            raise NavConfigError(__SECTION_ID, 'displayMode','Unknown method',
                                 f'Unrecognized display method: {display_mode}')


def get_output_dir(config: ConfigHandler) -> Path:

    get_path = partial(config.get_value_filepath, __SECTION_ID)
    get_bool = partial(config.get_bool, __SECTION_ID)
    output_dir = get_path("resultsPath")

    if get_bool("appendDateFolder", False):

        # Use the format 'dd mm yyyy  HH_MM_SS'
        now = datetime.now()
        date_str = now.strftime("%d %b %Y  %H_%M_%S")
        output_dir /= date_str

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    return output_dir


def get_output_downsample(config: ConfigHandler) -> int:
    get_int = partial(config.get_int, __SECTION_ID)
    return max(1, get_int("outputDownSampleRate", 1))


def get_copy_config(config: ConfigHandler) -> bool:
    get_bool = partial(config.get_bool, __SECTION_ID)
    return get_bool("copyConfig", False)

def get_output_formats(config: ConfigHandler) -> list[SaveFormat]:

    get_csv = partial(config.get_csv_str, __SECTION_ID)
    requested_formats = get_csv("saveFormat", [])

    to_return: list[SaveFormat] = list()
    for req in requested_formats:
        req = str(req).casefold()

        for fmt in SaveFormat:
            if req == fmt.match_id.casefold():
                to_return.append(fmt)

    return to_return


def get_save_figures(config: ConfigHandler) -> bool:
    get_bool = partial(config.get_bool, __SECTION_ID)
    return get_bool("saveFigures", True)

def get_show_figures(config: ConfigHandler) -> bool:
    get_bool = partial(config.get_bool, __SECTION_ID)
    return get_bool("showFigures", False)

def get_figure_formats(config: ConfigHandler) -> list[str]:
    get_csv = partial(config.get_csv_str, __SECTION_ID)
    formats = set(get_csv("figureFormat", ["png"]))
    return list(formats)

def get_figure_renderer(config: ConfigHandler) -> Optional[str]:
    get_str = partial(config.get_str, __SECTION_ID)
    renderer = get_str("figureRenderer", "default")
    return renderer if renderer != "default" else None
