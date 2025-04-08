"""
==========
display.py
==========

:summary:
    Allows means for displaying simulation progress in real-time.
    This module is responsible for providing an interface for allowing the
    displaying of simulation data so that progress can be presented to the
    user in real-time.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation.
"""

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from sys import stdin

from qnav.estimation.state import EstimatedState
from qnav.waypoints.trajectory import GroundTruth
from qnav.waypoints.trajectory import Trajectory
from qnav.util.transformations import lla2ned

import numpy as np

#: Ascii/Terminal character for line up.
_LINE_UP = '\033[1A'

#: Ascii/Terminal character to clear line.
_LINE_CLEAR = '\x1b[2K'


def is_using_terminal() -> bool:
    """
    Checks if the application is being executed via terminal.
    Used to determine if the toolbox is being run and displayed through a
    command-line terminal, and not some interactive shell (or IDE). Used to
    obtain an idea of how output data is to be displayed.

    :return: True if standard output is being displayed inside terminal.
    :rtype: bool
    """
    return stdin.isatty()

def clear_terminal_line(num_lines: int = 1) -> None:
    """
    Clears a previously printed line in the console.
    Only works when executing application from command line.

    :param num_lines: Number of lines to clear (default = 1).
    :type num_lines: int
    """
    for _ in range(num_lines):
        print(_LINE_UP, end=_LINE_CLEAR)

def sec_to_str(time_sec: int) -> str:
    """
    Converts seconds to printed hours, minutes and seconds.
    Used to make printed time more human readable.

    :param time_sec: The time to display in seconds.
    :type time_sec: int

    :return: The given time in HH:MM:SS string format.
    :rtype: str
    """

    # Convert to minutes and seconds
    seconds = time_sec % 60
    minutes = time_sec // 60

    # Convert to hour, minutes, seconds
    hours = minutes // 60
    minutes %= 60

    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class Display(ABC):
    """
    A template used to define classes for displaying simulation progress.
    This provides the definition to be followed for classes that can be
    used for displaying simulation data during runtime. This is useful
    for users who wish to see verbose information or plot data during
    simulations, without having to wait until the end.
    """

    def __init__(self, update_interval: float):
        """
        Initialise displaying class by giving the update time interval.
        This defines how often displaying updates will be given (in
        simulation time).

        :param update_interval: The time intervals between updates in seconds.
        :type update_interval: float
        """
        self._time_step: float = update_interval
        self._next_update: float = 0

    @abstractmethod
    def on_start(self, trajectory: Trajectory):
        """
        Called at the start of the simulation before first iteration.
        Can be used to initialise any displaying components by extracting
        necessary data from the trajectory.

        :param trajectory: The ground truth trajectory instance.
        :type trajectory: Trajectory
        """
        self._next_update = trajectory.start_time

    @abstractmethod
    def on_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called during the simulation after each fusion update.
        Presents the current (true) timestep, estimated states and ground
        truth record.

        :param timestamp: The current (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The current estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        if timestamp >= self._next_update:
            self._next_update += self._time_step

    @abstractmethod
    def on_finish(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called at the end of the simulation after final iteration.
        Can be used to finish up any displaying of information. Presents the
        final (true) timestep, estimated states and ground truth record.

        :param timestamp: The final (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The final estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The final ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        pass


class ConsoleDisplay(Display):
    """
    A simple display for printing simulation progress in console.
    Used to display that periodically prints the simulation's runtime progress
    in the output console. Namely used as an example display class.
    """

    __slots__ = ('_start_time', 'end_time')

    def __init__(self, update_interval: float):
        """
        Initialise console printing by giving the update time interval.
        This defines how often displaying updates will be given (in
        simulation time).

        :param update_interval: The time intervals between updates in seconds.
        :type update_interval: float
        """
        super().__init__(update_interval)
        self._start_time = float('nan')
        self._end_time = float('nan')

    def on_start(self, trajectory: Trajectory):
        """
        Called at the start of the simulation before first iteration.
        Used to obtain the start and end times of the trajectory.

        :param trajectory: The ground truth trajectory instance.
        :type trajectory: Trajectory
        """
        self._start_time = trajectory.start_time
        self._end_time = trajectory.end_time

    def on_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called during the simulation after each fusion update.
        Updates the runtime progress display of the simulation.

        :param timestamp: The current (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The current estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        if timestamp >= self._next_update:
            self._next_update += self._time_step
            progress = (timestamp / self._end_time) * 100
            print(f"\r[{progress:8.4f}%]\t Time: {timestamp:12.4f}", end="", flush=True)

    def on_finish(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called at the end of the simulation after final iteration.
        Finishes the runtime progress display of the simulation.

        :param timestamp: The final (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The final estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The final ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        print(f"\r{'Simulation Complete!':50s}", flush=True)


class ConsoleDisplayPlus(ConsoleDisplay):
    """
    Extension to the basic displaying class that includes estimation errors.
    This subclass also calculates the estimated position error at each
    interval and displays it in the console. In addition to this, an
    optional threshold can be set for the maximum error distance for
    prematurely stopping the simulation.
    """

    __slots__ = ('_max_error', '_max_length')

    def __init__(self, update_interval: float, max_dist_error: float = None):
        """
        Initialise console printing by giving the update time interval.
        This defines how often displaying updates will be given (in
        simulation time).

        :param update_interval: The time intervals between updates in seconds.
        :type update_interval: float
        """
        super().__init__(update_interval)

        if max_dist_error is None:
            max_dist_error = float('inf')

        self._max_error = max_dist_error
        self._max_length = 0

    def print_and_process(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):


        progress = (timestamp / self._end_time) * 100
        position_error = lla2ned(estimated_state.position, ground_truth.position)
        north_error, east_error, down_error = position_error

        fmt_string = (f"[{progress:8.4f}%]\t NED Position Error: ("
                      f"{north_error:.5f}, "
                      f"{east_error:.5f}, "
                      f"{down_error:.5f})")

        self._max_length = max(self._max_length, len(fmt_string) + 2)
        print(f"\r{fmt_string:{self._max_length}s}", end="", flush=True)

        # TODO: Raise custom exception!
        assert np.linalg.norm(position_error) <= self._max_error, \
            f"Estimated position error has exceeded threshold of {self._max_error} metres"


    def on_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called during the simulation after each fusion update.
        Updates the runtime progress display of the simulation.

        :param timestamp: The current (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The current estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        if timestamp >= self._next_update:
            self._next_update += self._time_step
            self.print_and_process(timestamp, estimated_state, ground_truth)

            # self._next_update += self._time_step
            # progress = (timestamp / self._end_time) * 100
            #
            # position_error = lla2ned(
            #     estimated_state.position, ground_truth.position)
            # north_error, east_error, down_error = position_error
            #
            # fmt_string = (f"[{progress:8.4f}%]\t NED Position Error: ("
            #               f"{north_error:.5f}, "
            #               f"{east_error:.5f}, "
            #               f"{down_error:.5f})")
            #
            # self._max_length = max(self._max_length, len(fmt_string) + 2)
            # print(f"\r{fmt_string:{self._max_length}s}", end="", flush=True)
            #
            # # TODO: Raise custom exception!
            # assert np.linalg.norm(position_error) <= self._max_error, \
            #     f"Estimated position error has exceeded threshold of {self._max_error} metres"

    def on_finish(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called at the end of the simulation after final iteration.
        Finishes the runtime progress display of the simulation.

        :param timestamp: The final (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The final estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The final ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        self.print_and_process(timestamp, estimated_state, ground_truth)
        print('Simulation Complete!')
        # print(f"{'Simulation Complete!':{self._max_length}s}", flush=True)





class MultilineDisplay(ConsoleDisplayPlus):

    def __init__(self, update_interval: float, max_dist_error: float = None):
        super().__init__(update_interval, max_dist_error)
        self._is_using_terminal = is_using_terminal()
        self._sim_init_time = datetime.now()
        self._max_length: list[int] = [0, 0, 0, 0]

    def _get_real_time(self):
        return (datetime.now() - self._sim_init_time).total_seconds()


    def on_start(self, trajectory: Trajectory):

        super().on_start(trajectory)

        num_lines = len(self._max_length) \
            if is_using_terminal() else 1

        for _ in range(num_lines):
            print('Starting...')

    def on_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        if timestamp >= self._next_update:
            self._next_update += self._time_step
            self._print_update(timestamp, estimated_state, ground_truth)

    def on_finish(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        self._print_update(timestamp, estimated_state, ground_truth)

    def _print_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):

        progress = (timestamp / self._end_time) * 100
        sim_time = sec_to_str(int(timestamp))
        real_time = sec_to_str(int(self._get_real_time()))
        position_error = lla2ned(estimated_state.position, ground_truth.position)
        north_error, east_error, down_error = position_error

        fmt_strings = (
            f'Progress:\t {progress:g}%',
            f'Sim Time:\t {sim_time}',
            f'Real Time:\t {real_time}',
            f'NED Error:\t [{north_error:g}, {east_error:g}, {down_error:g}]'
        )

        str_lengths = [len(s) for s in fmt_strings]
        self._max_length = max(self._max_length, str_lengths)

        if self._is_using_terminal:
            num_lines = len(fmt_strings)
            clear_terminal_line(num_lines)
            for s in fmt_strings:
                print(s, flush=True)

        else:
            fmt_string = '\t\t'.join(fmt_strings)
            max_length = sum(self._max_length) + 2
            print(f'\r{fmt_string:{max_length}s}', end='', flush=True)

        assert np.linalg.norm(position_error) <= self._max_error, \
            f"Estimated position error has exceeded threshold of {self._max_error} metres"


if __name__ == '__main__':
    pass
