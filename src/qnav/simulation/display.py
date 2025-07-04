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
from enum import IntEnum, auto
from sys import stdin

from qnav.estimation.state import EstimatedState
from qnav.waypoints.trajectory import GroundTruth
from qnav.waypoints.trajectory import Trajectory
from qnav.util.transformations import lla2ned

import numpy as np
import multiprocessing as mp

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

    __slots__ = ('_start_time', '_end_time')

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


class DisplayStatus(IntEnum):
    """
    Progress states used by some displayers.
    Used to indicate the current progress of simulations.
    """
    NOT_STARTED = auto(),
    STARTING = auto(),
    RUNNING = auto(),
    FINISHED = auto(),
    FAILURE = auto()


class NoDisplay(Display):
    """
    A displayer that retains progress, but does not display or print it.
    Useful when automating simulations or running them silently. As this
    displayer still holds progress it can be monitored by other methods.
    """

    __slots__ = ('_start_time', '_end_time', '_status')

    def __init__(self, update_interval: float):
        """
        Initialise console printing by giving the update time interval.
        This defines how often displaying updates will be given (in
        simulation time).

        :param update_interval: The time intervals between updates in seconds.
        :type update_interval: float
        """

        # Call construct and typical initialisation
        super().__init__(update_interval)
        self._update_interval = update_interval
        self._start_time = float('nan')
        self._end_time = float('nan')

        # Record default status and progress percentage
        self._status = DisplayStatus.NOT_STARTED
        self._progress = 0.0

    def on_start(self, trajectory: Trajectory):
        """
        Called at the start of the simulation before first iteration.
        Used to obtain the start and end times of the trajectory.

        :param trajectory: The ground truth trajectory instance.
        :type trajectory: Trajectory
        """
        self._start_time = trajectory.start_time
        self._end_time = trajectory.end_time
        self._status = DisplayStatus.RUNNING

    def on_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called during the simulation after each fusion update.
        Updates the runtime progress of the simulation.

        :param timestamp: The current (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The current estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        self._progress = timestamp / self._end_time

    def on_finish(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called at the end of the simulation after final iteration.
        Marks simulation as complete

        :param timestamp: The final (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The final estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The final ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        self._progress = 1.0
        self._status = DisplayStatus.FINISHED

    @property
    def status(self) -> DisplayStatus:
        """
        The current display status of the simulation.
        Can be NOT_STARTED, STARTED, FINISHED, or FAILURE.

        :return: The current display status of the simulation.
        :rtype: DisplayStatus
        """
        return self._status

    @status.setter
    def status(self, new_status: DisplayStatus):
        """
        Set the current display status of the simulation.
        Supports NOT_STARTED, STARTED, FINISHED, or FAILURE.

        :param new_status: The new display status of the simulation.
        :type new_status: DisplayStatus
        """
        self._status = new_status

    @property
    def progress(self) -> float:
        """
        The current simulation progress percentage.

        :return: The simulation percentage between 0.0 and 1.0.
        :rtype: float
        """
        return self._progress

    @property
    def update_interval(self) -> float:
        """
        The requested update interval in simulated seconds.

        :return: The would be print delay in simulation seconds.
        :rtype: float
        """
        return self._update_interval


class MPDisplay(Display):
    """
    A displayer for multi-process simulations.
    This displayers is used when multiple simulations are being executed in
    parallel, where each holds the information for an individual simulation.
    """

    def __init__(self, update_interval: float,
                 simulation_index: int,
                 shared_progress: mp.RawArray,
                 shared_status: mp.RawArray):
        """
        Initialise console printing by giving the update time interval.
        This defines how often displaying updates will be given (in
        simulation time).

        :param update_interval: The time intervals between updates in seconds.
        :type update_interval: float
        """

        # Call construct and typical initialisation
        super().__init__(update_interval)
        self._update_interval = update_interval
        self._start_time = float('nan')
        self._end_time = float('nan')

        # Add additional fields
        self._progress: float = 0.0
        self._status = DisplayStatus.NOT_STARTED

        # Copy the index and shared progress arrays
        self._simulation_index = simulation_index
        self._shared_progress_array = shared_progress
        self._shared_status_array = shared_status

    def on_start(self, trajectory: Trajectory):
        """
        Called at the start of the simulation before first iteration.
        Used to obtain the start and end times of the trajectory.

        :param trajectory: The ground truth trajectory instance.
        :type trajectory: Trajectory
        """
        self._start_time = trajectory.start_time
        self._end_time = trajectory.end_time
        self.status = DisplayStatus.RUNNING

    def on_update(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called during the simulation after each fusion update.
        Updates the runtime progress of the simulation.

        :param timestamp: The current (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The current estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        self.progress = timestamp / self._end_time

    def on_finish(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Called at the end of the simulation after final iteration.
        Marks simulation as complete

        :param timestamp: The final (true) simulation time in seconds.
        :type timestamp: float

        :param estimated_state: The final estimated state of the simulation.
        :type estimated_state: EstimatedState

        :param ground_truth: The final ground truth state of the simulation.
        :type ground_truth: GroundTruth
        """
        self.progress = 1.0
        self.status = DisplayStatus.FINISHED

    @property
    def status(self) -> DisplayStatus:
        """
        The current display status of the simulation.
        Enumerator making current run status of simulation.

        :return: The current display status of the simulation.
        :rtype: DisplayStatus
        """
        return self._status

    @status.setter
    def status(self, new_status: DisplayStatus):
        """
        Controlled setting of the simulation display status.
        Updates the local flag and shared status.

        :param new_status: The new display status of the simulation.
        :type new_status: DisplayStatus
        """
        self._status = new_status
        ind = self._simulation_index
        self._shared_status_array[ind] = new_status.value

    @property
    def progress(self) -> float:
        """
        The current progress percentage of the simulation.
        A progress percentage between 0.0 and 1.0.

        :return: The current progress percentage of the simulation.
        :rtype: float
        """
        return self._progress

    @progress.setter
    def progress(self, new_progress: float):
        """
        Controlled setting of the simulation progress.
        Updates the local progress and shared percentage.

        :param new_progress: The new progress percentage of the simulation.
        :type new_progress: float
        """
        self._progress = new_progress
        ind = self._simulation_index
        self._shared_progress_array[ind] = new_progress


if __name__ == '__main__':
    pass
