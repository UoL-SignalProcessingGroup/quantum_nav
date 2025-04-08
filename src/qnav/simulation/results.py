"""
==========
results.py
==========

:summary:
    Responsible for the capturing and retaining of simulation results.
    This module provides templates and classes that are responsible for
    capturing results during simulations, maintaining them and presenting
    them for collection afterward. Each class can capture results in
    different fashions, incorporating filters or conditions for
    recording and/or applying post-processing before collection.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation.
"""

import numpy as np

from math import ceil
from copy import copy
from abc import ABC
from abc import abstractmethod
from tempfile import NamedTemporaryFile

from qnav.output.data import ResultsTable
from qnav.waypoints.trajectory import Trajectory
from qnav.waypoints.trajectory import GroundTruth
from qnav.estimation.state import EstimatedState

# TODO: Better initialisation/preparation


class ResultsCapture(ABC):
    """
    Abstract definition for classes used to capture simulation results.
    This abstract class defines the methods other classes must implement for
    the collecting of simulation results. These results consist of a series
    of timestamps, estimated states and ground truths. The primary purpose
    of this abstract class is to allow for flexibility when capturing results.
    """

    @abstractmethod
    def on_start(self, init_estimate: EstimatedState, trajectory: Trajectory):
        """
        Prepares the results capture giving initial estimate and trajectory.
        Uses the initial estimated state and trajectory information to
        initialise a results capture for the given trajectory.

        :param init_estimate: The initial estimate state (at start time).
        :type init_estimate: EstimatedState

        :param trajectory: The trajectory containing ground truth states.
        :type trajectory: Trajectory
        """
        pass

    @abstractmethod
    def record(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Records results containing current time, estimated state and ground.
        For given timestamp record the current estimated state and ground
        truth information. The given information can then be recorded,
        subject to any filtering or conditions implemented.

        :param timestamp: The (true) simulation time (in seconds).
        :type timestamp: float

        :param estimated_state: The current estimated states.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth record.
        :type ground_truth: GroundTruth
        """
        pass

    @abstractmethod
    def gather(self) -> any:
        """
        Gathers and returns all records collected so far.
        This formats and returns all recorded results in a common data
        structure. This is primarily used at the end of simulations to
        provide all output.

        :return: All results captured during simulation.
        :rtype: any
        """
        pass

    @abstractmethod
    def save(self,
             estimation_table: ResultsTable,
             ground_truth_table: ResultsTable = None):
        pass


class CaptureAll(ResultsCapture):
    """
    A simple class:`ResultsCapture` that collects everything.
    Unsuitable for typical simulations, but ideal when debugging and
    examining every update.
    """

    # Used to reduce memory and improve access time.
    __slots__ = '_data'

    def __init__(self):
        """
        Creates a capture all instance.
        """
        self._data = []

    def on_start(self, init_estimate: EstimatedState, trajectory: Trajectory):
        """
        Prepares the results capture giving initial estimate and trajectory.
        Uses the initial estimated state and trajectory information to
        initialise a results capture for the given trajectory.

        :param init_estimate: The initial estimate state (at start time).
        :type init_estimate: EstimatedState

        :param trajectory: The trajectory containing ground truth states.
        :type trajectory: Trajectory
        """
        self._data = []

    def record(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Records results containing current time, estimated state and ground.
        Simply appends the data to collection without filtering.

        :param timestamp: The (true) simulation time (in seconds).
        :type timestamp: float

        :param estimated_state: The current estimated states.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth record.
        :type ground_truth: GroundTruth
        """

        estimated_data = copy(estimated_state)
        ground_truth_data = copy(ground_truth)
        self._data.append((timestamp, estimated_data, ground_truth_data))

    def gather(self) -> list[tuple[float, EstimatedState, GroundTruth]]:
        """
        Gathers and returns all records collected so far.
        Simply returns a list containing a tuple for each record collected.

        :return: All results captured during simulation.
        :rtype: list[tuple[float, EstimatedState, GroundTruth]]
        """
        return self._data

    def save(self, estimations: ResultsTable, ground_truth: GroundTruth = None):

        # TODO: Implement or adjust base class
        raise NotImplementedError("Not implemented for class")


def _write_data(table_data: np.ndarray, description: str, results: ResultsTable):
    estimated_data = {
        'time_steps': table_data[:, 0],
        'position': table_data[:, 1:4],
        'velocity': table_data[:, 4:7],
        'acceleration': table_data[:, 7:10],
        'attitude': table_data[:, 10:13],
        'angle_rates': table_data[:, 13:16]
    }

    results.write(description, **estimated_data)


class CaptureAtFixedRate(ResultsCapture):
    """
    A results capturer for collecting results at fixed rate only.
    Efficiently collects results at fixed rate and pre-allocates memory.
    Supplies a better means of handling results at low memory consumption.
    """

    def __init__(self, frequency: float, use_virtual_memory: bool = False, est_data_only: bool = False):
        """
        Creates a fixed rate results capture instance.

        :param frequency: The frequency at which data is to be captured.
        :type frequency: float

        :param use_virtual_memory: Should temporary files be used for caching.
            If set to true, virtual memory will be used for holding results.
        :type use_virtual_memory: bool

        :param est_data_only: If only estimated data is to be recorded.
        :type est_data_only: bool
        """

        self._frequency = frequency
        self._time_step = 1 / frequency
        self._time_data = np.empty(0)
        self._est_data = np.empty(0)
        self._gt_data = np.empty(0)
        self._est_only = est_data_only
        self._num_rows = 0
        self._last_ind = -1

        self._est_data_mem_map = None
        self._gt_data_mem_map = None

        if use_virtual_memory is not None:
            self._time_data_mem_map = NamedTemporaryFile(prefix='time_data_mem_',  delete=False)
            self._est_data_mem_map = NamedTemporaryFile(prefix='est_data_mem_',  delete=False)
            self._gt_data_mem_map = NamedTemporaryFile(prefix='gt_data_mem_', delete=False)


    def __del__(self):
        """
        Class destructor used for clean-up when object is delete.
        Called by Python garbage collection (cannot be guaranteed)!
        """

        # TODO: IMPROVE!
        if self._time_data_mem_map is not None:
            self._time_data_mem_map.close()

        if self._est_data_mem_map is not None:
            self._est_data_mem_map.close()

        if self._gt_data_mem_map is not None:
            self._gt_data_mem_map.close()


    def on_start(self, init_estimate: EstimatedState, trajectory: Trajectory):
        """
        Prepares the results capture giving initial estimate and trajectory.
        Uses the initial estimated state and trajectory information to
        initialise a results capture for the given trajectory.

        :param init_estimate: The initial estimate state (at start time).
        :type init_estimate: EstimatedState

        :param trajectory: The trajectory containing ground truth states.
        :type trajectory: Trajectory
        """

        end_time = trajectory.end_time
        start_time = trajectory.start_time
        self._num_rows = ceil((end_time - start_time) / self._time_step) + 1

        if self._time_data_mem_map is None:
            self._time_data = np.zeros(self._num_rows)
        else:
            self._time_data = np.memmap(
                self._time_data_mem_map, dtype='float64',
                mode='w+', shape=self._num_rows)
        self._time_data += np.nan

        est_record = init_estimate.as_numpy()
        est_size = (self._num_rows, len(est_record))

        if self._est_data_mem_map is None:
            self._est_data = np.zeros(est_size)
        else:
            self._est_data = np.memmap(
                self._est_data_mem_map, dtype='float64',
                mode='w+', shape=est_size)
        self._est_data += np.nan

        if not self._est_only:
            gt_record = trajectory.get_record(start_time).as_numpy()
            gt_size = (self._num_rows, len(gt_record))

            if self._gt_data_mem_map is None:
                self._gt_data = np.zeros(gt_size)
            else:
                self._gt_data = np.memmap(
                    self._gt_data_mem_map, dtype='float64',
                    mode='w+', shape=gt_size)
            self._gt_data += np.nan

        # Record the first record on start-up!
        self.record(start_time, init_estimate, trajectory.get_record(start_time))

    def record(self, timestamp: float, estimated_state: EstimatedState, ground_truth: GroundTruth):
        """
        Records results containing current time, estimated state and ground.
        Simply appends the data to collection without filtering.

        :param timestamp: The (true) simulation time (in seconds).
        :type timestamp: float

        :param estimated_state: The current estimated states.
        :type estimated_state: EstimatedState

        :param ground_truth: The current ground truth record.
        :type ground_truth: GroundTruth
        """

        # If index not already covered:
        ind = self.__time_to_index(timestamp)
        if ind > self._last_ind:

            ind_diff = ind - self._last_ind
            self._last_ind = ind

            if ind_diff > 1:
                ind = np.arange(ind - ind_diff, ind) + 1

            self._time_data[ind] = timestamp
            self._est_data[ind, :] = estimated_state.as_numpy()

            if not self._est_only:
                self._gt_data[ind, :] = ground_truth.as_numpy()

    def gather(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Gathers and returns all records collected so far.
        Simply returns a tuple containing the estimation and ground truth
        results, respectively. The latter will be set to None if

        :return: All results captured during simulation.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        return self._est_data, self._gt_data

    def save(self, estimation_table: ResultsTable, ground_truth_table: ResultsTable = None):

        _write_data(self._est_data,
            "Estimation",
            estimation_table)

        if self._est_only or ground_truth_table is None:
            return

        _write_data(self._gt_data,
            "Ground Truth",
            ground_truth_table)


    def __time_to_index(self, timestamp: float) -> int:

        ind = int(round(timestamp / self._time_step))
        return max(0, min(ind, self._num_rows))

        # return int(timestamp // self._time_step)
