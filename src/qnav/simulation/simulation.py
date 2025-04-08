"""
=============
simulation.py
=============

:summary:
    The core harness responsible for coordinating simulations.
    This module contains the core functionalities for managing simulations.
    This mostly revolves around a core loop and the scheduling of sensor
    time intervals, sensor measurements and data fusion.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation.
"""

from typing import Collection

from qnav.estimation.state import EstimatedState
from qnav.measurement.clock import Clock, DummyClock
from qnav.measurement.sensor import SensorFusion
from qnav.simulation.holonomic import HolonomicConstraint
from qnav.simulation.scheduling import TimingHandler
from qnav.simulation.results import ResultsCapture
from qnav.simulation.results import CaptureAll
from qnav.simulation.display import ConsoleDisplay
from qnav.simulation.display import Display
from qnav.waypoints.trajectory import Trajectory

# TODO: Make simulation class with run(), resume(), stop(), results(), etc..

def run_simulation_loop(
        trajectory: Trajectory,
        estimated_state: EstimatedState,
        hardware: Collection[SensorFusion],
        clock: Clock = DummyClock(),
        results: ResultsCapture = CaptureAll(),
        displayer: Display = ConsoleDisplay(2.5)):
    """
    Executes the full simulation loop from start to finish.
    On succession returns states and results in requested format.

    :param trajectory: The trajectory to draw ground truth records from.
        This is what the virtual platform is to follow.
    :type trajectory: Trajectory

    :param estimated_state: The initial estimated state. This will be
        updated throughout the simulation via fusion.
    :type estimated_state: EstimatedState

    :param hardware: A collection of SensorFusion instances. This represents
        the hardware and fusion algorithms attached to the virtual platform.
    :type hardware: Collection[SensorFusion]

    :param clock: A virtual clock added to simulate clock errors.
        This is used when virtual clock drift is required.

    :param results: The results capture instance used to manage result data.
        This controls how and what data is collected throughout the simulation.

    :param displayer: The displayer instance used for displaying progress.
        This presents a display of the ongoing simulation allowing for
        immediate feedback of its progress.
    """

    # Get the time frame of the trajectory:
    current_time: float = trajectory.start_time
    end_time: float = trajectory.end_time

    # Create a scheduler for sensors and fusion methods:
    scheduler = TimingHandler(hardware)
    scheduler.update_sensors_past_time(current_time)
    sensor_sequence = scheduler.sensor_iterator(end_time)

    # Prepare the results and displaying instance:
    results.on_start(estimated_state, trajectory)
    displayer.on_start(trajectory)

    # For the next sensor in the sequence:
    for sensor in sensor_sequence:

        # Get current and estimated time:
        estimated_time = sensor.next_update
        current_time = clock.get_time(estimated_time)

        # Prevent going out of bounds of trajectory:
        if current_time > end_time:
            break

        # Remove clock drift for holonomic constraints:
        if isinstance(sensor, HolonomicConstraint):
            current_time = estimated_time

        # Obtain the current sensor's measurement:
        ground_truth = trajectory.get_record(current_time)
        sensor.take_measurement(estimated_time, ground_truth)

        # Update sensor errors:
        sensor.update(current_time)
        clock.update(current_time)

        # Mark that the sensor has been used:
        scheduler.set_sensor_used(sensor)

        # After measurement perform fusion when available:
        for fusion in scheduler.get_next_fusion(sensor):

            # Update estimated state:
            fusion.perform_fusion(estimated_state)

            # Update estimated time (assuming not be updated by fusion)
            estimated_state.update_estimates(timestamp=estimated_time)

            # Record results and update the display:
            results.record(current_time, estimated_state, ground_truth)
            displayer.on_update(current_time, estimated_state, ground_truth)

    # Update the display with final update:
    ground_truth = trajectory.get_record(current_time)
    displayer.on_finish(current_time, estimated_state, ground_truth)

    # Finally, return the collected results.
    return results.gather()

