"""
============
realistic.py
============

:summary:
    Provides quantum sensors with realistic implementations.
    Extends from the base quantum sensor and includes realistic
    physics for capturing measurements.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""

from math import pi, sqrt, sin, asin
from qnav.waypoints.trajectory import GroundTruth
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.platform import SensorAxis
from qnav.quantum.base import ConceptQuantumImu
from dataclasses import dataclass

import numpy as np

# Atomic mass (kg) - Caesium 133 (132.90545 amu's)
_CAESIUM_ATOM_MASS = 2.20693925e-25

# Planck's Constant (Joules*seconds, Js)
_H_BAR = 1.054571817e-34


@dataclass(frozen=True)
class ColdAtomInterferometer:
    """
    Parameters used for defining a Cold Atom Interferometer.
    These properties describe the qualities of an ultra-cold atom
    interferometer, which is used for capturing quantum 
    accelerations and angle rates.
    """

    # The number of atoms to use for each measurement
    num_of_atoms: int = 1e6

    # The mass of each of the atoms (in kilograms)
    atom_mass: float = _CAESIUM_ATOM_MASS

    # The length of the interferometer (in metres)
    sensor_length: float = 0.5

    # The interferometer beam width (in metres)
    beam_width: float = 0.01

    # The detection efficiency for measurements (decimal percentage)
    eta: float = 0.3

    # The time between measurement pulses (in seconds)
    time_pulse: float = 0.16

    # The recoil velocity of the cold atoms when absorbing/emitting a photon
    recoil_velocity: float = 7.0e-3


class QuantumIMU(ConceptQuantumImu):
    """
    Quantum IMU that uses an ultra-cold atom interferometer for measurements.
    This extends from the base concept quantum IMU by replacing the acquiring
    of measurement methods with realistic implementations. These feature
    """

    def __init__(self,
                 full_frequency: float,
                 imu_frequency: float,
                 accelerometer_errors: ErrorProperties,
                 gyroscope_errors: ErrorProperties,
                 interferometer: ColdAtomInterferometer,
                 sensor_axis: SensorAxis = SensorAxis(),
                 duty_cycle: float = 0.5,
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Creates a cold-atom interferometer based quantum IMU sensor.

        :param full_frequency: The frequency of quantum measurements. This is
            the rate at which the sensor produces a quantum measurement.
        :type full_frequency: float

        :param imu_frequency: The frequency of shared IMU components. This is
            the rate at which the sensor samples from the ground truth.
            Must be higher than the full frequency.
        :type imu_frequency: float

        :param accelerometer_errors: The accelerometer errors to present in
            the quantum accelerometer.
        :type accelerometer_errors: ErrorProperties

        :param gyroscope_errors: The gyroscope errors to present in
            the quantum gyroscope.
        :type gyroscope_errors: ErrorProperties

        :param interferometer: The interferometer parameters to use.
        :type interferometer: ColdAtomInterferometer

        :param sensor_axis: The quantum IMU sensor axis.
        :type sensor_axis: SensorAxis

        :param duty_cycle: The percentage of time the measurement period
            requires to gain a quantum measurement (between 0.0 and 1.0).
        :type duty_cycle: float

        :param start_time: The starting time of the sensor in seconds.
        :type start_time: float

        :param rand_seed: The random number generation seed.
        :type rand_seed: int
        """

        # Call parent constructor with given arguments:
        super().__init__(full_frequency, imu_frequency,
                         accelerometer_errors, gyroscope_errors,
                         sensor_axis, duty_cycle,
                         start_time, rand_seed)

        # Get the interferometer properties
        self._interferometer = interferometer

        # Also record the clock time of each measurement
        self._clock_record = np.zeros(self._num_active_steps)

    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        """
        Samples from the ground truth to collect data to produce measurements.
        Using the ground truth, captures the current acceleration and angles
        rates at the time. A series of these are required to produce a
        measurement before fusion.

        :param est_time: The estimated time the measurement of the sensor
            is expected to be captured at (in seconds).
        :type est_time: float

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth
        """

        # While within the duty cycle for measurements:
        if self._step < self._num_active_steps:

            # Produce measurements from "quantum IMU sensors":
            self._qs_accelerometer.take_measurement(est_time, ground_truth)
            self._qs_gyroscope.take_measurement(est_time, ground_truth)

            # Record the produced measurements for current time step:
            self._qs_acc_data[self._step, :] = self._qs_accelerometer.last_measurement
            self._qs_gyro_data[self._step, :] = self._qs_gyroscope.last_measurement
            self._clock_record[self._step] = est_time

        # If the final step:
        if self._step == (self._num_total_steps - 1):

            # Get the quantum measurements from the sensor
            self._last_measurement = self._get_measurements()

        # Increment the measurement step
        self._step += 1

    def reset(self):
        """
        Resets the sensor, clearing all previous measurements.
        Normally used directly after fusion.
        """
        self._clock_record[:] = 0
        super().reset()

    def _get_measurements(self) -> tuple[np.ndarray, np.ndarray]:
        """
        When ready, processes the sampled data to produce measurements.
        This takes the sampled quantum measurements and uses the
        interferometer parameters to calculate realistic measurements
        that can be subject to failure under certain conditions.

        :return: The capture accelerometer and gyroscope measurements.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Unpack the interferometer parameters
        num_atoms = self._interferometer.num_of_atoms
        atom_mass = self._interferometer.atom_mass
        recoil_velocity = self._interferometer.recoil_velocity
        t_pulse = self._interferometer.time_pulse
        eta = self._interferometer.eta

        # Effective wave number
        k_eff = atom_mass * recoil_velocity / _H_BAR

        # Obtain the average sensor measurement
        measured_acceleration = np.nanmean(self._qs_acc_data, 0)
        measured_angle_rate = np.nanmean(self._qs_gyro_data, 0)

        # Validate which readings have been successfully obtained.
        x_valid, y_valid, z_valid = self._validate_readings()

        # Calculate phase shifts
        # delta_phi_x, delta_phi_y, delta_phi_z = k_eff * np.mean(
        #     measured_acceleration, axis=0) * t_pulse ** 2

        delta_phi_x, delta_phi_y, delta_phi_z = (
                k_eff * measured_acceleration * t_pulse ** 2)

        # Random phase for each interferometer measurement
        phi_x0, phi_y0, phi_z0 = 2 * pi * self._rng.random(size=3)

        # Measured Signals
        tmp = eta * (num_atoms + self._rng.normal(size=3) * sqrt(num_atoms))
        s_x = tmp[0] * sin(delta_phi_x + phi_x0)
        s_y = tmp[1] * sin(delta_phi_y + phi_y0)
        s_z = tmp[2] * sin(delta_phi_z + phi_z0)

        n_x = round((delta_phi_x + phi_x0) / (2.0 * pi))
        n_y = round((delta_phi_y + phi_y0) / (2.0 * pi))
        n_z = round((delta_phi_z + phi_z0) / (2.0 * pi))

        # Lambda function for calculating angle difference
        # ang_diff = lambda a, b: (((b - a) + pi) % (2 * pi)) - pi

        # Inverted Signals to find measured acceleration values.
        # (Check for arc-sin to be defined sine must be between -1 and 1).
        if x_valid and abs(s_x / eta / num_atoms) <= 1.0:
            acc_x = (ang_diff(asin(s_x / eta / num_atoms), phi_x0) +
                     2.0 * pi * n_x) / (k_eff * t_pulse ** 2)
        else:
            acc_x = np.nan

        if y_valid and abs(s_y / eta / num_atoms) <= 1.0:
            acc_y = (ang_diff(asin(s_y / eta / num_atoms), phi_y0) +
                     2.0 * pi * n_y) / (k_eff * t_pulse ** 2)
        else:
            acc_y = np.nan

        if z_valid and abs(s_z / eta / num_atoms) <= 1.0:
            acc_z = (ang_diff(asin(s_z / eta / num_atoms), phi_z0) +
                     2.0 * pi * n_z) / (k_eff * t_pulse ** 2)
        else:
            acc_z = np.nan

        # Obtain the average acceleration and angle rates
        avg_acceleration = np.array([acc_x, acc_y, acc_z])
        avg_angle_rate = measured_angle_rate

        # Return the calculated measurement vector
        return avg_acceleration, avg_angle_rate

    def _validate_readings(self) -> tuple[bool, bool, bool]:
        """
        Validate that the acquired readings are usable.
        For this a check is performed of the lateral accelerations to ensure
        that no measurement drifts outside the interferometer beam. On
        completion, returns valid status for the X, Y and Z readings.

        :return: Whether the X, Y and Z axis readings are valid, respectively.
        :rtype: tuple[bool, bool, bool]
        """

        # Unpack the interferometer parameters
        delta_x = self._interferometer.beam_width
        delta_z = self._interferometer.sensor_length

        # Obtain the average time_step between measurements
        time_step = np.mean(np.diff(self._clock_record))
        time_step_sq = time_step * time_step

        # The initial position and velocity
        position_s = np.zeros(3)
        velocity_s = np.zeros(3)

        # The valid status for each axis
        x_valid = True
        y_valid = True
        z_valid = True

        # For each quantum acceleration measurement:
        for acceleration_s in self._qs_acc_data:

            # Update the position
            position_s += (velocity_s * time_step +
                           0.5 * acceleration_s * time_step_sq)

            # Update the velocity
            velocity_s += acceleration_s * time_step

            # Calculate the deviations in each axis
            deviation_x, deviation_y, deviation_z = np.abs(position_s)

            # Calculate if any exceed the bounds of the beam
            x_gt_dx = (deviation_x > delta_x)
            x_gt_dz = (deviation_x > delta_z)
            y_gt_dx = (deviation_y > delta_x)
            y_gt_dz = (deviation_y > delta_z)
            z_gt_dx = (deviation_z > delta_x)
            z_gt_dz = (deviation_z > delta_z)

            # Update the valid flags for each axis
            x_valid &= not (x_gt_dz or y_gt_dx or z_gt_dx)
            y_valid &= not (x_gt_dx or y_gt_dz or z_gt_dx)
            z_valid &= not (x_gt_dx or y_gt_dx or z_gt_dz)

        # Finally, return the valid status
        return x_valid, y_valid, z_valid


def ang_diff(a: float, b: float) -> float:
    """
    Calculate the difference between two angles (in radians).

    :param a: The first angle value.
    :type a: float

    :param b: The second angle value.
    :type b: float

    :return: The difference between the given angles.
    :rtype: float
    """
    return (((b - a) + pi) % (2 * pi)) - pi