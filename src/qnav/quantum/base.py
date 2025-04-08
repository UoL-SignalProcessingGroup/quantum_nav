"""
=======
base.py
=======

:summary:
    Provides basic concept quantum sensors and fusion methods.
    These serve as base models that more advanced and realistic
    implementations can extend from.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""

from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.platform import SensorAxis
from qnav.measurement.sensor import Sensor
from qnav.measurement.sensor import SensorFusion
from qnav.measurement.sensor import FusionTrigger
from qnav.estimation.state import EstimatedState
from qnav.quantum.dummy import DummyGyroscope
from qnav.quantum.dummy import DummyAccelerometer
from qnav.waypoints.trajectory import GroundTruth
from qnav.fusion.ins import NumericalINS

import numpy as np


class ConceptQuantumImu(Sensor):
    """
    Concept Quantum IMU featuring use of quantum accelerometer and gyroscope.
    While not realistic in implementation, this base class produces
    measurements mimicking a quantum accelerometer and gyroscope,
    requiring multiple samples of the ground truth.
    """

    def __init__(self,
                 full_frequency: float,
                 imu_frequency: float,
                 accelerometer_errors: ErrorProperties,
                 gyroscope_errors: ErrorProperties,
                 sensor_axis: SensorAxis = SensorAxis(),
                 duty_cycle: float = 0.5,
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Create a basic concept quantum IMU sensor from given configuration.

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

        # Call parent class's constructor
        super().__init__(imu_frequency, start_time, rand_seed)

        # Ensure the duty cycle is suitable:
        if duty_cycle <= 0 or duty_cycle > 1:
            raise ValueError("Duty cycle must be positive between 0 and 1")

        # Ensure the frequency is suitable:
        if full_frequency > imu_frequency:
            raise ValueError("full_frequency cannot be higher than imu_frequency")

        # TODO: Ensure sensor axis are the same?

        self._sensor_axis: SensorAxis = sensor_axis
        self._duty_cycle: float = duty_cycle

        # Create highly-accurate "quantum IMU sensors" reusing IMU classes
        self._qs_accelerometer = Accelerometer(imu_frequency, accelerometer_errors)
        self._qs_gyroscope = Gyroscope(imu_frequency, gyroscope_errors)

        # Obtain the measurement amounts required:
        freq_ratio = imu_frequency / full_frequency
        self._num_total_steps: int = int(freq_ratio)
        self._num_active_steps: int = int(duty_cycle * freq_ratio)

        # TODO: NEW
        self._num_total_steps = max(self._num_total_steps, 1)
        self._num_active_steps = max(self._num_active_steps, 1)

        self._step: int = 0

        # Pre-allocate array for holding measurement data
        self._qs_acc_data = np.zeros([self._num_active_steps, 3])
        self._qs_gyro_data = np.zeros([self._num_active_steps, 3])

        # Define the last measurement produced
        self._last_measurement = None
        self._active_axis = np.ones(3, dtype=bool)

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

        # If the final step:
        if self._step == (self._num_total_steps - 1):

            # Obtain the average sensor measurement
            acceleration = np.nanmean(self._qs_acc_data, 0)
            angle_rates = np.nanmean(self._qs_gyro_data, 0)

            # If any nan values, abort measurement
            # if (not np.any(np.isnan(acceleration)) and
            #         not np.any(np.isnan(angle_rates))):
            self._last_measurement = (acceleration, angle_rates)

        # Increment the measurement step
        self._step += 1

    def update(self, time_sec: float = None):
        """
        Called on demand to update the sensor's state.
        Typically called after measurements have been taken. When called
        this increases the next measurement time and propagates internal
        states, to the time difference.

        :param time_sec: The current simulation time in seconds.
        :type time_sec: float
        """

        # Update using parent class
        super().update(time_sec)

        # Then update the internal sensors
        self._qs_accelerometer.update(time_sec)
        self._qs_gyroscope.update(time_sec)

    def reset(self):
        """
        Resets the sensor, clearing all previous measurements.
        Normally used directly after fusion.
        """
        self._last_measurement = None
        self._qs_acc_data[:] = 0
        self._qs_gyro_data[:] = 0
        self._step = 0

    @property
    def last_measurement(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the last acceleration and angle rate measurements captured
        by the quantum sensor, respectively.

        :return: The last measurements of the sensor.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        return self._last_measurement

    @property
    def num_active_steps(self) -> int:
        """
        Returns the number of measurement steps required,

        :return: Number of measurements required.
        :rtype: int
        """
        return self._num_active_steps

    @property
    def num_steps(self) -> int:
        """
        Returns the number of total steps required,

        :return: Number of steps required for fusion.
        :rtype: int
        """
        return self._num_total_steps

    @property
    def sensor_axis(self) -> SensorAxis:
        """
        Returns the true sensor axis for the sensor.

        :return: The sensor axis used by the sensor.
        :rtype: SensorAxis
        """
        return self._sensor_axis


class ConceptQuantumFusion(SensorFusion):
    """
    A basic version of quantum fusion, namely used as a base class.
    This is the simplest implementation of position correction fusion for
    quantum IMU. Many of its methods can be extended and swapped out for
    more advanced implementations.
    """

    def __init__(self, qs_imu: ConceptQuantumImu,
                 shared_accelerometer: Accelerometer,
                 shared_gyroscope: Gyroscope,
                 est_qs_imu_axis: SensorAxis = None,
                 est_acc_axis: SensorAxis = None,
                 est_gyro_axis: SensorAxis = None):
        """
        Generates a quantum sensor fusion instance for position correction.

        :param qs_imu: The quantum sensor to use for quantum IMU measurements.
        :type qs_imu: ConceptQuantumImu

        :param shared_accelerometer: The IMU's accelerometer instance.
        :type shared_accelerometer: Accelerometer

        :param shared_gyroscope: The IMU's gyroscope instance.
        :type shared_gyroscope: Gyroscope

        :param est_qs_imu_axis: (Optional) Estimated sensor axis for
            quantum IMU. If not provided, true axis will be used.
        :type est_qs_imu_axis: SensorAxis

        :param est_acc_axis: (Optional) Estimated sensor axis for
            accelerometer. If not provided, true axis will be used.
        :type est_acc_axis: SensorAxis

        :param est_gyro_axis: (Optional) Estimated sensor axis for
            gyroscope. If not provided, true axis will be used.
        :type est_gyro_axis: SensorAxis
        """

        # Call parent class's constructor
        super().__init__(
            FusionTrigger.ALL, qs_imu,
            shared_accelerometer, shared_gyroscope)

        # set_or_default = lambda a, b: a if a is not None else b
        #
        # self._qs_est_axis: SensorAxis = set_or_default(
        #     est_qs_imu_axis, qs_imu.sensor_axis)
        #
        # self._acc_est_axis: SensorAxis = set_or_default(
        #     est_acc_axis, shared_accelerometer.sensor_axis)
        #
        # self._gyro_est_axis: SensorAxis = set_or_default(
        #     est_gyro_axis, shared_gyroscope.sensor_axis)

        self._qs_est_axis: SensorAxis = est_qs_imu_axis \
            if est_qs_imu_axis is not None else qs_imu.sensor_axis

        self._acc_est_axis: SensorAxis = est_acc_axis \
            if est_acc_axis is not None else shared_accelerometer.sensor_axis

        self._gyro_est_axis: SensorAxis = est_gyro_axis \
            if est_gyro_axis is not None else shared_gyroscope.sensor_axis

        # Ensure frequencies are valid:
        if (qs_imu.frequency
                != shared_accelerometer.frequency
                != shared_gyroscope.frequency):
            raise ValueError("Sensor frequencies must be identical")

        # Unpack given parameters
        self._qs_imu = qs_imu
        self._accelerometer = shared_accelerometer
        self._gyroscope = shared_gyroscope

        # Initialise steps
        self._step = 0
        self._num_steps = self._qs_imu.num_steps

        # Set-up measurement table
        self._imu_acc_data = np.zeros([self._num_steps, 3])
        self._imu_gyro_data = np.zeros([self._num_steps, 3])

        # Pre-allocate last estimated state
        self._estimated_state = None

    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Collects measurements and applies position fixing.
        Each cycle collects measurements from the typical IMU sensors and
        quantum sensor. After a set number of calls, this uses the single
        quantum measurement to calculate an approximate bias correction for
        the IMU. If successful, the position fixing is applied.

        :param estimated_state: The current estimated state to update.
        :type estimated_state: EstimatedState
        """

        # Get latest measurements from sensors
        qs_measurement = self._qs_imu.last_measurement
        acc_measurement = self._accelerometer.last_measurement
        ang_measurement = self._gyroscope.last_measurement

        # On the first step:
        if self._step == 0:

            # Preserve the current estimated state.
            # self._estimated_state = copy(estimated_state)
            self._estimated_state = estimated_state.clone()

        # Record latest IMU measurements:
        self._imu_acc_data[self._step, :] = acc_measurement
        self._imu_gyro_data[self._step, :] = ang_measurement

        # On the final step:
        if self._step == (self._num_steps - 1):

            # If successful qs_measurement:
            if qs_measurement is not None:

                # Estimate sensor biases using sensor measurement differences
                qs_measurement_imu = self._measurement_to_imu_axis(*qs_measurement)
                acc_bias, gyro_bias = self._calculate_bias_errors(*qs_measurement_imu)

                # Apply correction using corrected IMU measurements
                self._perform_correction(acc_bias, gyro_bias, estimated_state)

            # Reset sensor and measurement table
            self.reset()

        else:
            self._step += 1


    def reset(self):
        """
        Resets the state of the fusion, clearing internal temporary states.
        When call the fusion will return to expecting the first measurement,
        then continuing from there.
        """
        self._qs_imu.reset()
        self._step = 0


    def _measurement_to_imu_axis(self, acceleration_qs,
                                 angle_rate_qs) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts quantum measurements to assumed IMU axes.
        A requirement for comparing quantum IMU values against standard IMU
        values, by converting them to a common axes.

        :param acceleration_qs: The measured acceleration from the quantum sensor.
        :type acceleration_qs: np.ndarray (3-elements)

        :param angle_rate_qs:
        :type angle_rate_qs: np.ndarray (3-elements)

        :return: The given quantum measurements converted to IMU axes.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Unpack the lever arm values
        qs_lever_arm = self._qs_est_axis.lever_arm
        acc_lever_arm = self._acc_est_axis.lever_arm

        # Unpack the sensor rotation matrix values
        qs_rot_body2sensor = self._qs_est_axis.body2sensor_mat
        acc_rot_body2sensor = self._acc_est_axis.body2sensor_mat
        gyro_rot_body2sensor = self._gyro_est_axis.body2sensor_mat

        # Obtain the measured acceleration and angle rate.
        # acceleration_qs, angle_rate_qs = qs_measurement
        angle_rate_qs = np.radians(angle_rate_qs)

        # Convert acceleration from quantum sensors to body axes
        # (includes the lever arm correction).
        acceleration_qs_body = np.linalg.solve(
            qs_rot_body2sensor, acceleration_qs) - np.cross(
            angle_rate_qs, np.cross(angle_rate_qs, qs_lever_arm))

        # Convert acceleration from body axes to IMU sensor axes
        # (includes the lever arm correction).
        acceleration_qs_imu = acc_rot_body2sensor.dot(
            acceleration_qs_body + np.cross(angle_rate_qs, np.cross(
                angle_rate_qs, acc_lever_arm)))

        # Rotate angle rates from body axes to IMU axes.
        angle_rate_qs_imu = gyro_rot_body2sensor.dot(angle_rate_qs)
        angle_rate_qs_imu = np.degrees(angle_rate_qs_imu)  # TODO: NEW

        return acceleration_qs_imu, angle_rate_qs_imu

    def _calculate_bias_errors(self, acceleration_qs_imu,
                               angle_rate_qs_imu) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates and returns the accelerometer and gyroscope bias.
        Using the collected IMU measurements and calculated quantum IMU
        measurements, this method examines the difference and returns the
        estimated bias for each.

        :param acceleration_qs_imu: The quantum acceleration measurement
            (given in converted IMU axis).

        :param angle_rate_qs_imu: The quantum gyroscope measurement
            (given in converted IMU axis).

        :return: The estimated accelerometer and gyroscope biases, respectively.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Obtain the average acceleration value from the measurement record.
        mean_imu_acceleration = np.mean(self._imu_acc_data, 0)

        # Obtain the average angle rate value from the measurement record.
        mean_imu_angle_rate = np.mean(self._imu_gyro_data, 0)

        # Convert angle rates to radians.
        # mean_imu_angle_rate = np.radians(mean_imu_angle_rate)
        # angle_rate_qs_imu = np.radians(angle_rate_qs_imu)

        # Calculate estimated bias values for IMU.
        accelerometer_biases = mean_imu_acceleration - acceleration_qs_imu
        gyroscope_biases = (mean_imu_angle_rate - angle_rate_qs_imu)
        # gyroscope_biases = np.degrees(gyroscope_biases)

        # Replace nan values with zero (prevent numerical errors).
        accelerometer_biases = np.nan_to_num(accelerometer_biases)
        gyroscope_biases = np.nan_to_num(gyroscope_biases)

        # Return the calculated bias errors
        return accelerometer_biases, gyroscope_biases

    def _perform_correction(self, acc_bias: np.ndarray,
                            gyro_bias: np.ndarray,
                            estimated_state: EstimatedState):
        """
        Reprocesses IMU solution using estimated biases and correct for
        biases in measurement record.

        :param acc_bias: The calculated acceleration biases for correction.
        :type acc_bias: np.ndarray

        :param gyro_bias: The calculated angle rate biases for correction.
        :type gyro_bias: np.ndarray

        :param estimated_state: The current estimated state to be updated.
        """

        # TODO: Remove or improve
        if np.any(np.isnan(acc_bias)) or np.any(np.isnan(gyro_bias)):
            print("Rejecting correction and resetting...\n")
            return

        # Correct for biases in measurement record
        acc_data = self._imu_acc_data - acc_bias
        gyro_data = self._imu_gyro_data - gyro_bias

        # acc_data = self._imu_acc_data
        # gyro_data = self._imu_gyro_data

        tmp_state: EstimatedState = self._estimated_state

        dummy_accelerometer = DummyAccelerometer(self._accelerometer)
        dummy_gyroscope = DummyGyroscope(self._gyroscope)
        # dummy_accelerometer.next_update = timestep
        # dummy_gyroscope.next_update = timestep

        tmp_ins = NumericalINS(dummy_accelerometer, dummy_gyroscope)

        # TODO: Fix depending on fusion ordering!
        # for i in range(self._num_steps):
        # for i in range(self._num_steps - 1):
        for i in range(1, self._num_steps):
            dummy_accelerometer.last_measurement = acc_data[i, :]
            dummy_gyroscope.last_measurement = gyro_data[i, :]
            tmp_ins.perform_fusion(tmp_state)

        # Finally, overwrite the current estimates
        estimated_state.update_estimates(
            position=tmp_state.position,
            velocity=tmp_state.velocity,
            acceleration=tmp_state.acceleration,
            attitude=tmp_state.attitude,
            angle_rates=tmp_state.angle_rates
        )

    @property
    def sensors(self) -> (ConceptQuantumImu, Accelerometer, Gyroscope):
        return self._qs_imu, self._accelerometer, self._gyroscope

