"""
===================
gravity_gradient.py
===================

:summary:
    Provides implementation for Quantum gravity gradient sensing and fusion.
    Implementation is mainly based on Peters et al, 'High precision gravity
    measurements using atom interferometry', Metrologia, Vol.38,
    pp.25-61 (2001).

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implementation of class.
"""


import math
import numpy as np

import qnav.util.transformations as trans

from dataclasses import dataclass
from qnav.estimation.state import EstimatedState
from qnav.gravity.base import GravityModel
from qnav.quantum.particle_filter import weighted_cov
from qnav.util.transformations import cross_prod_xy
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.platform import SensorAxis
from qnav.measurement.sensor import Sensor, SensorFusion, FusionTrigger
from qnav.waypoints.trajectory import GroundTruth

# Atomic mass (kg) - Rubidium-87 (86.909184 amu s)
_RUBIDIUM_ATOM_MASS = 1.4431608262e-25

# Planck's Constant (Joules*seconds, Js)
_H_BAR = 1.054571817e-34


@dataclass(frozen=True)
class GravityGradInterferometer:
    """
    Parameters used by the Cold Atom Gravity Gradient Interferometer.
    These properties describe the qualities of an ultra-cold atom
    interferometer, which is used for capturing gravity gradient
    estimations from the local environment.
    """

    # TODO: Merge with ColdAtomInterferometer class?

    # The number of atoms to use for each measurement
    num_of_atoms: int = 200000

    # The mass of each of the atoms (in kilograms)
    atom_mass: float = _RUBIDIUM_ATOM_MASS

    # The recoil velocity of the cold atoms when absorbing/emitting a photon
    recoil_velocity: float = 5.8845e-3 # 7.0e-3

    # Average vertical velocity of the atoms during the interferometer cycle
    # when the system is stationary (m/s)
    stationary_velocity: float = 1.57

    # First height offset of interferometer sensor (m)
    sensor_z_top: float = 0.0

    # Second height offset of interferometer sensor (m)
    sensor_z_bottom: float = -1.0

    # The detection efficiency for measurements (decimal percentage)
    eta: float = 0.25

    # The time between measurement pulses (in seconds)
    time_pulse: float = 0.2

    # Relative phase noise for between paired interferometers
    sigma_phi: float = 25e-3

    # Relative scale of white noise for each interferometers.
    sigma_n: float = 1.0

    # Interferometer beam width (metres)
    beam_width: float = 0.01

    # Length of interferometer (metres)
    sensor_length: float = 1.5

    # The probability for purely random measurement failure
    rand_failure_prob: float = 0.0

    def __post_init__(self):
        """
        Performs validation of properties after initialisation.
        Ensure property values supplied are valid and usable.
        """
        if self.sensor_z_top <= self.sensor_z_bottom:
            raise ValueError('sensor_z_top must be larger than sensor_z_bottom')

    @property
    def k_eff(self) -> float:
        """
        Effective wave number, the number of wave cycles per unit distance.
        A property provided for a value commonly calculated.

        :return: A measure of the spatial frequency of a wave.
        :rtype: float
        """
        return self.atom_mass * self.recoil_velocity / _H_BAR


class GravityGradiometer(Sensor):
    """
    Quantum Gravity Gradiometer Sensor for estimating vertical gravity gradient.
    A Quantum sensor that features two ultra-cold atom accelerometers, measuring
    the change in the vertical gravity gradient.
    """
    def __init__(self,
                 full_frequency: float,
                 imu_frequency: float,
                 accelerometer_errors: ErrorProperties,
                 interferometer: GravityGradInterferometer,
                 gravity_model: GravityModel,
                 sensor_axis: SensorAxis = SensorAxis(),
                 duty_cycle: float = 0.5,
                 start_time: float = 0.0,
                 rand_seed: int = None):
        """
        Creates a gravity gradiometer sensor instance.

        :param full_frequency: The full frequency for complete measurements
            to be produced at (in Hz).
        :type full_frequency: float

        :param imu_frequency: The frequency of IMU measurements (in Hz)
        :type imu_frequency: float

        :param accelerometer_errors: The error profile for the internal
            accelerometers of the interferometer.
        :type accelerometer_errors: ErrorProperties

        :param interferometer: The interferometer parameters to use.
        :type interferometer: GravityGradInterferometer

        :param gravity_model: The gravity model to use to acquire real
            gravity gradient measurements from.
        :type gravity_model: GravityModel

        :param sensor_axis: The sensor axis of the sensor.
        :type sensor_axis: SensorAxis

        :param duty_cycle: The decimal percentage of time the sensor samples
            from the ground truth to collect a measurement.
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
            raise ValueError('Duty cycle must be positive between 0 and 1')

        # Ensure the frequency is suitable:
        if full_frequency > imu_frequency:
            raise ValueError('full_frequency cannot be higher than imu_frequency')

        # Obtain the measurement amounts required:
        freq_ratio = imu_frequency / full_frequency
        self._num_total_steps: int = int(freq_ratio)
        self._num_active_steps: int = int(duty_cycle * freq_ratio)
        self._step: int = 0

        # Get the interferometer properties
        self._interferometer = interferometer
        self._gravity_model = gravity_model
        self._sensor_axis = sensor_axis

        # Get the internal accelerometer sensor axes
        z_top_axis = _offset_vertical_axis(sensor_axis, interferometer.sensor_z_top)
        z_btm_axis = _offset_vertical_axis(sensor_axis, interferometer.sensor_z_bottom)

        # Initialise internal accelerometer sensors
        self._qs_top_accelerometer = Accelerometer(imu_frequency, accelerometer_errors, z_top_axis)
        self._qs_btm_accelerometer = Accelerometer(imu_frequency, accelerometer_errors, z_btm_axis)

        # Pre-allocate array for holding measurement data
        self._measurement_data = np.zeros([self._num_active_steps, 6])
        self._acceleration_data = np.zeros([self._num_active_steps, 6])

        # Define the last measurement produced
        self._last_measurement = None

    def take_measurement(self, est_time: float, ground_truth: GroundTruth):
        """
        Samples from the ground truth to collect data to produce measurements.
        Using the current ground truth, captures the

        :param est_time: The estimated time the measurement of the sensor
            is expected to be captured at (in seconds).
        :type est_time: float

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth
        """

        # While within the duty cycle for measurements:
        if self._step < self._num_active_steps:

            # Produce gravity gradient values from ground truth
            gravity_grads = self._get_gravity_gradients(ground_truth)

            # Obtain the local sensor accelerations and measurements for each
            accelerations = self._gravity_to_sensor_axis(ground_truth, *gravity_grads)
            measurements = self._gravity_to_measurements(ground_truth, *gravity_grads)

            # Record each for the current step
            self._acceleration_data[self._step, 0:3] = accelerations[0]
            self._acceleration_data[self._step, 3:6] = accelerations[1]
            self._measurement_data[self._step, 0:3] = measurements[0]
            self._measurement_data[self._step, 3:6] = measurements[1]

        # If the final step:
        if self._step == (self._num_total_steps - 1):

            # Get the quantum measurements from the sensor
            self._last_measurement = self._get_measurement()
            # TODO: Use random failure chance?

        # Increment the measurement step
        self._step += 1

    def reset(self):
        """
        Resets the sensor, clearing all previous measurements.
        Normally used directly after fusion.
        """
        self._last_measurement = None
        self._measurement_data[:] = np.nan
        self._acceleration_data[:] = np.nan
        self._step = 0

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
        self._qs_top_accelerometer.update(time_sec)
        self._qs_btm_accelerometer.update(time_sec)

    def _get_gravity_gradients(self, ground_truth: GroundTruth) -> tuple[np.ndarray, np.ndarray]:
        """
        Extracts partial gravity gradient accelerations from ground truth.
        Given a current ground truth record, this calculates the current
        gravity gradient for both the top and bottom quantum accelerometers. On
        succession two sets of values are produced. A series of these
        measurements are later collected to form realistic measurements.

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth

        :return: The partial gravity gradient accelerations,
            for top and bottom accelerometers respectively.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Obtain the true gravity gradient
        position = ground_truth.position
        dgz_dz = self._gravity_model.calc_vertical_grad(*position)

        # Rotate from body to earth axis
        attitude_rad = np.radians(ground_truth.attitude)
        rot_earth2body = trans.rotate_3d(*attitude_rad)
        gravity_top = np.linalg.solve(rot_earth2body, ground_truth.acceleration)
        gravity_btm = np.copy(gravity_top)

        # Add vertical gradient term for each acceleration
        gravity_top += dgz_dz * self._interferometer.sensor_z_top
        gravity_btm += dgz_dz * self._interferometer.sensor_z_bottom

        # Rotate back from earth to body
        gravity_top = rot_earth2body @ gravity_top
        gravity_btm = rot_earth2body @ gravity_btm
        return gravity_top, gravity_btm


    def _gravity_to_sensor_axis(self, ground_truth: GroundTruth, gravity_top: np.ndarray,
                                gravity_btm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts partial gravity gradients values from body to sensor axis.
        On succession, aligns and returns the gravity gradient measurements
        with the top and bottom accelerometer axes. Used for calculating the
        deviation of atoms in the interferometer.

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth

        :param gravity_top: The gravity acceleration at the top accelerometer.
        :type gravity_top: np.ndarray

        :param gravity_btm: The gravity acceleration at the bottom accelerometer.
        :type gravity_btm: np.ndarray

        :return: The gravity acceleration aligned with accelerometer axes,
            for top and bottom accelerometers respectively.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Unpack the true angle rates
        angle_rates = ground_truth.angle_rates

        # Obtain the sensor axis for each sensor
        top_axis = self._qs_top_accelerometer.sensor_axis
        btm_axis = self._qs_btm_accelerometer.sensor_axis

        # Obtain the angle rate acceleration for top and bottom sensors
        angle_rates_top = cross_prod_xy(angle_rates, cross_prod_xy(angle_rates, top_axis.lever_arm))
        angle_rates_btm = cross_prod_xy(angle_rates, cross_prod_xy(angle_rates, btm_axis.lever_arm))

        # Convert given values to sensor axis
        acceleration_top = top_axis.body2sensor_mat.dot(gravity_top + angle_rates_top)
        acceleration_btm = btm_axis.body2sensor_mat.dot(gravity_btm + angle_rates_btm)
        return acceleration_top, acceleration_btm


    def _gravity_to_measurements(self, ground_truth: GroundTruth, gravity_top: np.ndarray,
                                 gravity_btm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts partial gravity gradients values to sensor measurements.
        On succession, produces the partial measurements from top and
        bottom accelerometers, using expected error/noise behaviours.

        :param ground_truth: The corresponding ground truth data.
        :type ground_truth: GroundTruth

        :param gravity_top: The gravity acceleration at the top accelerometer.
        :type gravity_top: np.ndarray

        :param gravity_btm: The gravity acceleration at the bottom accelerometer.
        :type gravity_btm: np.ndarray

        :return: The acceleration measurements for top and bottom
            accelerometers respectively.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        # Unpack ground truth values
        time_step = ground_truth.timestamp
        position = ground_truth.position
        velocity = ground_truth.velocity
        attitude = ground_truth.attitude
        angle_rates = ground_truth.angle_rates

        # Generate a state for upper sensor measurement.
        tmp_gt_up = GroundTruth(time_step, position, velocity,
                                gravity_top, attitude, angle_rates)

        # Generate a state for lower sensor measurement.
        tmp_gt_down = GroundTruth(time_step, position, velocity,
                                  gravity_btm, attitude, angle_rates)

        # Take the produced ground truth input
        self._qs_top_accelerometer.take_measurement(time_step, tmp_gt_up)
        self._qs_btm_accelerometer.take_measurement(time_step, tmp_gt_down)

        # Finally, return the captured measurements
        top_measurement = self._qs_top_accelerometer.last_measurement
        bottom_measurement = self._qs_btm_accelerometer.last_measurement
        return top_measurement, bottom_measurement

    def _get_measurement(self) -> tuple[float, float]:
        """
        Produces measurement signal using collected partial measurements.
        Using the collected partial accelerometer measurements, this
        generates a signal for the top and bottom accelerometers. This is
        then used by the appointed fusion method.

        :return: Generated signal for the top and bottom accelerometers.
        :rtype: tuple[float, float]
        """

        num_atoms = self._interferometer.num_of_atoms
        sigma_phi = self._interferometer.sigma_phi
        sigma_n = self._interferometer.sigma_n
        t_pulse = self._interferometer.time_pulse
        k_eff = self._interferometer.k_eff
        eta = self._interferometer.eta

        num_atoms_sqrt = math.sqrt(num_atoms)
        t_pulse_2 = t_pulse * t_pulse

        # Get the top and bottom measurements
        top_measurements = self._measurement_data[:, 0:3]
        btm_measurements = self._measurement_data[:, 3:6]

        # Validate both sets of measurements
        # TODO: Use somewhere with acceleration measurements (Jason doesn't)
        # valid_top_measurements = self._validate_readings(top_measurements)
        # valid_btm_measurements = self._validate_readings(btm_measurements)

        # Get the average measurement value for each
        mean_top_measurement = np.mean(top_measurements[:, 2], axis=0)
        mean_btm_measurement = np.mean(btm_measurements[:, 2], axis=0)

        # Calculate phase shifts
        delta_phi_top = k_eff * mean_top_measurement * t_pulse_2
        delta_phi_btm = k_eff * mean_btm_measurement * t_pulse_2

        # Apply random phase for each interferometer measurement
        # (This is common for both accelerometers)
        phi_z0 = 2 * math.pi * self._rng.random()

        # Acquire the top accelerometer's measured signals
        s_top = math.sin(delta_phi_top + phi_z0 + sigma_phi * self._rng.normal())
        # s_top *= eta * (num_atoms + self._rng.normal() * num_atoms_sqrt)
        s_top *= eta * (num_atoms + sigma_n * self._rng.normal() * num_atoms_sqrt)  # TODO: NEW
        s_top = s_top / eta / num_atoms

        # Acquire the bottom accelerometer's measured signals
        s_btm = math.sin(delta_phi_btm + phi_z0 + sigma_phi * self._rng.normal())
        # s_btm *= eta * (num_atoms + self._rng.normal() * num_atoms_sqrt)
        s_btm *= eta * (num_atoms + sigma_n * self._rng.normal() * num_atoms_sqrt)  # TODO: NEW
        s_btm = s_btm / eta / num_atoms

        # Finally, return the signals
        return s_top, s_btm


    def _validate_readings(self, acceleration_record: np.ndarray) -> bool:
        """
        Validates if readings for the acceleration values are usable.
        Performs a check for each acceleration record (in sensor axis) to
        ensure atoms remain in the beam for usable reading. Used to detect
        when invalid measurements are likely to occur due to

        :return: Whether the X, Y and Z axis readings are valid, respectively.
        :rtype: tuple[bool, bool, bool]
        """

        # Unpack the interferometer parameters
        delta_x = self._interferometer.beam_width
        delta_z = self._interferometer.sensor_length
        bounds = np.array([delta_x, delta_x, delta_z])

        # The time step and time step squared
        ts = self._time_step
        ts_sq = ts * ts

        # The initial position and velocity
        position_s = np.zeros(3)
        velocity_s = np.zeros(3)

        for acceleration_s in acceleration_record:

            # Calculate the deviations in each axis
            position_s += (velocity_s * ts +  0.5 * acceleration_s * ts_sq)
            velocity_s += acceleration_s * ts
            deviations = np.abs(position_s)

            if np.any(deviations > bounds):
                return False

        return True

    @property
    def last_measurement(self) -> tuple[float, float]:
        """
        Returns the latest measurement produced from the sensor.

        :return: The last produced gravity gradient signal.
        :rtype: tuple[float, float]
        """
        return self._last_measurement

    @property
    def num_steps(self) -> int:
        """
        Returns the total number of internal measurement steps required.
        Used namely for sharing array sizes.

        :return: The number of internal measurement steps used.
        :rtype: int
        """
        return self._num_total_steps

    @property
    def interferometer(self) -> GravityGradInterferometer:
        """
        Returns the interferometer properties used by the sensor.

        :return: Interferometer properties.
        :rtype: GravityGradInterferometer
        """
        return self._interferometer

    @property
    def gravity_model(self) -> GravityModel:
        """
        Returns the internal gravity model used by the sensor.
        Can be accessed for testing or producing similar measurements.

        :return: Gravity Model instance used by the sensor.
        :rtype: GravityModel
        """
        return self._gravity_model


@dataclass(frozen=True)
class ParticleFilterParams:
    """
    Particle Filter Properties for the gravity gradient fusion methods.
    A list of properties relating to the particle filter processing.
    """

    num_particles: int = 1000

    init_position_sigma: float = 20.0
    init_velocity_sigma: float = 1e-5
    init_attitude_sigma: float = 1e-5

    weight_sigma: float = 0.00126

    alpha: float = 0.05
    beta: float = 0.05
    gamma: float = 0.05

    position_sigma: float = 2.0
    velocity_sigma: float = 0.005
    attitude_sigma: float = 0.001

    ellipse_intervals: int = 361
    ellipse_iterations: int = 5


class GravityGradientPF(SensorFusion):
    """
    Gravity Gradient Particle Filter Fusion.
    Applies position correction using gravity gradient signals,
    via a particle filter approach.
    """

    def __init__(self,
                 qs_gravity_grad: GravityGradiometer,
                 filter_params: ParticleFilterParams = ParticleFilterParams(),
                 est_interferometer: GravityGradInterferometer = None,
                 rng_seed: int = None):
        """
        Creates a gravity gradient particle fusion instance.

        :param qs_gravity_grad: The gravity gradient sensor to use for
            acquiring vertical gravity gradient measurements from.
        :type qs_gravity_grad: GravityGradiometer

        :param filter_params: The particle filtering parameters to use.
        :type filter_params: ParticleFilterParams

        :param est_interferometer: The estimated interferometer parameters to
            use. If not provided the real sensor parameters will be used.
        :type est_interferometer: GravityGradInterferometer

        :param rng_seed: The random number generator seed to use.
        :type rng_seed: int
        """

        # Call parent class's constructor
        super().__init__(FusionTrigger.ALL, qs_gravity_grad)
        self._gravity_gradiometer = qs_gravity_grad
        self._filter_params = filter_params

        # Obtain the interferometer properties to use:
        if est_interferometer is None:
            self._interferometer = qs_gravity_grad.interferometer
        else:
            self._interferometer = est_interferometer

        # Initialise local random number generation
        self._rng = np.random.default_rng(rng_seed)

        # Initialise particle filter properties
        num_particles = filter_params.num_particles
        self._w_particles = np.ones(num_particles) / num_particles
        self._x_particles = np.zeros((num_particles, 3))
        self._v_particles = np.zeros((num_particles, 3))
        self._a_particles = np.zeros((num_particles, 3))
        self._num_particles = num_particles

        # Indicates particle filterer requires initialising
        self._first_iteration = True

    def perform_fusion(self, estimated_state: EstimatedState) -> None:
        """
        Performs the fusion with sensors and updates the estimated state.
        When called performs fusion when gravity gradient sensor
        measurements are ready. When so, the particle filter algorithms is
        employed and the estimated state is updated.

        :param estimated_state: The current estimated states of the platform.
                                This is what the fusion method will update.
        :type estimated_state: EstimatedState
        """

        # Get the latest measurement from the sensor
        qs_measurement = self._gravity_gradiometer.last_measurement
        if qs_measurement is None:
            return

        # Initialise particle filter if not already
        if self._first_iteration:
            self.init_particle_filter(estimated_state)
            self._first_iteration = False

        # Perform weighting and conditional resampling of particles
        self._weight_particles(estimated_state, qs_measurement)
        self._resample_particles()

        # Update the estimation and particle states
        self._update_estimation(estimated_state)
        self._propagate_particles(estimated_state)

        # Reset sensor and measurement table
        self.reset()

    def init_particle_filter(self, estimated_state: EstimatedState) -> None:
        """
        Initialises the particle filter using current estimate.
        This is called during the first iteration, setting the particles'
        estimated position, velocity and attitude values based around the
        current estimated state. The range of estimation is based on the
        initial sigma values set in the filter properties for each.

        :param estimated_state: The current estimated states of the platform.
        :type estimated_state: EstimatedState
        """

        # Use estimate as reference position
        array_size = (self._num_particles, 3)

        # Unpack relevant filter properties
        pos_sigma = self._filter_params.init_position_sigma
        vel_sigma = self._filter_params.init_velocity_sigma
        att_sigma = self._filter_params.init_attitude_sigma

        # Create random offset from reference LLA
        ned_offset = self._rng.normal(size=array_size) * pos_sigma
        ned_offset[:, 2] = 0

        # Randomise particles for position, velocity and attitude
        self._x_particles = ned_offset
        self._v_particles = self._rng.normal(size=array_size) * vel_sigma
        self._a_particles = self._rng.normal(size=array_size) * att_sigma

        # Ensure there is particle for current estimate
        # self._x_particles[0, :] = 0
        # self._v_particles[0, :] = 0
        # self._a_particles[0, :] = 0


    def _weight_particles(self, estimated_state: EstimatedState,
                          measurements: tuple) -> None:
        """
        Weights particles against given measurements.
        For the given (upper and lower) sensor measurements, this re-weighs
        the particles subject to their minimum ellipse distance. On
        succession, new weights are assigned to each particle.

        :param estimated_state: The current estimated states of the platform.
        :type estimated_state: EstimatedState

        :param measurements: The upper and lower measurements received
            from the vertical gravity gradient sensor.
        :type measurements: tuple
        """

        est_position = estimated_state.position
        est_acceleration = estimated_state.acceleration
        est_attitude = estimated_state.attitude
        est_angle_rates = estimated_state.angle_rates

        # Convert from candidate NEDs to LLA positions
        cand_position = trans.ned2lla_vec(self._x_particles, est_position)
        cand_position[:, 2] = est_position[2]

        # Obtain the candidate values from particles
        lat = cand_position[:, 0]
        lon = cand_position[:, 1]
        alt = cand_position[:, 2]

        # Calculate the gravity gradient for each
        gravity_model = estimated_state.gravity_model
        cand_gradients = gravity_model.calc_vertical_grad_vec(lat, lon, alt)
        cand_attitudes = est_attitude + self._a_particles

        # Calculate the distances from ellipsoid
        distances = np.zeros(self._num_particles)
        for i in range(self._num_particles):
            distances[i] = self._min_ellipse_distance(
                measurements, est_acceleration, cand_attitudes[i],
                est_angle_rates, float(cand_gradients[i]))

        # print(f'Min Dist: {min(distances)}\t Mean Dist: {np.mean(distances)}')
        sigma2 = self._filter_params.weight_sigma
        self._w_particles *= np.exp(-distances ** 2 / (2 * sigma2))

        # TODO: IMPROVE WORKAROUND
        if np.all(self._w_particles == 0):
            self._x_particles[0, :] = 0
            self._v_particles[0, :] = 0
            self._a_particles[0, :] = 0
            self._w_particles[0] = 1

        self._w_particles /= np.sum(self._w_particles)


    def _improve_poor_weights(self, weight_threshold: float = 1e-40):

        #
        poor_particles = self._w_particles <= weight_threshold
        num_poor_particles = np.sum(poor_particles)

        # Return if no poor weights
        if num_poor_particles == 0:
            return

        if num_poor_particles == self._num_particles:
            # print(f'GAMM particle weight failure: All weights are zero!')
            self._w_particles[:] = 1 / self._num_particles
            return

        # Unpack relevant filter properties
        pos_sigma = self._filter_params.position_sigma
        vel_sigma = self._filter_params.velocity_sigma
        att_sigma = self._filter_params.attitude_sigma

        # Use estimate as reference position
        array_size = (num_poor_particles, 3)

        ned_offset = self._rng.normal(size=array_size) * pos_sigma
        vel_offset = self._rng.normal(size=array_size) * vel_sigma
        att_offset = self._rng.normal(size=array_size) * att_sigma
        ned_offset[:, 2] = 0

        indices = np.arange(self._num_particles, dtype=int)
        indices = indices[~poor_particles]

        probs = np.copy(self._w_particles[indices])
        probs /= np.sum(probs)

        i = poor_particles
        j = self._rng.choice(indices, num_poor_particles, True, probs)

        self._x_particles[i, :] = self._x_particles[j, :] + ned_offset
        self._v_particles[i, :] = self._v_particles[j, :] + vel_offset
        self._a_particles[i, :] = self._a_particles[j, :] + att_offset

        # IMPROVE
        self._w_particles[i] = self._w_particles[j]


    def _resample_particles(self):
        """
        Performs conditional particle resampling.
        If the combined weight of particles is below a given threshold,
        resampling of values and weights is applied.
        """

        n_eff = 1.0 / np.sum(self._w_particles * self._w_particles)
        threshold = self._num_particles / 3.0

        if n_eff >= threshold:
            return

        new_particles = np.column_stack([
            self._x_particles[:, 0:2], self._v_particles, self._a_particles])
        w_particles, new_particles, _ = self._resample(self._w_particles, new_particles)

        num_states = 8
        min_cov = np.eye(num_states) * 1e-6

        co_var_q = weighted_cov(new_particles, w_particles) + min_cov
        l_q = np.linalg.cholesky(co_var_q)

        prob = 0.85
        low_noise = 0.1
        high_noise = 2.0

        for i in range(self._num_particles):
            if self._rng.random() < prob:
                l_q *= low_noise
            else:
                l_q *= high_noise

            dx = l_q @ self._rng.normal(size=num_states)
            new_particles[i, :] += dx

        self._x_particles[:, 0:2] = new_particles[:, 0:2]
        self._v_particles = new_particles[:, 2:5]
        self._a_particles = new_particles[:, 5:8]
        self._w_particles = w_particles  # TODO: NEW

    def _update_estimation(self, estimated_state: EstimatedState):
        """
        Using the particle filter values and weights, updates the estimation.
        Updates the current estimated position, velocity and attitude using
        the associated particle values and weights. On succession, the
        particle values  are also updated by this change.

        :param estimated_state: The current estimated states of the platform.
        :type estimated_state: EstimatedState
        """

        # Unpack required parameters
        alpha = self._filter_params.alpha
        beta = self._filter_params.beta
        gamma = self._filter_params.gamma

        # Obtain weighted average
        mean_x = self._w_particles @ self._x_particles
        mean_v = self._w_particles @ self._v_particles
        mean_a = self._w_particles @ self._a_particles

        # Obtain weighted average
        # weights = self._w_particles[:, None]
        # sum_weights = np.sum(self._w_particles)
        # mean_x = np.sum(weights * self._x_particles / sum_weights, axis=0)
        # mean_v = np.sum(weights * self._v_particles / sum_weights, axis=0)
        # mean_a = np.sum(weights * self._a_particles / sum_weights, axis=0)

        # Apply fixed gain to updates
        est_d_position = alpha * mean_x
        est_d_velocity = beta * mean_v
        est_d_attitude = gamma * mean_a

        # Adjustment position for accidental altitude drift
        old_alt = estimated_state.position[2]
        new_position = trans.ned2lla(est_d_position, estimated_state.position)
        new_position[2] = old_alt

        # Apply update to estimated state
        estimated_state.update_estimates(
            position=new_position,
            velocity=(est_d_velocity + estimated_state.velocity),
            attitude=(est_d_attitude + estimated_state.attitude))

        # Adjust particle values for update applied
        self._x_particles -= est_d_position
        self._v_particles -= est_d_velocity
        self._a_particles -= est_d_attitude

    def _propagate_particles(self, estimated_state):
        """
        Propagates particles and add process noise.

        :param estimated_state: The current estimated states of the platform.
        :type estimated_state: EstimatedState
        """

        sigma_x = self._filter_params.position_sigma  # metres
        sigma_v = self._filter_params.velocity_sigma  # metres/s
        sigma_a = self._filter_params.attitude_sigma  # degrees

        est_attitude = estimated_state.attitude
        attitude_rad = np.radians(est_attitude)
        rot_earth2body = trans.rotate_3d(*attitude_rad)

        sensor = self._gravity_gradiometer
        num_steps = sensor.num_steps
        time_step = sensor.time_step

        dt = time_step * num_steps

        for i in range(self._num_particles):

            # Define the random noise amounts to add
            position_noise = sigma_x * self._rng.normal(size=3)
            velocity_noise = sigma_v * self._rng.normal(size=3)
            attitude_noise = sigma_a * self._rng.normal(size=3)

            # Update position
            position = self._x_particles[i] + np.linalg.solve(rot_earth2body, self._v_particles[i]) * dt
            position += position_noise

            # Update particles
            self._x_particles[i, 0:2] = position[0:2]
            self._v_particles[i] += velocity_noise
            self._a_particles[i] += attitude_noise


    def _resample(self, weights: np.ndarray, values: np.ndarray) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs weighted random resampling of the particles.

        :param weights: An array containing the current particle weights.
        :type weights: np.ndarray

        :param values: An array containing the current particle values.
        :type values: np.ndarray

        :return: The particles, their updated weights and their indices.
        :rtype: tuple (3-elements)
        """

        # Obtain the normalised weights
        probs = weights / np.sum(weights)
        n = len(weights)

        # Select Points for Resampling
        new_weights = np.ones(weights.shape) / n
        new_values = np.zeros(values.shape)
        p_i = np.zeros(weights.shape)
        indices = np.arange(n)

        for i in indices:
            j = self._rng.choice(indices, 1, True, probs)
            new_values[i, :] = values[j, :]
            p_i[i] = j

        return new_weights, new_values, p_i

    def _min_ellipse_distance(self, measurements: tuple,
                              acceleration: np.ndarray,
                              attitude: np.ndarray,
                              angle_rates: np.ndarray,
                              grav_grad_z: float) -> float:
        """
        Finds the minimum ellipse distance between measurements and candidate
        estimated states. Given the measurements produced by the sensor, and
        a particle's estimated acceleration, attitude and vertical gravity
        gradient (generated from its estimated position), this function
        computes the closest distance to an ellipsoid. The distance
        generated is employed for the re-weighing of particles for
        the particle filter.

        :param measurements: The upper and lower measurements received
            from the vertical gravity gradient sensor.
        :type measurements: tuple

        :param acceleration: The candidate acceleration values of the
            current particle (in m/s^2).
        :type acceleration: np.ndarray (3-elements)

        :param attitude: The candidate attitude values of the current
            particle (in degrees).
        :type attitude: np.ndarray (3-elements)

        :param grav_grad_z: The computed vertical gravity gradient
            for the position of the current particle (in m/s^2). This is
            computed externally to benefit from vectorisation.
        :type grav_grad_z: float

        :return: The minimum distance to the ellipsoid.
        :rtype: float
        """

        # Unpack interferometer parameters
        num_atoms = self._interferometer.num_of_atoms
        t_pulse = self._interferometer.time_pulse
        sigma_n = self._interferometer.sigma_n
        k_eff = self._interferometer.k_eff
        eta = self._interferometer.eta
        z0 = self._interferometer.sensor_z_top
        z1 = self._interferometer.sensor_z_bottom

        # Rotation matrix from earth to body axes
        rot_earth2body = trans.rotate_3d(*np.radians(attitude))

        # Convert actual acceleration from body to earth axes
        acceleration_b_z0 = np.linalg.solve(rot_earth2body, acceleration)
        acceleration_b_z1 = np.copy(acceleration_b_z0)

        # Add vertical gradient term
        acceleration_b_z0 += grav_grad_z * z0
        acceleration_b_z1 += grav_grad_z * z1

        # Convert acceleration from earth back to body axes
        acceleration_b_z0 = rot_earth2body @ acceleration_b_z0
        acceleration_b_z1 = rot_earth2body @ acceleration_b_z1

        # TODO: Experimental - Added recently
        tmp = self._acceleration_to_sensor_axis(
            angle_rates, acceleration_b_z0, acceleration_b_z1)
        acceleration_b_z0, acceleration_b_z1 = tmp

        # Calculate phase shifts
        t_pulse_2 = t_pulse * t_pulse
        delta_phi_z_0 = k_eff * acceleration_b_z0[2] * t_pulse_2
        delta_phi_z_1 = k_eff * acceleration_b_z1[2] * t_pulse_2

        # Calculate expected levels of noise (fixed per particle)
        num_atoms_sqrt = math.sqrt(num_atoms)
        top_noise = sigma_n * self._rng.normal() * num_atoms_sqrt
        btm_noise = sigma_n * self._rng.normal() * num_atoms_sqrt

        # Set current search range and minimum distance
        range_start = 0.0
        range_end = 2.0 * math.pi
        min_dist = math.inf

        # Unpack settings from particle filter properties
        num_intervals = self._filter_params.ellipse_intervals
        num_attempts = self._filter_params.ellipse_iterations

        # Iteratively solve closest point:
        for i in range(num_attempts):

            # Phase for each interferometer measurement
            phi_z0 = np.linspace(range_start, range_end, num_intervals)

            # Acquire the top accelerometer's measured signals
            top_signal = np.sin(delta_phi_z_0 + phi_z0)
            top_signal *= eta * (num_atoms + top_noise)

            # Acquire the bottom accelerometer's measured signals
            btm_signal = np.sin(delta_phi_z_1 + phi_z0)
            btm_signal *= eta * (num_atoms + btm_noise)

            # Measure and obtain the distance between signals and measurement
            signals = np.column_stack((top_signal, btm_signal)) / eta / num_atoms
            dists = np.linalg.norm(measurements - signals, axis=1)
            ranks = np.argsort(dists)

            # Adjust the search range
            min_dist = min(min_dist, dists[ranks[0]])
            closest_pair = phi_z0[ranks[:2]]
            range_start = min(closest_pair)
            range_end = max(closest_pair)

            # Adjustment for if between first and last interval.
            if range_start == phi_z0[0] and range_end == phi_z0[-1]:
                range_start = range_end
                range_end = 2.0 * math.pi

        # Return the distance of the closest point found
        return min_dist

    def _acceleration_to_sensor_axis(self, angle_rates, acceleration_b_z0, acceleration_b_z1):

        # Obtain the sensor axis for each sensor
        sensor = self._gravity_gradiometer
        if hasattr(sensor, "_qs_top_accelerometer"):
            top_axis = sensor._qs_top_accelerometer.sensor_axis
            btm_axis = sensor._qs_btm_accelerometer.sensor_axis
        else:
            top_axis = _offset_vertical_axis(
                sensor.sensor_axis, self._interferometer.sensor_z_top)
            btm_axis = _offset_vertical_axis(
                sensor.sensor_axis, self._interferometer.sensor_z_bottom)

        # Obtain the angle rate acceleration for top and bottom sensors
        angle_rates_top = cross_prod_xy(angle_rates, cross_prod_xy(angle_rates, top_axis.lever_arm))
        angle_rates_btm = cross_prod_xy(angle_rates, cross_prod_xy(angle_rates, btm_axis.lever_arm))

        # Convert given values to sensor axis
        acceleration_top = top_axis.body2sensor_mat.dot(acceleration_b_z0 + angle_rates_top)
        acceleration_btm = btm_axis.body2sensor_mat.dot(acceleration_b_z1 + angle_rates_btm)
        return acceleration_top, acceleration_btm

    def reset(self):
        """
        Resets the state of the fusion, clearing internal temporary states.
        When call the fusion will return to expecting the first measurement,
        then continuing from there.
        """
        self._gravity_gradiometer.reset()


def _offset_vertical_axis(sensor_axis: SensorAxis, vertical_offset: float) -> SensorAxis:
    """
    Applies vertical offset to sensor axis's.
    Creates a copy of given SensorAxis and applies the given offset amounts
    to its vertical lever arm. Required for accelerometer displacement.

    :param sensor_axis: The sensor axis instance to effectively copy.
    :type sensor_axis: SensorAxis

    :param vertical_offset: The amount to offset the vertical axis by (in metres)
    :type vertical_offset: float

    :return: A copy of given sensor axis with vertical offset applied.
    :rtype: SensorAxis
    """

    sensor_angles = sensor_axis.sensor_angles_deg
    sensor_angles_rad = sensor_axis.sensor_angles_rad
    lever_arm = sensor_axis.lever_arm

    lever_offset = np.array([0, 0, vertical_offset])
    rot_body2sensor = trans.rotate_3d(*sensor_angles_rad)
    new_lever_arm = lever_arm + np.linalg.solve(rot_body2sensor, lever_offset)

    return SensorAxis(sensor_angles, new_lever_arm)
