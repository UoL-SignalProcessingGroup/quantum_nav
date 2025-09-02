"""
====================
take_measurements.py
====================

:summary:
    A module for performing simulations from pre-generated IMU measurements.
    This provides an example of performing a simple simulation using a series
    of pre-generated IMU (accelerometer and gyroscope) measurements. This
    allows the selection of the INS fusion method and uses it along with
    dummy sensors to perform a simulation, reviewing estimation accuracy
    on completion.

    When executed (see bottom), this module runs the following steps:
        1.  Read pre-generated ground truth records
        2.  Read pre-generated sensor measurements records
        3.  Initialise dummy sensor and INS solver solution
        4.  Perform simple simulation loop
        5.  Review estimation errors

:version:
    0.9.0 - Awaiting adding support for Altimeter
"""

# Other imports
import numpy as np
from math import inf
from pathlib import Path

# Reuse functions from the other script
from create_measurements import get_accelerometer
from create_measurements import get_gyroscope

# Qnav gravity classes
from qnav.gravity.base import GravityModel
from qnav.gravity.somigliana import Somigliana

# Qnav estimation state classes
from qnav.estimation.kalman import generate_matrices
from qnav.estimation.state import EstimatedState
from qnav.estimation.state import KalmanEstimatedState

# Qnav INS fusion solver classes
from qnav.fusion.ins import NumericalINS
from qnav.fusion.ins import RungeKutta
from qnav.fusion.ins import AdamsBashforth
from qnav.fusion.kalman import KalmanINS

# Qnav sensor classes
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.altimeter import Altimeter
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.sensor import SensorFusion

# Qnav dummy sensor classes
from qnav.quantum.dummy import DummyAccelerometer
from qnav.quantum.dummy import DummyGyroscope

# Qnav trajectory (ground truth) related classes
from qnav.waypoints.trajectory import GroundTruth

# Qnav frame transformations (lla2ned)
from qnav.util.transformations import lla2ned


def get_ground_truth(filepath: Path) -> dict[str, np.ndarray]:
    """
    Reads a given ground truth CSV file and returns its contents.
    For a given CSV file containing ground truth records, this function will
    attempt to read it and return a dictionary containing records for the
    time, positon, velocity, acceleration, attitude and angle rate values.

    :param filepath: Path to the ground truth CSV file.
    :type filepath: Path

    :return: Dictionary containing ground truth record columns.
    :rtype: dict[str, np.ndarray]
    """

    # Attempt to read the ground truth file
    records = np.genfromtxt(filepath, delimiter=',')

    # Represent columns in simple dictionary
    return {
        'timestamps': records[:, 0],
        'position': records[:, 1:4],
        'velocity': records[:, 4:7],
        'acceleration': records[:, 7:10],
        'attitude': records[:, 10:13],
        'angle_rates': records[:, 13:16],
    }

def get_measurements(filepath: Path) -> dict[str, np.ndarray]:
    """
    Reads a given measurement CSV file and returns its contents.
    For a given CSV file containing sensor measurement records, this function
    will attempt to read it and return a dictionary containing records the
    timesteps and measurement values.

    :param filepath: Path to the sensor measurement records.
    :type filepath: Path

    :return: Dictionary containing ground truth record columns.
    :rtype: dict[str, np.ndarray]
    """

    # Attempt to read the measurements file
    records = np.genfromtxt(filepath, delimiter=',')

    # Extract data columns
    timestamps = records[:, 0]
    measurements = records[:, 1:]
    freq = 1 / float(np.mean(np.diff(timestamps)))

    # Represent columns in simple dictionary
    return {
        'frequency': freq,
        'timestamps': timestamps,
        'measurements': measurements,
    }

def get_gravity_model() -> GravityModel:
    """
    Initialises and returns a gravity model to use for estimation.
    This function returns a simple Somigliana formula gravity model.
    Can be replaced with more sophisticated solutions.

    :return: An example gravity instance to use.
    :rtype: GravityModel
    """
    return Somigliana()

def get_estimated_state(gt_records: dict, use_kalman: bool = True) -> EstimatedState:
    """
    Initialises and returns the initial estimated state.
    Using the given ground truth records, this function will create and
    return an estimated state, initialised using the first ground truth
    record. Optionally, a Kalman Estimated State can be requested that
    also included shared Kalman Filter matrices.

    :param gt_records: Dictionary containing ground truth record columns.
    :type gt_records: dict[str, np.ndarray]

    :param use_kalman: Set to True to return Kalman Estimated State,
        otherwise a normal Estimated State will be given.
    :type use_kalman: bool

    :return: An estimated state initialised using first ground truth record.
    :rtype: EstimatedState
    """

    # Extract the first records from ground truth
    init_timestep = gt_records['timestamps'][0]
    init_position = gt_records['position'][0]
    init_velocity = gt_records['velocity'][0]
    init_acceleration = gt_records['acceleration'][0]
    init_attitude = gt_records['attitude'][0]
    init_angle_rates = gt_records['angle_rates'][0]

    # Convert first records to a ground truth record instance
    gt_record = GroundTruth(init_timestep, init_position, init_velocity,
                            init_acceleration, init_attitude, init_angle_rates)

    # Define gravity model to use
    gravity = get_gravity_model()

    # If a kalman estimated state is to be used:
    if use_kalman:

        # Configure Kalman filter
        # (see 'Estimation' section in example settings files)
        imu_rate = 1 / 500  # Assumed time step (1 / imu freq)
        acc_meas_error = 1e-3  # Assumed avg. accelerometer measurement error
        vel_meas_error = 1e-3  # Assumed avg. accelerometer measurement error
        acc_proc_noise = 1e-3  # Assumed avg. acceleration process noise
        vel_proc_noise = 1e-6  # Assumed avg. velocity process noise
        att_proc_noise = 1e-5  # Assumed avg. attitude process noise
        ang_proc_noise = 1e-5  # Assumed avg. angle rate process noise

        # Initialise error matrix
        state_errors = np.eye(15) * 1e-9

        # Initialise required Kalman filter matrices
        h_matrix, r_matrix, f_matrix, q_matrix = generate_matrices(
            imu_rate, acc_meas_error, vel_meas_error, acc_proc_noise,
            vel_proc_noise, ang_proc_noise, att_proc_noise)

        # Return kalman estimated state
        return KalmanEstimatedState(
            gt_record, h_matrix, r_matrix, f_matrix,
            q_matrix, state_errors, gravity)
    else:

        # Otherwise, return simple estimated state
        return EstimatedState(gt_record, gravity)

def get_dummy_accelerometer(freq: float) -> DummyAccelerometer:
    """
    Initialises and returns a dummy accelerometer model to use.
    Given a measurement frequency, this function will create a dummy copy of
    an accelerometer sensor. This allows modified measurements to be used
    inside a simulation loop.

    :param freq: Measurement frequency of the accelerometer. Must match that
        of the true accelerometer whose measurements are being used.
    :type freq: float

    :return: An example dummy accelerometer instance.
    :rtype: DummyAccelerometer
    """
    # Get the original accelerometer (a copy will do),
    # then create and return a dummy version (copy configuration)
    acc = get_accelerometer(freq)
    return DummyAccelerometer(acc)

def get_dummy_gyroscope(freq: float) -> DummyGyroscope:
    """
    Initialises and returns a dummy gyroscope model to use.
    Given a measurement frequency, this function will create a dummy copy of
    a gyroscope sensor. This allows modified measurements to be used
    inside a simulation loop.

    :param freq: Measurement frequency of the gyroscope. Must match that
        of the true gyroscope whose measurements are being used.
    :type freq: float

    :return: An example dummy gyroscope instance.
    :rtype: DummyGyroscope
    """
    # Get the original gyroscope (a copy will do),
    # then create and return a dummy version (copy configuration)
    gyro = get_gyroscope(freq)
    return DummyGyroscope(gyro)

def get_ins_solver(acc: Accelerometer, gyro: Gyroscope, solver: str = 'simple') -> NumericalINS:
    """
    Generates INS solver instance for given IMU sensors.
    This function initialises and returns a selected INS solver for given
    accelerometer and gyroscope sensors. This is used to apply fusion and
    update the estimated state using IMU measurements. Supported solver
    options: 'simple', 'runge kutta', 'adams bashforth' or 'kalman'.

    :param acc: Accelerometer sensor instance to take measurements from.
    :type acc: Accelerometer

    :param gyro: Gyroscope sensor instance to take measurements from.
    :type gyro: Gyroscope

    :param solver: Key for the solver type to use (supports 'simple',
        'runge kutta', 'adams bashforth' or 'kalman'). Case insensitive.
    :type solver: str

    :return: Initialised INS solver instance.
    :rtype: NumericalINS
    """
    # For the requested solver type:
    match solver.casefold():

        case 'simple':
            return NumericalINS(acc, gyro)

        case 'runge kutta':
            return RungeKutta(acc, gyro)

        case 'adams bashforth':
            return AdamsBashforth(acc, gyro)

        case 'kalman':
            return KalmanINS(acc, gyro)

        case _:
            raise ValueError(f'Unknown solver: {solver}')

def perform_simulation(est_state: EstimatedState,
                       imu_solver: NumericalINS,
                       accelerometer: DummyAccelerometer,
                       gyroscope: DummyGyroscope,
                       accelerometer_outputs: dict,
                       gyroscope_outputs: dict,
                       alt_solver: SensorFusion = None,
                       altimeter: Altimeter = None,
                       altimeter_outputs: dict = None) -> None:
    """
    Executes a simple simulation using predefined IMU measurements.
    This function performs a simple simulation loop, taking given IMU
    measurement series and applying them in order. At each interval the
    estimated state is updated using the corresponding fusion methods.
    Optional Altimeter information can also be provided.

    :param est_state: Estimated state to take measurements from.
    :type est_state: EstimatedState

    :param imu_solver: IMU solver instance used to preform INS fusion.
    :type imu_solver: NumericalINS

    :param accelerometer: Accelerometer sensor instance to take measurements
        from. Must be a dummy accelerometer to override inputs.
    :type accelerometer: DummyAccelerometer

    :param gyroscope: Gyroscope sensor instance to take measurements from.
        Must be a dummy gyroscope to override inputs..
    :type gyroscope: DummyGyroscope

    :param accelerometer_outputs: Dictionary of accelerometer measurements.
    :type accelerometer_outputs: dict[str, float | np.ndarray]

    :param gyroscope_outputs: Dictionary of gyroscope measurements.
    :type gyroscope_outputs: dict[str, float | np.ndarray]

    :param alt_solver: Altimeter solver instance used to preform altimeter
        fusion (optional, default is None).
    :type alt_solver: SensorFusion

    :param altimeter: Altimeter sensor instance to take measurements from.
        Must be a dummy altimeter to override inputs (optional, default
        is None).
    :type altimeter: Altimeter

    :param altimeter_outputs: Dictionary of altimeter measurements (optional,
        default is None).
    :type altimeter_outputs: dict[str, float | np.ndarray]
    """

    # Assume Accelerometer and Gyroscope times are identical
    imu_times = accelerometer_outputs['timestamps']
    alt_times = accelerometer_outputs['timestamps']

    # Extract imu sensor measurement outputs
    acc_measurement = accelerometer_outputs['measurements']
    gyro_measurement = gyroscope_outputs['measurements']

    # Create indices for imu measurements
    imu_len = len(imu_times)
    imu_index = 0

    # create indices for altimeter measurements
    alt_len = 0
    alt_index = 0

    # If all altimeter parameters have been provided:
    if (alt_solver is not None) \
            and (altimeter is not None) \
            and (altimeter_outputs is not None):

        # Unpack additional altimeter information
        alt_times = altimeter_outputs['timestamps']
        alt_measurement = altimeter_outputs['measurements']
        alt_len = len(alt_times)

    # Repeat until all measurements have been processed
    while True:

        # Extract the next measurement times
        imu_time = imu_times[imu_index] if imu_index < imu_len else inf
        alt_time = alt_times[alt_index] if alt_index < alt_len else inf

        # If an IMU measurement is due:
        if imu_time < alt_time:

            # Override last measurements from dummy sensor with record
            accelerometer.last_measurement = acc_measurement[imu_index]
            gyroscope.last_measurement = gyro_measurement[imu_index]

            # Perform imu fusion and increment imu index
            imu_solver.perform_fusion(est_state)
            imu_index += 1

            # Update the time estimate state timestamp
            estimated_state.update_estimates(timestamp=imu_time)

        # If an altimeter measurement is due:
        elif alt_time < imu_time:

            # TODO: Add implementation
            # Override last measurements from dummy sensor with record
            # altimeter.last_measurement = alt_measurement[alt_index]

            # Perform altimeter fusion and increment imu index
            # alt_solver.perform_fusion(est_state)
            alt_index += 1

            # Update the time estimate state timestamp
            estimated_state.update_estimates(timestamp=imu_time)

        # If no measurements are remaining:
        elif imu_time == alt_time == inf:
            break  # Break from the loop

def review_simulation(est_state: EstimatedState, gt_records: dict) -> None:
    """
    Displays a review of the simulation results.
    Simple calculates the difference between the final ground truth record
    and estimated state, then displays them for each property in the output
    console. An minor error can be expected if time of the last measurement
    does not match time of last ground truth record.

    :param est_state: Estimated state after simulation.
    :type est_state: EstimatedState

    :param gt_records: Dictionary of ground truth records.
    :type gt_records: dict[str, float | np.ndarray]
    """

    # Unpack final ground truth records
    true_final_time = gt_records['timestamps'][-1]
    true_final_pos = gt_records['position'][-1]
    true_final_vel = gt_records['velocity'][-1]
    true_final_acc = gt_records['acceleration'][-1]
    true_final_att = gt_records['attitude'][-1]
    true_final_ang = gt_records['angle_rates'][-1]

    # Calculate errors from estimated state properties
    time_error = true_final_time - est_state.timestamp
    pos_error = lla2ned(true_final_pos, est_state.position)
    vel_error = true_final_vel - est_state.velocity
    acc_error = true_final_acc - est_state.acceleration
    att_error = true_final_att - est_state.attitude
    ang_error = true_final_ang - est_state.angle_rates

    # Display the calculate errors in output console
    print('Simulation Estimation Errors:')
    print(f' • {'Time:':<15}{time_error:.8f} (s)')
    print(f' • {'Position:':<15}{pos_error} (m)')
    print(f' • {'Velocity:':<15}{vel_error} (m/s)')
    print(f' • {'Acceleration:':<15}{acc_error} (m/s²)')
    print(f' • {'Attitude:':<15}{att_error} (deg)')
    print(f' • {'Angle Rates:':<15}{ang_error} (deg/s)')


# Run if script is called directly
if __name__ == "__main__":

    # 1. Load ground truth records
    input_dir = Path('sensor_data')
    print('[1/5] Reading ground truth records...')
    ground_truth = get_ground_truth(input_dir / 'ground_truth.csv')

    # 2. Load pre-generated measurement records
    print('[2/5] Reading sensor measurements...')
    acc_measurements = get_measurements(input_dir / 'accelerometer.csv')
    gyro_measurements = get_measurements(input_dir / 'gyroscope.csv')
    # alt_measurements = get_measurements(input_dir / 'altimeter.csv')

    # 3. Define the initial estimated state (uses first record)
    #   (Change 'use_kalman' option if using Kalman INS)
    print('[3/5] Initialising sensors and fusion...')
    estimated_state = get_estimated_state(ground_truth, use_kalman=False)

    # 4. Get dummy sensors (used to replace actual sensors)
    acc_sensor = get_dummy_accelerometer(freq=500)
    gyro_sensor = get_dummy_gyroscope(freq=500)
    # altimeter = get_dummy_altimeter(freq=500)

    # 5. Initialise IMU fusion instance:
    #    (Options: 'simple', 'runge kutta', 'adams bashforth' or 'kalman')
    ins_fusion = get_ins_solver(acc_sensor, gyro_sensor, solver='simple')

    # 6. Initialise altimeter fusion instance
    #    (Options: 'simple', 'alpha beta')
    # alt_fusion = get_altimeter_solver(altimeter, solver='simple')

    # 7. Perform simulation with measurements
    print('[4/5] Performing simulation (will take a few minutes)...',)
    perform_simulation(estimated_state, ins_fusion, acc_sensor, gyro_sensor,
                       acc_measurements, gyro_measurements)

    # 8. Review simulation results
    print('[5/5] Reviewing simulation results...\n')
    review_simulation(estimated_state, ground_truth)
