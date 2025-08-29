"""
======================
create_measurements.py
======================

:summary:
    A module for generating complete IMU records for trajectories.
    Providing a complete trajectory for given basepoint positions, this
    module will create a complete series of (accelerometer and gyroscope)
    measurements. These will be produced using desired simulated sensor
    grade levels (i.e. marine, aviation, consumer) and outputted at a
    requested resolution. The resulting measurements can then be fed
    directly into configured INS solvers.

    When executed (see bottom), this module runs the following steps:
        1.  Create example trajectory
        2.  Initialise IMU sensors
        3.  Capture IMU outputs
        4.  Save produced measurements
        5.  Save selected ground truth records

:version:
    1.1.0 - Included exporting of ground truth records
"""

# Other imports
import numpy as np
from pathlib import Path

# Qnav gravity classes
from qnav.gravity.base import GravityModel
from qnav.gravity.somigliana import Somigliana

# Qnav sensor and measurement classes
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.altimeter import Altimeter
from qnav.measurement.platform import SensorAxis

# Qnav trajectory related classes
from qnav.waypoints.trajectory import Trajectory
from qnav.waypoints.trajectory import Waypoints
from qnav.waypoints.vehicle import Vehicle


def get_gravity_model() -> GravityModel:
    """
    Initialises and returns a gravity model to use.
    This function returns a simple Somigliana formula gravity model.
    Can be replaced with more sophisticated solutions.

    :return: An example gravity instance to use.
    :rtype: GravityModel
    """
    return Somigliana()

def get_trajectory() -> Trajectory:
    """
    Initialises and returns a trajectory interpolator to use.
    This function return an example trajectory instance, containing the
    ground truth that measurements can be produced from. This function can be
    modified to change the arrival times, destinations, vehicle constraints,
    gravity model or interpolation method used to produce these records.

    :return: An example Trajectory instance to use.
    :rtype: Trajectory
    """

    # Define trajectory waypoints (arrival times and positions)
    times = np.array([0.0, 175.0, 500.0, 850.0])
    lats = np.array([53.407579, 53.410000, 53.700000, 53.407579])
    lons = np.array([-2.967853, -2.70000, -2.80000, -2.967853])
    alts = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    waypoints = Waypoints(times, lats, lons, alts)

    # Define a vehicle profile
    vehicle = Vehicle(
        description='Example Plane',   # Name of the profile
        turn_rate_max=5.0,             # Maximum turn rate (deg/s)
        acceleration_max=2.0,          # Maximum acceleration (m/s^2)
        deceleration_max=1.0,          # Maximum deceleration (m/s^2)
        time_delay_acc=1.0,            # Time delay before accelerating (s)
        time_delay_turn=1.0,           # Time delay before turning (s)
    )

    # Initialise gravity model to use
    gravity = get_gravity_model()

    # Define the base frequency (in Hertz)
    interp_freq = 200.0

    # Define the interpolation method ('cubic', 'pchip' or 'akima')
    interp_method = 'pchip'

    # Finally, attempt to generate and return full trajectory instance
    return Trajectory(waypoints, vehicle, gravity, interp_freq, interp_method)

def get_accelerometer(freq: float) -> Accelerometer:
    """
    Initialises and returns an accelerometer model to use.
    This function returns an accelerometer instance with an error profile
    where all errors are set to zero (theoretically perfect). This function
    can be modified to return an accelerometer with more realistic errors
    (see configuration profiles for suitable values).

    :param freq: Frequency of accelerometer measurements in Hertz.
    :type freq: float

    :return: An example Accelerometer instance to use.
    :rtype: Accelerometer
    """

    # Define the accelerometer error profile
    acc_errors = ErrorProperties(
        bias_error=np.random.randn(3) * 0.0,        # (micro-g)
        bias_drift_rate=np.random.randn(3) * 0.0,   # (micro-g/root sec)
        scale_error=np.random.randn(3) * 0.0,       # (ppm)
        non_orth_error=np.random.randn(6) * 0.0,    # (micro-radians)
        avg_meas_noise=np.random.randn(3) * 0.0     # (micro-g.root Hz)
    )

    # Define the accelerometer sensor axis
    acc_axis = SensorAxis(
        sensor_angles=np.zeros(3),
        lever_arm=np.zeros(3)
    )

    # Define internal random number generation seed to use
    acc_seed = 12345  # (Use None to randomise each time)

    # Finally, initialise and return an accelerometer sensor
    return Accelerometer(freq, acc_errors, acc_axis, rand_seed=acc_seed)

def get_gyroscope(freq: float) -> Gyroscope:
    """
    Initialises and returns an gyroscope model to use.
    This function returns an gyroscope instance with an error profile
    where all errors are set to zero (theoretically perfect). This function
    can be modified to return an gyroscope with more realistic errors
    (see configuration profiles for suitable values).

    :param freq: Frequency of gyroscope measurements in Hertz.
    :type freq: float

    :return: An example Gyroscope instance to use.
    :rtype: Gyroscope
    """

    # Define the gyroscope error profile
    acc_errors = ErrorProperties(
        bias_error=np.random.randn(3) * 0.0,        # (micro-radians/sec)
        bias_drift_rate=np.random.randn(3) * 0.0,   # (micro-rad/sec.root sec)
        scale_error=np.random.randn(3) * 0.0,       # (ppm)
        non_orth_error=np.random.randn(6) * 0.0,    # (micro-radians)
        avg_meas_noise=np.random.randn(3) * 0.0     # (micro-radians/sec.root Hz)
    )

    # Define the gyroscope sensor axis
    acc_axis = SensorAxis(
        sensor_angles=np.zeros(3),
        lever_arm=np.zeros(3)
    )

    # Define internal random number generation seed to use
    acc_seed = 12345  # (Use None to randomise each time)

    # Finally, initialise and return an gyroscope sensor
    return Gyroscope(freq, acc_errors, acc_axis, rand_seed=acc_seed)

def get_altimeter(freq: float) -> Altimeter:
    """
    Initialises and returns an altimeter model to use.
    This function returns an altimeter instance with an error profile
    where all errors are set to zero (theoretically perfect). This function
    can be modified to return an altimeter with more realistic errors
    (see configuration profiles for suitable values).

    :param freq: Frequency of altimeter measurements in Hertz.
    :type freq: float

    :return: An example altimeter instance to use.
    :rtype: Altimeter
    """

    # Define altimeter bias error (metres)
    alt_bias_error = 0.0

    # Define altimeter bias drif rate (metres/root sec)
    alt_bias_drift_rate = 0.0

    # Define internal random number generation seed to use
    alt_seed = 12345  # (Use None to randomise each time)

    # Finally, initialise and return an altimeter sensor
    return Altimeter(freq, alt_bias_error, alt_bias_drift_rate, rand_seed=alt_seed)

def get_measurements(sensor: Accelerometer | Gyroscope | Altimeter, traj: Trajectory) -> dict:
    """
    Produces series of measurements from given sensor and trajectory.
    This function will collect measurements from the ground truth trajectory
    using the sensor, at its configured frequency. This features a simple
    simulation loop, updating the sensor time and error states with each
    iteration. On succession a dictionary containing the output frequency,
    measurement values and measurement times is produced.

    :param sensor: A fresh sensor instance, being either an
        accelerometer, gyroscope or altimeter instance.
    :type sensor: Accelerometer | Gyroscope | Altimeter

    :param traj: A trajectory instance to use.
    :type traj: Trajectory
    """

    # Unpack required values
    current_time = sensor.next_update
    end_time = traj.end_time
    freq = sensor.frequency

    # Determine the size of the output
    output_size = 1 if isinstance(sensor, Altimeter) else 3
    num_steps = int((end_time - current_time) * freq)

    # Pre-allocate time step and measurement arrays
    measurements = np.zeros((num_steps, output_size))
    time_steps = np.zeros(num_steps)

    # Capture measurements for time step
    for i in range(num_steps):

        # Update current simulation time
        current_time = sensor.next_update
        clock_time = current_time

        # Take measurement for
        ground_truth = trajectory.get_record(current_time)
        sensor.take_measurement(clock_time, ground_truth)

        # Update sensor timer and errors
        sensor.update()

        # Collect measurements
        measurements[i] = sensor.last_measurement
        time_steps[i] = current_time

    # Finally, return dictionary containing results
    return {
        'frequency': freq,
        'timestamps': time_steps,
        'measurements': measurements
    }


def save_trajectory(traj: Trajectory, filepath: Path) -> None:
    """
    Writes trajectory records to given CSV file.
    Creates a CSV containing required trajectory ground truth records.

    :param traj: A trajectory instance to acquire records from.
    :type traj: Trajectory

    :param filepath: Path to CSV file.
    :type filepath: Path
    """

    # Create parent directories if missing:
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)

    # Get the start and end times of trajectory
    start_time = traj.start_time
    end_time = traj.end_time

    # Interpolate records for required timestamps
    first_record = traj.get_record(start_time).as_numpy()
    last_record = traj.get_record(end_time).as_numpy()

    # Format the file header comment
    file_header = ', '.join(['Timestamp',
                   'Latitude', 'Longitude', 'Altitude',
                   'Velocity Along', 'Velocity Across', 'Velocity Down',
                   'Acc. Along', 'Acc. Across', 'Acc. Down',
                   'Heading', 'Pitch', 'Roll',
                   'P Rate', 'Q Rate', 'R Rate'])

    # Concatenate and write results to csv file
    data = np.vstack((first_record, last_record))
    np.savetxt(filepath, data, fmt='%.18e', delimiter=',', header=file_header)

def save_measurements(results: dict, filepath: Path) -> None:
    """
    Writes measurement results to given CSV file.
    Creates a CSV file containing the measurement times and records.

    :param results: Dictionary containing measurement results.
    :type results: dict

    :param filepath: Path to CSV file to write records to.
    :type filepath: Path
    """

    # Create parent directories if missing:
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)

    # Unpack given result records
    timestamps = results['timestamps']
    measurements = results['measurements']

    # Format the file header comment
    measurement_size = measurements.shape[1]
    file_columns = ['Timestamp']
    file_columns += [f'Measurement [{i+1}]' for i in range(measurement_size)]
    file_header = ', '.join(file_columns)

    # Concatenate and write results to csv file
    data = np.column_stack((timestamps, measurements))
    np.savetxt(filepath, data, fmt='%.18e', delimiter=',', header=file_header)


# Run if script is called directly
if __name__ == '__main__':

    # 1. Generate low-resolution trajectory
    print('[1/7] Generating trajectory...')
    trajectory = get_trajectory()

    # 2. Define sensors to use
    print('[2/7] Initialising sensor instances...')
    accelerometer = get_accelerometer(freq=500.0)
    gyroscope = get_gyroscope(freq=500.0)
    altimeter = get_altimeter(freq=1.0)

    # 3. Extract accelerometer measurements
    print('[3/7] Capturing accelerometer measurements...')
    acc_measurements = get_measurements(accelerometer, trajectory)

    # 4. Extract gyroscope measurements
    print('[4/7] Capturing gyroscope measurements...')
    gyro_measurements = get_measurements(gyroscope, trajectory)

    # 5. Extract altimeter measurements
    print('[5/7] Capturing altimeter measurements...')
    alt_measurements = get_measurements(altimeter, trajectory)

    # 6. Save measurement results to disk
    print('[6/7] Saving measurements...')
    output_dir = Path('sensor_data')  # Change me to prevent overwriting results
    save_measurements(acc_measurements, output_dir / 'accelerometer.csv')
    save_measurements(gyro_measurements, output_dir / 'gyroscope.csv')
    save_measurements(alt_measurements, output_dir / 'altimeter.csv')

    # 7. Save ground truth records to disk
    print('[7/7] Saving ground truth...')
    save_trajectory(trajectory, output_dir / 'ground_truth.csv')

    # 8. Finish and any clean-up
    print(f'\n\nResults saved to: {output_dir.absolute()}')
