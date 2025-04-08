"""
=========
config.py
=========

:summary:
    Obtains primary simulation components from configuration file.
    This module is used for initialising and obtaining the main
    components used in simulations and for logging details to
    the console.

:author:
    Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First full implemented version.
"""

from pathlib import Path

from qnav.estimation.state import EstimatedState
from qnav.gravity.map import set_default_grid
from qnav.input.config_handler import ConfigHandler
from qnav.measurement.clock import Clock
from qnav.measurement.sensor import SensorFusion
from qnav.simulation.display import Display
from qnav.simulation.results import ResultsCapture
from qnav.waypoints.trajectory import Trajectory

import qnav.input.ini.altimeter_config as alt_conf
import qnav.input.ini.estimation_config as est_conf
import qnav.input.ini.gps_config as gps_conf
import qnav.input.ini.gravity_config as grav_conf
import qnav.input.ini.holonomic_config as hol_conf
import qnav.input.ini.measurement_config as meas_conf
import qnav.input.ini.quantum_config as qs_conf
import qnav.input.ini.quantum_grav_config as qmm_conf
import qnav.input.ini.vehicle_config as vc_conf
import qnav.input.ini.waypoint_config as way_conf
import qnav.input.ini.output_config as out_conf


# The default configuration file to assume.
__DEFAULT_CONFIG_FILE = Path("default_settings.ini")

# The column width to use
__BANNER_WIDTH = 64


def get_configuration(config_file: Path = None) -> ConfigHandler:
    """
    Initialises and returns a :class:`qnav.input.config.ConfigHandler`
    instance using the contents of the given file. On succession, the
    file's variables, along with those in any sub-configurations,
    will be parsed and loaded. If not file is given, a default
    settings file will be used.

    :param config_file: A path for a configuration (.ini) file to use.
    :type config_file: Path

    :return: An initialised ConfigHandler instance.
    :rtype: ConfigHandler
    """

    # If none provided, use default config file:
    if config_file is None:
        config_file = __DEFAULT_CONFIG_FILE

    print_banner("Initialisation")
    print_line("Configuration File", config_file)

    # Initialise a configuration file handler.
    config = ConfigHandler(config_file)

    # A collection of sections and properties containing the
    # sub-configuration files to auto read and import.
    sub_configs = [
        ("Waypoints", "waypointConfig", None),
        ("Vehicle",  "vehicleConfig", None),
        ("Measurement", "accelerometerConfig", "Accelerometer"),
        ("Measurement", "gyroscopeConfig", "Gyroscope"),
        ("Measurement", "clockConfig", "Clock"),
        ("Estimation", "accelerometerConfig", "Accelerometer"),
        ("Estimation", "gyroscopeConfig", "Gyroscope"),
        ("Estimation", "clockConfig", "Clock"),
        ("Altimeter", "altimeterConfig", None),
        ("Quantum", "quantumImuConfig", None),
        ("Quantum", "quantumGravityConfig", None),
        ("GPS", "gpsZoneConfig", "GpsZones"),
    ]

    for section, prop, sub_section in sub_configs:
        config.expand_section(section, prop, sub_section)

    # If requested by the user, display warnings about the configuration.
    config.warn_on_fallback = config.get_bool(
        "Misc", "displayFallbackWarnings", False)

    print_line("Number of Variables Read", len(config.get_all_values()))

    # Finally, return the configuration handler instance.
    return config


def get_trajectory(config: ConfigHandler) -> Trajectory:
    """
    Extracts values and produces ground truth trajectories.
    When called, this function reads the required variables from the
    given configuration and uses them to generate a trajectory. On
    succession, this is return ready for use in a simulation.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :return: The trajectory ready for use in a simulation.
    :rtype: Trajectory
    """

    # TODO: Add start and end times
    print_banner("Trajectory")
    vehicle = vc_conf.get_vehicle_model(config)
    print_line("Vehicle Model", vehicle)

    gravity = grav_conf.get_true_gravity_model(config)

    # TODO: NEW
    # Set the default grid for calculating gravity gradient
    grid_params = grav_conf.get_gradient_grid(config)
    set_default_grid(**grid_params)

    if gravity.base_model is None:
        print_line("Gravity Function", gravity)
        print_line("Gravity Correction", 'N/A')
    else:
        print_line("Gravity Function", gravity.base_model)
        print_line("Gravity Correction", gravity)

    mode = way_conf.get_waypoint_mode(config)
    interp = way_conf.get_interp_method(config)
    print_line("Generation Mode", mode)
    print_line("Interpolation", interp)

    return way_conf.generate_trajectory(config, vehicle, gravity)


def get_estimation(config: ConfigHandler, trajectory: Trajectory) -> EstimatedState:
    """
    Extracts values and produces the initial estimation state.
    When called, this function reads the required variables from the
    given configuration and uses them to initialise the starting
    estimation information. This uses the first record from the
    previously generated trajectory and induces any requested errors.
    On succession, an initial estimated state is return.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :param trajectory: The trajectory to be used during the simulation.
        Required to obtain the true state at the first time step.
    :type trajectory: Trajectory

    :return: The initial estimated state to use in the simulation.
    :rtype: EstimatedState
    """

    print_banner("Estimation")
    ins_method = est_conf.get_ins_str(config)
    print_line("IMU Integration", ins_method.title())

    # Initialise at correct start time
    ground_truth = trajectory.get_record(0)
    gravity = grav_conf.get_estimated_gravity_model(config)

    if gravity.base_model is None:
        print_line("Gravity Function", gravity)
        print_line("Gravity Correction", 'N/A')
    else:
        print_line("Gravity Function", gravity.base_model)
        print_line("Gravity Correction", gravity)

    estimated_state = est_conf.get_estimated_state(config, ground_truth, gravity)
    est_conf.add_estimation_errors(config, estimated_state)

    has_errors = est_conf.has_estimation_errors(config)
    print_line("Perfect Initialisation", ~has_errors)

    return estimated_state


def get_platform_hardware(config: ConfigHandler, estimated_state: EstimatedState) -> list[SensorFusion]:
    """
    Extracts values and produces the virtual platform hardware instances.
    When called, this function reads the required variables from the
    given configuration and uses them initialize and return the requested
    Sensor-Fusion instance to use during the simulation. On succession,
    a list of SensorFusion classes to use in a simulation are returned.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :param estimated_state: The initial estimates before start of simulation.
        Required as some fusion methods need an initial estimate to start with.
    :type estimated_state: EstimatedState

    :return: The list of SensorFusion instances to use in a simulation.
    :rtype: list[SensorFusion]
    """

    print_banner("Hardware")
    accelerometer = meas_conf.get_accelerometer(config)
    gyroscope = meas_conf.get_gyroscope(config)
    ins_sf = est_conf.get_ins(config, accelerometer, gyroscope)
    print_line("INS Instance", __class_name(ins_sf))

    altimeter_sf = alt_conf.get_altimeter_sf(config)
    print_line("Altimeter Sensor", __class_name(altimeter_sf))

    gps_sf = gps_conf.get_gps_sf(config, estimated_state)
    print_line("GPS Sensor", __class_name(gps_sf))

    q_imu_sf = qs_conf.get_quantum_imu_sf(config, accelerometer, gyroscope)
    print_line("Quantum IMU", __class_name(q_imu_sf))

    q_grav_sf = qmm_conf.get_quantum_grav_sf(config)
    print_line("Quantum GMM", __class_name(q_grav_sf))

    holonomic_sf = hol_conf.get_holonomic_sf(config)
    print_line("Holonomic Constrains", __class_name(holonomic_sf))

    hardware = [ins_sf, altimeter_sf, gps_sf,
                q_imu_sf, q_grav_sf, holonomic_sf]

    # Return all valid
    return [h for h in hardware if h is not None]


def get_platform_clock(config: ConfigHandler) -> Clock:
    """
    Extracts values and produces a virtual platform clock instances.
    When called, this function reads the required variables from the
    given configuration and uses them to initialise and return the
    platform clock to use.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :return: The Clock instance to use in a simulation.
    :rtype: Clock
    """
    return meas_conf.get_clock(config)


def get_collector(config: ConfigHandler) -> ResultsCapture:
    """
    Extracts values and produces the results collection instance to use.
    When called, this function reads the required variables from the
    given configuration and uses them initialise and return a results
    collection instance, responsible for filtering and collecting
    simulation results.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :return: The ResultsCapture instance to use in a simulation.
    :rtype: ResultsCapture
    """
    return out_conf.get_results_capture(config)


def get_displayer(config: ConfigHandler) -> Display:
    """
    Extracts values and produces the displaying instance to use.
    When called, this function reads the required variables from the
    given configuration and uses them initialise and return a display
    instance, responsible for feeding back real-time information about
    the simulation. Can also be used for stopping the simulation
    prematurely when criteria is met.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :return: The Display instance to use in a simulation.
    :rtype: Display
    """
    return out_conf.get_displayer(config)


def get_results_settings(config: ConfigHandler) -> dict:
    """
    Extracts values and produces the results generation settings.
    When called, this function reads the required variables from the
    given configuration and uses them to define how results will be
    gathered after the simulation and exported.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :return: A dictionary containing settings for exporting results.
    :rtype: dict
    """

    print_banner("Exporting Results")

    settings = {
        'output_dir': out_conf.get_output_dir(config),
        'copy_config': out_conf.get_copy_config(config),
        'down_sampling': out_conf.get_output_downsample(config),
        'formats': out_conf.get_output_formats(config),
    }

    print_line(f"Include Full Configuration", settings['copy_config'])
    print_line(f"Down Sample Ratio", settings['down_sampling'])

    for i, fmt in enumerate(settings['formats']):
        print_line(f"Export Format {i + 1}", fmt)

    return settings


def get_figure_settings(config: ConfigHandler) -> dict:
    """
    Extracts values to use for results figure and report generation.
    When called, this function reads the required variables from the
    given configuration and returns them in a dictionary. These can
    then be used later for generating the relevant after-simulation
    graphics and report results.

    :param config: The ConfigHandler instance to use for simulation settings.
    :type config: ConfigHandler

    :return: Dictionary containing figure generation settings.
    :rtype: dict
    """

    print_banner("Figure Generation")

    settings = {
        'do_save': out_conf.get_save_figures(config),
        'do_show': out_conf.get_show_figures(config),
        'renderer': out_conf.get_figure_renderer(config),
        'formats': out_conf.get_figure_formats(config)
    }

    print_line(f"Saving Figures", settings['do_save'])
    print_line(f"Render Figures", settings['do_show'])
    print_line(f"Selected Render", settings['renderer'])
    return settings


def print_banner(text: str = None):
    """
    Used to print formatted heading in the output console.
    Simply used to group list points together cleanly.

    :param text: (Optional) The text to include in the heading.
        If not provided a dividing line will be used.
    :type text: str
    """
    if text is None:
        print(f"\n{"":─<{__BANNER_WIDTH}}")
    else:
        print(f"\n{f'[ {text} ]':─^{__BANNER_WIDTH}}")


def print_line(field: str, value: any):
    """
    Used to print formated lines in the output console.
    Simply used to produce clean uniformly spaced list points.

    :param field: The text to place on the left-hand side.
    :type field: str

    :param value: The value to place on the right-hand side.
    :type value: any
    """
    col_width = __BANNER_WIDTH - (len(field) + 6)
    value_str = f" {str(value)}"
    print(f" • {field}: {value_str:.>{col_width}} ")


def __class_name(obj: object) -> str:
    """
    Extracts the printable name for a given class.
    Simply used to produce clean text output to the console.

    :param obj: The object to obtain the printable name for.
    :type obj: object

    :return: The printable name of the object.
    :rtype: str
    """
    return "N/A" if obj is None else type(obj).__name__
