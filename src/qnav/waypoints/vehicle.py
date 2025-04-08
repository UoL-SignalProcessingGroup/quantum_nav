"""
==========
vehicle.py
==========

:summary:
    Represents vehicle properties for waypoint generation.
    This module contains a template Vehicle tuple listing all the properties
    applied during waypoint generation. In addition to this a handful of
    integrated vehicle profiles are include for testing. It is recommended
    that a custom Vehicle instance is created for finer control.

:authors:
    | Prof. Jason Ralph - jfralph@liverpool.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.1.0 - Added integrated vehicle types for testing.
"""

from typing import NamedTuple
from math import inf


class Vehicle(NamedTuple):
    """
    A structure used for holding vehicle model information.
    This is used to describe limitations, offsets and expected
    noise/oscillations. Unlike a normal class, the variables held within this
    structure are immutable and cannot be changed. This is done to enforce
    restrictions without them being accidentally overwritten later.
    """

    # Model descriptive name
    description: str = "Unnamed"

    # Vehicle maneuvering limitations:
    turn_rate_max: float = 5.0
    acceleration_max: float = 2.0
    deceleration_max: float = 1.0
    time_delay_acc: float = 1.0
    time_delay_turn: float = 5.0

    # Pitch oscillations:
    pitch_constant_offset: float = 0
    pitch_osc_period: float = 0
    pitch_osc_magnitude: float = 0
    pitch_max: float = 0

    # Roll oscillations:
    roll_constant_offset: float = 0
    roll_osc_period: float = 0
    roll_osc_magnitude: float = 0
    roll_max: float = 0
    d_roll_d_turn_rate: float = 0

    # Vibrations:
    vibration_noise_acceleration: float = 0
    vibration_noise_angle_rates: float = 0
    vibration_damping_period: float = 0

    # Variations:
    variation_per_second1: float = 0.005
    variation_per_second2: float = 0.01

    # Random number generation:
    rand_seed: int = 0

    def __str__(self):
        return self.description


def get_default_profile(rand_seed: int = 0) -> Vehicle:
    """
    Returns a Vehicle instance representing the default profile.
    This profile is for a typical aircraft with no noise/vibrations.

    :param rand_seed: The seed to use for random number generation.
    :type rand_seed: int

    :return: The default Vehicle profile.
    :rtype: Vehicle
    """
    return Vehicle(description="Default", rand_seed=rand_seed)


def get_large_aircraft_profile(rand_seed: int = 0) -> Vehicle:
    """
    Returns a Vehicle instance representing that of a large aircraft.
    This profile supplies the parameters for a 737 type of air-vehicle,
    stating slow acceleration and deceleration with wide turning.

    :param rand_seed: The seed to use for random number generation.
    :type rand_seed: int

    :return: The Vehicle profile for a large aircraft.
    :rtype: Vehicle
    """

    turn_rate_max = 10.0
    roll_max = 30.0

    return Vehicle(
        description="Large Aircraft",
        turn_rate_max=turn_rate_max,
        acceleration_max=0.2*9.81,
        deceleration_max=0.1*9.81,
        time_delay_acc=0.5,
        time_delay_turn=0.5,
        pitch_constant_offset=2.0,
        pitch_osc_period=0.0,
        pitch_osc_magnitude=0.0,
        pitch_max=45.0,
        roll_constant_offset=0.0,
        roll_osc_period=0.0,
        roll_osc_magnitude=0.0,
        roll_max=roll_max,
        d_roll_d_turn_rate=roll_max/turn_rate_max,
        vibration_noise_acceleration=0.05,
        vibration_noise_angle_rates=0.5,
        vibration_damping_period=1.0,
        variation_per_second1=0.005,
        variation_per_second2=0.01,
        rand_seed=rand_seed
    )


def get_large_ship_profile(rand_seed: int = 0) -> Vehicle:
    """
    Returns a Vehicle instance representing that of a large aircraft.
    This profile supplies the parameters for a large ship, such as a cargo
    vessel.

    :param rand_seed: The seed to use for random number generation.
    :type rand_seed: int

    :return: The Vehicle profile for a large ship.
    :rtype: Vehicle
    """

    turn_rate_max = 0.5
    roll_max = 5.0

    return Vehicle(
        description="Large Ship",
        turn_rate_max=turn_rate_max,
        acceleration_max=0.1*9.81,
        deceleration_max=0.1*9.81,
        time_delay_acc=2.0,
        time_delay_turn=2.0,
        pitch_constant_offset=0.0,
        pitch_osc_period=6.0,
        pitch_osc_magnitude=0.5,
        pitch_max=2.5,
        roll_constant_offset=0.0,
        roll_osc_period=10.0,
        roll_osc_magnitude=2.0,
        roll_max=roll_max,
        d_roll_d_turn_rate=-roll_max/turn_rate_max,
        vibration_noise_acceleration=0.05,
        vibration_noise_angle_rates=0.5,
        vibration_damping_period=1.0,
        variation_per_second1=0.005,
        variation_per_second2=0.01,
        rand_seed=rand_seed
    )


def get_large_van_profile(rand_seed: int = 0) -> Vehicle:
    """
    Returns a Vehicle instance representing that of a large van.
    This profile supplies the parameters for a large ground vehicle, such as
    a heavy van vessel, slow acceleration/deceleration and little turning
    limitations.

    :param rand_seed: The seed to use for random number generation.
    :type rand_seed: int

    :return: The Vehicle profile for a large van.
    :rtype: Vehicle
    """

    turn_rate_max = 5.0
    roll_max = 5.0

    return Vehicle(
        description="Large Van",
        turn_rate_max=turn_rate_max,
        acceleration_max=0.25*9.81,
        deceleration_max=1.00*9.81,
        time_delay_acc=0.25,
        time_delay_turn=1.0,
        pitch_constant_offset=0.0,
        pitch_osc_period=0.0,
        pitch_osc_magnitude=0.0,
        pitch_max=33.0,
        roll_constant_offset=0.0,
        roll_osc_period=10.0,
        roll_osc_magnitude=2.0,
        roll_max=roll_max,
        d_roll_d_turn_rate=-roll_max/turn_rate_max,
        vibration_noise_acceleration=0.05,
        vibration_noise_angle_rates=0.5,
        vibration_damping_period=1.0,
        variation_per_second1=0.005,
        variation_per_second2=0.01,
        rand_seed=rand_seed
    )


def get_submarine_profile(rand_seed: int = 0) -> Vehicle:
    """
    Returns a Vehicle instance representing that of a submarine.
    This profile supplies the parameters for a submarine vessel.

    :param rand_seed: The seed to use for random number generation.
    :type rand_seed: int

    :return: The Vehicle profile for a submarine vessel.
    :rtype: Vehicle
    """

    return Vehicle(
        description="Submarine",
        turn_rate_max=0.5,
        acceleration_max=0.05*9.81,
        deceleration_max=0.05*9.81,
        time_delay_acc=2.0,
        time_delay_turn=2.0,
        pitch_constant_offset=0.0,
        pitch_osc_period=0.0,
        pitch_osc_magnitude=0.0,
        pitch_max=10.0,
        roll_constant_offset=0.0,
        roll_osc_period=0.0,
        roll_osc_magnitude=0.0,
        roll_max=5.0,
        d_roll_d_turn_rate=0.0,
        vibration_noise_acceleration=0.0,
        vibration_noise_angle_rates=0.0,
        vibration_damping_period=0.0,
        variation_per_second1=0.005,
        variation_per_second2=0.01,
        rand_seed=rand_seed
    )


def get_unrestricted_profile() -> Vehicle:
    """
    Returns a Vehicle instance with no movement restrictions
    This profile is solely used for following tightly placed waypoints in
    situations where placing vehicle limitations would be impractical. No
    vehicle noise or vibrations are contained in this profile.

    :return: The Vehicle profile with no limitations or noise.
    :rtype: Vehicle
    """

    return Vehicle(
        description="Unrestricted",
        turn_rate_max=inf,
        acceleration_max=inf,
        deceleration_max=inf,
        time_delay_acc=1e-06,
        time_delay_turn=1e-06,
        pitch_constant_offset=0.0,
        pitch_osc_period=0.0,
        pitch_osc_magnitude=0.0,
        pitch_max=90.0,
        roll_constant_offset=0.0,
        roll_osc_period=0.0,
        roll_osc_magnitude=0.0,
        roll_max=180.0,
        d_roll_d_turn_rate=0.0,
        vibration_noise_acceleration=0.0,
        vibration_noise_angle_rates=0.0,
        vibration_damping_period=0.0,
        variation_per_second1=0.0,
        variation_per_second2=0.0,
        rand_seed=0
    )


if __name__ == '__main__':
    p = get_unrestricted_profile()
