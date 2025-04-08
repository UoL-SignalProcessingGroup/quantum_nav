"""
============
satellite.py
============

:summary:
    All GPS satellite and solver functionalities.
    This module is responsible for simulating up-to-date satellites and
    supplying a solver to produce measurements (with configured errors
    and noise). During a simulation these classes will be used for
    acquiring GPS information.

:authors:
    | Dr. Niall Moroney - niall.moroney17@imperial.ac.uk
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    0.9.0 - Awaiting further implementations.
"""

import qnav.util.transformations as trans
import qnav.util.constants as const
import numpy as np
import math

from datetime import datetime
from datetime import timedelta
from collections import namedtuple


#: The field names for satellite parameters.
satellite_param_fields = (
    'type', 'id', 'year', 'month', 'day', 'hour', 'minute', 'second',
    'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
    't_0e', 'prn', 'delta_n', 'i_dot', 'omega_dot', 'omega_0',
    'c_us', 'c_uc', 'c_is', 'c_ic', 'c_rs', 'c_rc',
    'sqrt_a', 'e', 'i_0', 'w', 'm_bar_0')

#: A collection of satellite parameters needed to initialise a Satellite instance.
SatelliteParams = namedtuple('SatelliteParams', satellite_param_fields)


class Satellite:
    """
    Represents a GPS satellite and its current states/properties.
    This class is responsible for representing information about a GPS
    satellite. Methods are supplied for updating these states and allowing
    for measurement information to be acquired. During simulations a
    collection of satellite instances will be used for generating a GPS signal
    that will be used to improve navigation accuracy.
    """

    # All properties of the class
    __slots__ = (
        '_params',
        '_gps_time',
        '_pseudo_range_noise',
        '_pseudo_range_rate_noise',
        '_position_ecef',
        '_velocity_ecef',
        '_rho_range_noise',
        '_rho_range_rate_noise',
        '_rng'
    )

    def __init__(self,
                 params: SatelliteParams,
                 gps_time: datetime = None,
                 pseudo_range_noise: float = 0,
                 pseudo_range_rate_noise: float = 0,
                 rand_seed: int = None):
        """
        Generates a Satellite instance from given information.

        :param params: The satellite parameters extracted from a
            RINEX/Ephemeris file.
        :type params: SatelliteParams

        :param gps_time: The current time of the GPS satellite.
        :type gps_time: datetime.datetime

        :param pseudo_range_noise: The amount of pseudo range noise to be
            applied to measurements.
        :type pseudo_range_noise: float

        :param pseudo_range_rate_noise: The amount of drift to use when
            updating pseudo range noise.
        :type pseudo_range_rate_noise: float

        :param rand_seed: The optional initialisation seed to use to
            generate random numbers for noise and other errors.
        :type rand_seed: int
        """

        self._params = params
        self._gps_time = gps_time
        self._pseudo_range_noise = pseudo_range_noise
        self._pseudo_range_rate_noise = pseudo_range_rate_noise

        self._position_ecef = np.full(3, np.nan)
        self._velocity_ecef = np.full(3, np.nan)
        self._rng = np.random.default_rng(rand_seed)
        self._rho_range_noise = 0
        self._rho_range_rate_noise = 0

        # Use recorded satellite time if not give:
        if self._gps_time is None:
            self._gps_time = self._get_param_datetime()

        # Bootstrap satellite state
        self.update(0)

    def update(self, dt: float):
        """
        Updates the position, velocity and errors given the time difference.
        This method updates the state of the satellite by a given amount of
        time (in seconds). The updated position and velocity of the satellite
        are calculated along with the new error states for generating noisy
        measurements.

        :param dt: The time difference since the last update was requested.
        :type dt: float
        """

        # Increase the GPS clock time
        # if dt < 0: raise ValueError('dt must be greater than zero')
        self._gps_time += timedelta(seconds=dt)

        # Update the measurement noise
        self._update_noise()

        gm = const.GM
        omega_e = const.OMEGA_E

        # TODO: Check as only seconds are used
        gps_time = trans.utc2gps(self._gps_time)

        # Time (seconds)
        t_k = gps_time['seconds'] - self._params.t_0e

        # Earth semi-major axis (meters)
        a = self._params.sqrt_a ** 2

        # Computed mean motion
        n_0 = (gm / a ** 3) ** 0.5

        # Corrected mean motion
        n = n_0 + self._params.delta_n

        # Mean anomaly
        m_bar_k = self._params.m_bar_0 + n * t_k

        # Solve Kepler's equation of eccentric anomaly by iteration
        e = self._params.e
        e_k = m_bar_k
        e_old = m_bar_k
        d_e = 1

        while d_e > 1e-9:
            e_k = m_bar_k + e * math.sin(e_old)
            d_e = abs(e_k - e_old)
            e_old = e_k

        # ---------------------------------------------------------------------- #

        # Following method described in:
        # https://www.navipedia.net/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
        nu_k = math.atan2((math.sin(e_k) * math.sqrt(1 - e ** 2)), (math.cos(e_k) - e))

        # Argument of Latitude
        phi_k = nu_k + self._params.w
        cos_2_phi_k = math.cos(2 * phi_k)

        # Argument of Latitude correction
        delta_u_k = self._params.c_uc * cos_2_phi_k + self._params.c_us * cos_2_phi_k

        # Radius correction
        delta_r_k = self._params.c_rc * cos_2_phi_k + self._params.c_rs * cos_2_phi_k

        # Inclination correction
        delta_i_k = self._params.c_ic * cos_2_phi_k + self._params.c_is * cos_2_phi_k

        # Corrected argument of latitude
        # u_k = (phi_k + delta_u_k) % (2 * np.pi)
        u_k = (phi_k + delta_u_k) % (2 * math.pi)

        # Corrected radius
        r_k = a * (1 - e * math.cos(e_k)) + delta_r_k

        # Corrected inclination
        i_k = (self._params.i_0 + self._params.i_dot * t_k + delta_i_k)

        # Position in orbital plane
        x_prime_k = r_k * math.cos(u_k)
        y_prime_k = r_k * math.sin(u_k)

        # Corrected longitude of ascending node
        omega_k = ((self._params.omega_0 + t_k * (self._params.omega_dot - omega_e)
                    - self._params.t_0e * omega_e) % (2 * math.pi))

        # Earth fixed geocentric satellite coordinates
        x_k = (x_prime_k * math.cos(omega_k) - y_prime_k * math.sin(omega_k) * math.cos(i_k))
        y_k = (x_prime_k * math.sin(omega_k) - y_prime_k * math.cos(omega_k) * math.cos(i_k))
        z_k = y_prime_k * math.sin(i_k)

        # Update the position of the satellite
        self._position_ecef = np.array([x_k, y_k, z_k])

        # ---------------------------------------------------------------------- #

        # Satellite velocity calculation as per method in "Principals of
        # GNSS, Inertial, and Multi-sensor Integrated Navigation Systems" - Mason
        e_dot = ((n_0 + self._params.delta_n) / (1 - self._params.e * math.cos(e_k)))

        phi_dot = e_dot * math.sin(nu_k) / math.sin(e_k)

        r_dot = (a * self._params.e * math.sin(e_k) * e_dot
                 + 2 * phi_dot * self._params.c_rs * math.cos(2 * phi_k)
                 - self._params.c_rc * math.sin(2 * phi_k))

        u_dot = ((1 + 2 * self._params.c_us * math.cos(2 * phi_k) - 2
                  * self._params.c_uc * math.sin(2 * phi_k)) * phi_dot)

        x_prime_dot = r_dot * math.cos(u_k) - r_k * u_dot * math.sin(u_k)
        y_prime_dot = r_dot * math.sin(u_k) + r_k * u_dot * math.cos(u_k)

        i_k_dot = (self._params.i_dot + 2 * phi_dot *
                   (self._params.c_is * math.cos(2 * phi_k)
                    - self._params.c_ic * math.sin(2 * phi_k)))

        x_dot = (x_prime_dot * math.cos(omega_k)
                 - y_prime_dot * math.cos(i_k) * math.sin(omega_k)
                 + i_k_dot * y_prime_k * math.sin(i_k) * math.sin(omega_k) +
                 (omega_e - self._params.omega_dot) *
                 (x_prime_k * math.sin(omega_k) + y_prime_k * math.cos(i_k) *
                  math.cos(omega_k)))

        y_dot = (x_prime_dot * math.sin(omega_k)
                 + y_prime_dot * math.cos(i_k) * math.cos(omega_k)
                 - i_k_dot * y_prime_k * math.sin(i_k) * math.cos(omega_k) +
                 (omega_e - self._params.omega_dot) *
                 (-x_prime_k * math.cos(omega_k) + y_prime_k * math.cos(i_k) *
                  math.sin(omega_k)))

        z_dot = y_prime_dot * math.sin(i_k) + i_k_dot * y_prime_k * math.cos(i_k)

        # Update the velocity of the satellite
        self._velocity_ecef = np.array([x_dot, y_dot, z_dot])

        # ---------------------------------------------------------------------- #


    def _get_pseudo_ranges(self,
                           ant_pos_lla: np.ndarray,
                           ant_att_deg: np.ndarray,
                           ant_vel_body_axes: np.ndarray) -> tuple:
        """
        Calculates the true pseudo ranges between the satellite and the given
        antenna location.

        .. TODO:
            Include correct noise of velocities?

        :param ant_pos_lla: The true antenna position (latitude, longitude
            and altitude given in deg-deg-metres).
        :type ant_pos_lla: numpy.array (3-elements)

        :param ant_att_deg: The true antenna attitude (heading, pitch and roll
            in degrees).
        :type ant_att_deg: numpy.array (3-elements)

        :param ant_vel_body_axes: The true antenna velocity (in body axes in
            metres/second).
        :type ant_vel_body_axes: numpy.array (3-elements)

        :return: The true rho and rho rate values.
        :rtype: tuple (2-elements)
        """

        sat_pos_ecef = self._position_ecef
        sat_vel_ecef = self._velocity_ecef
        ant_pos_ecef = trans.lla2ecef(ant_pos_lla)

        rot_mat = trans.rotate_3d(*np.radians(ant_att_deg))
        ant_vel_ned = np.linalg.inv(rot_mat) @ ant_vel_body_axes
        ant_vel_ecef = trans.vned2vecef(ant_vel_ned, ant_pos_lla)

        # Relative position of satellite from antenna
        d_p = sat_pos_ecef - ant_pos_ecef
        rho_true = np.sqrt(d_p.dot(d_p))
        vel_rel_true = (sat_vel_ecef - ant_vel_ecef)
        los_vector_true = d_p / rho_true
        # vel_rel_true = np.array(vel_rel_true)

        # TODO: Check changes (removed loop)
        rho_rate_true = los_vector_true.dot(vel_rel_true)

        return rho_true, rho_rate_true

    def _update_noise(self):
        self._rho_range_noise = self._pseudo_range_noise * self._rng.normal()
        self._rho_range_rate_noise = self._pseudo_range_rate_noise * self._rng.normal()


    def get_location_azimuth_elevation(self, ant_pos_lla: np.ndarray) -> tuple[float, float]:
        """
        Get the relative location of the satellite from the current position,
        in azimuth and elevation (deg-deg).

        :param ant_pos_lla: The true antenna position (latitude, longitude
            and altitude given in deg-deg-metres).
        :type ant_pos_lla: numpy.array (3-elements)

        :return: Returns the corresponding azimuth and elevation (both in
            degrees).
        :rtype: tuple[float, float]
        """

        # Satellite position in NED from antenna
        sat_pos_ecef = self._position_ecef
        sat_ned = trans.ecef2ned(sat_pos_ecef, ant_pos_lla)
        return trans.ned2azel(sat_ned)

    def get_measurements(self, ant_pos_lla: np.ndarray,
                         ant_att_deg: np.ndarray,
                         ant_vel_body_axes: np.ndarray) -> dict:
        """
        Calculates and returns the noisy pseudo ranges, position, velocity
        and time information for the satellite.

        :param ant_pos_lla: The true antenna position (latitude, longitude
            and altitude given in deg-deg-metres).
        :type ant_pos_lla: numpy.array (3-elements)

        :param ant_att_deg: The true antenna attitude (heading, pitch and roll
            in degrees).
        :type ant_att_deg: numpy.array (3-elements)

        :param ant_vel_body_axes: The true antenna velocity (in body axes in
            metres/second).
        :type ant_vel_body_axes: numpy.array (3-elements)

        :return: The relevant measurement information in a dict.
        :rtype: dict
        """

        # TODO: Add noise to these?
        gps_time = self._gps_time
        sat_pos_ecef = self._position_ecef
        sat_vel_ecef = self._velocity_ecef

        rho_true, rho_rate_true = self._get_pseudo_ranges(
            ant_pos_lla, ant_att_deg, ant_vel_body_axes)

        # GPS Antenna only has access to noisy pseudo range (rate) measurements.
        rho_meas = rho_true + self._rho_range_noise
        rho_rate_meas = rho_rate_true + self._rho_range_rate_noise
        # rho_meas = (rho_true + self._pseudo_range_noise * self._rng.normal())
        # rho_rate_meas = (rho_rate_true + self._pseudo_range_rate_noise * self._rng.normal())

        return {
            'time': gps_time,
            'pseudoRangeNoisy': rho_meas,
            'pseudoRangeRatesNoisy': rho_rate_meas,
            'satellitePosition': sat_pos_ecef,
            'satelliteVelocity': sat_vel_ecef
        }

    # def get_position_and_velocity(self) -> tuple:
    #     """
    #     Returns a tuple containing the current position and velocity of the
    #     satellite in ECEF format. This is namely for debugging and testing.
    #
    #     :return: The current position and velocity of the satellite.
    #     :rtype: tuple
    #     """
    #     return self._position_ecef, self._velocity_ecef

    def _get_param_datetime(self) -> datetime:
        """
        Returns the recorded timestamp of the satellite.
        Obtains the time recorded in the satellite parameters in the format
        of Python datetime.

        :return: The recorded satellite date and time.
        :rtype: datetime.datetime
        """
        time_params = self._params._asdict()
        time_fields = ['year', 'month', 'day', 'hour', 'minute', 'second']
        time_dict = {k: v for k, v in time_params.items() if k in time_fields}
        return datetime(**time_dict)

    @property
    def sat_id(self) -> str:
        """
        Returns the full satellites ID from given parameters.

        :return:
        :rtype: str
        """
        return f"{self._params.type}{self._params.id}"

    @property
    def sat_time(self) -> datetime:
        """
        The satellite's current clock time.

        :return: Current satellite time.
        :rtype: datetime.datetime
        """
        return self._gps_time

    @property
    def position(self) -> np.ndarray:
        """
        The satellite's ECEF position since last update.

        :return: Current ECEF position.
        :rtype: numpy.ndarray
        """
        return self._position_ecef

    @property
    def velocity(self) -> np.ndarray:
        """
        The satellite's ECEF velocity since last update.

        :return: Current ECEF velocity.
        :rtype: numpy.ndarray
        """
        return self._velocity_ecef

    @property
    def pseudo_range_noise(self) -> float:
        """
        Returns the satellite's initial pseudo range noise.

        :return: Pseudo range noise.
        :rtype: float
        """
        return self._pseudo_range_noise

    @property
    def pseudo_range_rate_noise(self) -> float:
        """
        Returns the satellite's initial pseudo range rate noise.

        :return: Pseudo range rate noise.
        :rtype: float
        """
        return self._pseudo_range_rate_noise

if __name__ == '__main__':

    tmp = SatelliteParams("a", "b", 2000, 1, 2, 3, 4, 5,
                          0, 0, 0, 0, 0, 0, 0,0, 0,
                          0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0)


    # satellite_param_fields = (
    #     'type', 'id', 'year', 'month', 'day', 'hour', 'minute', 'second',
    #     'clock_bias', 'drift_freq_bias', 'drift_rate_message_time',
    #     't_0e', 'prn', 'delta_n', 'i_dot', 'omega_dot', 'omega_0',
    #     'c_us', 'c_uc', 'c_is', 'c_ic', 'c_rs', 'c_rc',
    #     'sqrt_a', 'e', 'i_0', 'w', 'm_bar_0')