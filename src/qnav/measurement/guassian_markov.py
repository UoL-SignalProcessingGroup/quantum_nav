
from qnav.measurement.accelerometer import Accelerometer
from qnav.measurement.error_properties import ErrorProperties
from qnav.measurement.gyroscope import Gyroscope
from qnav.measurement.platform import SensorAxis
from qnav.util.rng import GaussianMarkov


class AccelerometerGM(Accelerometer):

    def __init__(self,
                 frequency: float,
                 error_profile: ErrorProperties,
                 correlation_time: float,
                 pow_spec_density: float,
                 sensor_axis: SensorAxis = SensorAxis(),
                 start_time: float = 0.0,
                 rand_seed: int = None):

        # Call the parent class constructor
        super().__init__(frequency, error_profile, sensor_axis, start_time, rand_seed)
        self._bias_drift_rate = 0 # TODO: REMOVE ME!

        # Initialise a Gaussian Markov RNG for producing white noise
        self._gm_rng = GaussianMarkov(correlation_time, pow_spec_density,
                                      3, start_time, rand_seed)


    def update(self, time_sec: float = None):

        # Call parent class method
        super().update(time_sec)

        # Replace the value for measurement noise
        clock_time = self.time_step if time_sec is None else time_sec
        noise = self._gm_rng.get_random(clock_time)
        self._meas_noise = self._avg_meas_noise_mg * noise
        # self._meas_noise = noise


class GyroscopeGM(Gyroscope):

    def __init__(self,
                 frequency: float,
                 error_profile: ErrorProperties,
                 correlation_time: float,
                 pow_spec_density: float,
                 sensor_axis: SensorAxis = SensorAxis(),
                 start_time: float = 0.0,
                 rand_seed: int = None):

        # Call the parent class constructor
        super().__init__(frequency, error_profile, sensor_axis, start_time, rand_seed)
        self._bias_drift_rate = 0  # TODO: REMOVE ME!

        # Initialise a Gaussian Markov RNG for producing white noise
        self._gm_rng = GaussianMarkov(correlation_time, pow_spec_density,
                                      3, start_time, rand_seed)

    def update(self, time_sec: float = None):

        # Call parent class method
        super().update(time_sec)

        # Replace the value for measurement noise
        clock_time = self.time_step if time_sec is None else time_sec
        noise = self._gm_rng.get_random(clock_time)
        self._meas_noise = self._avg_meas_noise_md * noise
        # self._meas_noise = noise

