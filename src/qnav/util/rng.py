"""
======
rng.py
======

:summary:
    Provides alternative means for pseudo-random number generation.
    This module contains implementations for controlled random number
    generation, subject to realistic factors. Currently, this only
    supplies gaussian-markov random walk based number generation.

:authors:
    | Michael Wright - mjwright@liverpool.ac.uk

:version:
    1.0.0 - First implementation of module.
"""

import numpy as np

class GaussianMarkov:
    """
    Gaussian Markov First-Order Random Walk Model.
    Generates controlled random numbers following a Gaussian Markov random
    walk. Can be used to replace typical normal random number generation.
    """

    def __init__(self,
                 corr_time: float,
                 psd: float,
                 output_shape: int | tuple[int] = 1,
                 start_time: float = 0,
                 seed: int = None):
        """
        Construct a Gaussian Markov Random Walk Model.
        This model can be used for sequential random number generation.

        :param corr_time: (T) The correlation time in seconds.
        :type corr_time: float

        :param psd: (q) The power spectral density of the random walk.
        :type psd: float

        :param output_shape: The output shape for the random numbers.
            By default only a single number is generated and returned.
        :type output_shape: int | tuple[int]

        :param start_time: (t) The start time of the random walk.
        :type start_time: float

        :param seed: Seed to use for random number generation.
        :type seed: int
        """

        # Unpack given arguments
        self._last_time: float = start_time
        self._corr_time: float = corr_time
        self._output_shape: int = output_shape
        self._psd: float = psd

        # Initialise random number generation
        self._rng = np.random.default_rng(seed)
        self._last_value: float | np.ndarray | None = None

        # Validate class parameters
        assert self._corr_time > 0, 'Correlation time must be a positive number'

    def get_random(self, clock_time: float) -> float | np.ndarray:
        """
        Generates random number(s) for requested time and shape.
        Calling this method continues the random walk to the given time in
        seconds. Calling with a time earlier than that of the last
        measurement will result in a value error due to illegal state.

        :param clock_time: The current simulation clock time.
        :type clock_time: float

        :return: Randomly generated number(s).
        :rtype: float | np.ndarray
        """

        # Calculate the time difference
        dt = clock_time - self._last_time

        if dt < 0:
            raise ValueError('Clock time is less than time of last generation')

        # if dt == 0:
        #     return self._last_value

        normal = self._rng.normal(0, 1, self._output_shape)

        if self._last_value is None:
            rand = np.sqrt(self._psd * self._corr_time / 2) * normal
        else:
            rand = self._last_value * np.exp(-dt / self._corr_time) \
                + np.sqrt((self._psd * self._corr_time / 2) * (
                    1 - np.exp(-2 * dt / self._corr_time))) * normal

        self._last_time = clock_time
        self._last_value = rand
        return rand

    def reset(self, clock_time: float = 0) -> None:
        """
        Resets the internal state of the random walk, clearing previous
        history and information.

        :param clock_time: Optional new initialisation clock time.
            By default this is 0 seconds.
        :type clock_time: float
        """

        self._last_time = clock_time
        self._last_value = None

    @property
    def output_shape(self) -> int | tuple[int]:
        """
        Returns the output shape numbers are being generated for.
        This is the dimensions of the output of the number generator.

        :return: Output shape for random number generation.
        :rtype: int | tuple[int]
        """
        return self._output_shape

    @property
    def last_value(self) -> float | np.ndarray | None:
        """
        Returns the last random value generated.

        :return: Last value returned by generator.
        :rtype: float | np.ndarray | None
        """
        return self._last_value

    @property
    def last_time(self) -> float:
        """
        Returns time of last number(s) generated in seconds.

        :return: Time of last number(s) generated.
        :rtype: float
        """
        return self._last_time


# For debugging and testing:
if __name__ == "__main__":

    # Specify the number of values to produce.
    num_time_steps = 10000
    num_values = 6

    # Create a new time correlated random number generator.
    rng = GaussianMarkov(1000, 0.1, num_values)

    # Specify the times at which to produce values.
    times = np.linspace(0, 1000, num_time_steps)

    # Initialise an array to record the random values produced.
    values = np.zeros([num_time_steps, num_values])

    # Produce random values for each time step.
    for i, t in enumerate(times):
        values[i, :] = rng.get_random(t)

    # Create a figure plot to show the values produced.
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(11.51, 8.14))

    # Plot each of the random values produced at their given times.
    for i in range(num_values):
        plt.scatter(times, values[:, i], label=f"Random {i+1}")

    # Plot the average random values on top of the random values.
    plt.plot(times, np.mean(values, axis=1), 'r', label="Average")

    # Add labels and legends.
    plt.legend(loc='best', frameon=True)
    plt.ylabel("Random Values Produced")
    plt.xlabel("Time (seconds)")

    # Finally, show the figure.
    plt.show()
