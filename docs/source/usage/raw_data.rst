Raw IMU and GNSS data
=====================

Recorded CSV data is selected with ``[Input] inputMode = raw``. The main
configuration may contain the mappings directly, or point to a reusable
mapping file with ``rawDataConfig``. Relative CSV paths are resolved from the
file containing the mapping.

Each enabled stream maps canonical values to exact CSV header names. A raw INS
run needs both ``[RawAccelerometer]`` and ``[RawGyroscope]``. ``[RawGnss]``
accepts receiver latitude, longitude and altitude, with optional NED velocity
and quality columns. Accelerometer values are normalized to m/s2, gyroscope
values to degrees/s, positions to degrees/metres, and velocity to m/s.

By default, ``imuInputLevel = measurement`` treats the IMU columns as final
sensor measurements. Set ``imuInputLevel = truth`` under ``[RawData]`` to
interpret them as ideal body-frame specific force and ideal body P/Q/R rates.
Truth values are then passed through the accelerometer and gyroscope models
configured under ``[Measurement]``, including bias, drift, scale factor,
non-orthogonality, white noise and optional Gaussian-Markov noise. A stationary
level accelerometer truth record is approximately ``0, 0, -9.81`` m/s2.
Configured random seeds retain their normal reproducibility behavior.

When truth mode is used, ``[Output] saveProcessedImu = yes`` additionally
writes ``processed_accelerometer.csv`` and ``processed_gyroscope.csv``. These
files are streamed during replay, so enabling them does not retain the full
IMU recording in memory.

The initial heading, pitch and roll are required in ``[RawInitialState]``.
Position comes from that section when supplied, otherwise replay begins at the
first accepted GNSS fix and earlier measurements are reported as discarded.

CSV files are read in bounded-memory chunks. ``invalidRecordPolicy = strict``
stops with a file, row and column diagnostic; ``drop`` skips bad records and
counts them in ``ingestion_report.json``. Combined files and separate files are
both supported by setting ``file`` in each sensor section or a shared
``[RawData] combinedFile``.

See ``examples/example_10`` for a runnable combined-file configuration.
