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

Additional recorded sensors use the same section-per-stream style::

    [RawAltimeter]
    timestampColumn = time
    altitudeColumn = pressure_altitude
    altitudeUnits = ft

    [RawQuantumImu]
    timestampColumn = time
    accelerationXColumn = qax
    accelerationYColumn = qay
    accelerationZColumn = qaz
    angularRateXColumn = qp
    angularRateYColumn = qq
    angularRateZColumn = qr
    accelerationUnits = g
    angularRateUnits = rad/s
    frame = sensor

    [RawGravityGradiometer]
    timestampColumn = time
    topSignalColumn = upper_signal
    bottomSignalColumn = lower_signal

Altimeter measurements feed the configured ``[Altimeter]`` fusion. Quantum
IMU replay also requires conventional accelerometer and gyroscope streams and
uses the configured ``[Quantum]`` fusion. Gradiometer columns are the
dimensionless upper and lower interferometer signals used by the configured
quantum-gravity fusion.

Supported input conversions include acceleration in ``m/s2``, ``m/s^2``,
``g`` or ``mg``; angular rates in degrees/s or radians/s; lengths in metres or
feet; velocities in m/s, km/h or knots; and coordinate/attitude angles in
degrees or radians. ``xSign``, ``ySign`` and ``zSign`` may each be ``1`` or
``-1`` when a source axis convention needs reversing. Values are normalized
to QNav's m/s2, degrees/s, metres, m/s and degree conventions before replay.

``timeUnit`` accepts relative seconds, milliseconds, microseconds and
nanoseconds (including common long-form aliases). Numeric Unix/epoch time uses
``unix``, ``unix_s``, ``unix_ms``, ``unix_us`` or ``unix_ns``; the equivalent
``epoch`` aliases are also accepted. ISO/date-time columns use ``datetime`` or
``iso8601`` with optional ``datetimeFormat`` and ``timezone``. Do not mix
relative timestamps with Unix or date-time streams in one run.

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

Initial heading, pitch and roll must come from a complete
``[RawInitialState]`` group or the first reference record. Position follows the
same explicit-then-reference precedence and otherwise uses the first accepted
GNSS fix. Earlier measurements are reported as discarded.

Recorded ground truth
---------------------

An optional ``[RawReference]`` section maps a separately recorded reference
trajectory. Complete position, velocity, acceleration, attitude and angular
rate groups may be mapped independently. Explicit complete groups in
``[RawInitialState]`` take precedence; absent position, velocity and attitude
groups are initialized from the first reference record. Reference records are
used only for evaluation and are never fused into the navigation estimate.

Attitude may be mapped as heading, pitch and roll, or as named scalar-first
quaternion components::

    [RawReference]
    file = ground_truth.csv
    timestampColumn = time
    latitudeColumn = latitude
    longitudeColumn = longitude
    altitudeColumn = altitude
    velocityXColumn = velocity_north
    velocityYColumn = velocity_east
    velocityZColumn = velocity_down
    velocityFrame = ned
    quaternionWColumn = qw
    quaternionXColumn = qx
    quaternionYColumn = qy
    quaternionZColumn = qz
    quaternionDirection = bodyToNed

Quaternion components are dimensionless and normalized during loading.
``quaternionDirection`` is mandatory and accepts ``bodyToNed`` or
``nedToBody``. Named components remove source-order ambiguity. Position angles
and Euler attitude can use separate ``coordinateUnits`` and ``attitudeUnits``;
the existing ``angleUnits`` remains a fallback for both.

Reference velocity accepts ``body``, ``ned``, ``enu`` or ``ecef`` frames and
is normalized to QNav body axes. Non-body velocity requires attitude in the
same reference row, while ECEF velocity additionally requires position.

Reference fields are interpolated onto the retained estimation timestamps:
position in ECEF, vectors linearly, and attitude using shortest-path quaternion
interpolation. Values outside reference coverage are not extrapolated. When a
reference is configured, aligned ``ground_truth.qnr`` and requested exported
formats are produced and used by the existing summaries and comparison plots.

CSV files are read in bounded-memory chunks. ``invalidRecordPolicy = strict``
stops with a file, row and column diagnostic; ``drop`` skips bad records and
counts them in ``ingestion_report.json``. Combined files and separate files are
both supported by setting ``file`` in each sensor section or a shared
``[RawData] combinedFile``.

See ``examples/example_10`` for a runnable combined-file configuration.
