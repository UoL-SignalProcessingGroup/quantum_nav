
User Configuration
==================
Configuration is primarily performed by providing the location of a configuration file to use as the
first command line argument. Examples of how to do this is cover under :ref:`Software Usage`.
This section discusses the various sections contained with a standard configuration files, along with their
purpose and key variables.

In total, there is over 400 customisable variables that can be changed. Except for a few cases, each variable has its
own default value which is used if it is not included in the current configuration file. In the event a configuration
file is not provided, the default configuration ``default_settings.ini`` will be used automatically. It is recommended
that a copy of ``default_settings.ini`` is made and renamed. This copy can serve as a fully commented template
configuration file to use and modify.


.. note:: Configuration files used by the project are formatted as “initialisation” files (.ini format). Because .ini
    files are a popular and commonly used method for software configuration, many editor programs support syntax
    highlighting for them. To avoid making mistakes or typos, it is recommended to use an editor that supports
    initialisation file format.

.. important::
    * Both ``=`` and ``:`` can be used for assigning values.
    * The symbols ``#`` and ``;`` are used to state comments.
    * String values do not require quotes such as ``"`` or ``'``.
    * In-line comments (comments after variables) are **not** supported.
    * For numerical values anything after a string character is ignored.
    * File paths can be given in either Windows or POSIX (Linux) formats.
    * Valid boolean values are ``true``, ``false``, ``yes``, ``no``, ``on``, ``off``, ``1`` or ``0``.
    * Mathematics expressions such as ``+``, ``-``, ``*``, ``/``, ``//``, ``^`` and ``e`` are supported.
    * To reduce the likely-hood of errors and typo, variable names are **case insensitive**.

Configuration Sections
----------------------
A configuration file is composed of 15 sections in total: **Output**, **Database**, **Waypoints**, **Racetrack**,
**Vehicle**, **Gravity**, **Geoid**, **Measurement**, **Estimation**, **Altimeter**, **GPS**, **Quantum**,
**Holonomics**, **Random** and **Misc**. The use of each of these section is discussed below along with a
description for the most common variables used:


Output Settings
...............
The **Output** section holds all variables used for controlling the output of the simulation. This includes
settings for stating what is to be displayed during runtime, what files at to be produced, where files are to be saved,
any down-sampling or compression that should be applied.

.. list-table:: Key Output Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``saveResults``
     - Should summary results be saved to disk?
     - Yes *or* No
   * - ``saveFigures``
     - Should figures be generated and saved to disk?
     - Yes *or* No
   * - ``showFigures``
     - Should figures be generated in the web-browser?
     - Yes *or* No
   * - ``copyConfig``
     - Should a copy of the configuration file be created?
     - Yes *or* No
   * - ``resultsDownSampleRate``
     - A factor to down-sample the collection of results by.
     - Integer
   * - ``outputDownSampleRate``
     - A factor to down-sample the exporting of results data by.
     - Integer
   * - ``resultsPath``
     - The location where all results are to be saved.
     - Filepath
   * - ``appendDateFolder``
     - Should the date be appended to the output?
     - Yes *or* No

.. note:: Toolbox supports the following save file formats:

    * ``csv`` - Comma Seperated Values
    * ``csv_gz`` - Comma Seperated Values with G-Zip compression
    * ``numpy`` - Numpy binary array file (generated with np.save())
    * ``numpy_gz`` - Numpy array file with G-Zip compression
    * ``binary`` - Standard float64 big-endian binary file
    * ``mat`` - MATLAB v7.2 save file without compression
    * ``mat_comp`` - MATLAB v7.2 save file with compression


Database Settings
.................
The **Database** section controls the location where datasets are read from and where they will be written to.

.. list-table:: Key Database Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``geoidDatabase``
     - Directory for the geoid dataset files.
     - Filepath
   * - ``gpsDatabase``
     - Directory for the GPS Rinex dataset files.
     - Filepath
   * - ``geosatGravityDatabase``
     - Directory for the Geosat-44 gravity model files.
     - Filepath
   * - ``marineGravityDatabase``
     - Directory for the Marine gravity model files.
     - Filepath
   * - ``ggmPlusGravityDatabase``
     - Directory for the GGM Plus gravity model files.
     - Filepath
   * - ``srtm2GravityDatabase``
     - Directory for the SRTM2Gravity gravity model files.
     - Filepath
   * - ``irishSeaGravityDatabase``
     - Directory for the Irish Sea gravity model files.
     - Filepath


Waypoints Settings
..................
The **Waypoints** section controls how the true trajectory is generated and used. This is where information of the
effective trajectory path is stated. This also controls the timings and frequency of the data.

.. list-table:: Key Waypoint Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``waypointFrequency``
     - The waypoint/simulated world's update frequency to use.
     - Float
   * - ``waypointInterpolation``
     - The interpolation method to use for ground truth look-up.
     - cubic, pchip *or* akima
   * - ``waypointMode``
     - The mode for reading base waypoints.
     - config, csv *or* racetrack
   * - ``waypointConfig``
     - A waypoint sub-configuration file.
     - Yes *or* No
   * - ``waypointCSV``
     - A waypoint CVS file.
     - Yes *or* No
   * - ``useVirtualMemory``
     - Should disk space be used to hold waypoint data?
     - Yes *or* No


Racetrack Settings
..................
The **Racetrack** section controls how arithmetical looping trajectories will be generated, if the waypoint method
is set to racetrack. This can be useful for quickly experimenting with different settings or profiles, without
having to present a planned route to use. The produced track is a looping trajectory which will be followed until
the simulations maximum time is reached.

.. list-table:: Key Waypoint Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``circuitTime``
     - The time (in seconds) to complete a single lap of the circuit.
     - Float
   * - ``distanceX``
     - The distance span in the X (North) direction (in metres).
     - Float
   * - ``distanceY``
     - The distance span in the Y (East) direction (in metres).
     - Float
   * - ``distanceZ``
     - The distance span in the Z (Down) direction (in metres).
     - Float
   * - ``numVariationsY``
     - The number of loops in the Y axis with respect to the X axis.
     - Float
   * - ``numVariationsZ``
     - The number of loops in the Z axis with respect to the X axis.
     - Float


Vehicle Settings
................
The **Vehicle** section lists all the values used for stating the limitations of the vehicle (model) that is to be use.
This values effect the shape of the requested trajectory, the manoeuvring limitations and the movement noise of the
vehicle.

.. list-table:: Key Vehicle Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``vehicleConfig``
     - The location of the vehicle sub-profile to use.
     - Filepath

Gravity Settings
................
The **Gravity** section controls all gravity related calculations, specifically the gravity functions and gravity
correction maps used in both ground truth and estimation.

.. list-table:: Key Gravity Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``autoDownload``
     - Should missing gravity data be downloaded automatically?
     - Yes *or* No
   * - ``trueGravityFunction``
     - The gravity function to use for the ground truth.
     - See Below
   * - ``trueGravityCorrectionMap``
     - The optional gravity correction map to use for the ground truth.
     - See Below
   * - ``estimatedGravityFunction``
     - The gravity function to use for the simulation estimation.
     - See Below
   * - ``estimatedGravityCorrectionMap``
     - The optional gravity correction map to use for the simulation estimation.
     - See Below

.. note:: Toolbox supports the following gravity function options:

    * ``fixed`` - Simple fixed constant value
    * ``uniform`` - Uniform sphere with height interpolation
    * ``somigliana`` - Somigliana formula for theoretical gravity
    * ``wgs84`` - NIMA's WGS-84 ellipsoid gravity model (used by MATLAB).

.. note:: Toolbox supports the following gravity correction map options:

    * ``none`` - Do not use a map, simple use the function alone.
    * ``geoid`` - Use the EGM model (defined under Geoid section).
    * ``geosat`` - Use the geosat-44 gravity correction map.
    * ``marine`` - Use the marine gravity correction map.
    * ``ggmPlusAcc`` - Use GGMPlus gravity acceleration map.
    * ``ggmPlusDist`` - Use GGMPlus gravity disturbance map.
    * ``srtm2gravityFS`` - Use SRTM2Gravity full-scale gravity map.
    * ``srtm2gravityRes`` - Use SRTM2Gravity residual gravity map.
    * ``irishSea`` - Use bespoke Irish sea gravity anomaly map.

Geoid Settings
..............
The **Geoid** section controls all the selection of geoid model for the ground truth and simulation estimation,

.. list-table:: Key Geoid Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``trueGeoidModel``
     - The geoid model to apply when calculating the ground truth.
     - See Below
   * - ``estimatedGeoidModel``
     - The geoid model to apply during estimation simulations.
     - See Below

.. note:: Toolbox supports the following geoid model options:

    * ``none`` - Do not use a geoid model
    * ``egm84-30`` or ``egm84-15`` - For the EGM 1984 model (at 30 or 15 minute resolutions).
    * ``egm96-15`` or ``egm96-5`` - For the EGM 1996 model (at 15 or 5 minute resolutions).
    * ``egm2008-5``, ``egm2008-2_5`` or ``egm2008-1`` - For the EGM 2008 model (at 5, 2.5 or 1 minute resolutions).


Measurement Settings
....................
The **Measurement** section contains mostly hardware configurations for the sensors to use. This includes stating
measurement frequencies and related errors.

.. list-table:: Key Measurement Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``imuMeasurementFreq``
     - The virtual hardware measurement frequency in Hertz.
     - Float
   * - ``altimeterMeasurementFreq``
     - The altimeter measurement frequency in Hertz (if enabled).
     - Float
   * - ``gpsMeasurementFreq``
     - The GPS measurement frequency in Hertz (if enabled).
     - Float
   * - ``accelerometerConfig``
     - The accelerometer profile to use for measurements.
     - Filepath
   * - ``gyroscopeConfig``
     - The accelerometer profile to use for measurements.
     - Filepath
   * - ``altimeterConfig``
     - The altimeter profile to use for measurements.
     - Filepath
   * - ``clockConfig``
     - The clock profile to use for measurements.
     - Filepath


Estimation Settings
...................
The **Estimation** section contains values relating to estimation methods that will be used for predicting the
vehicle's current position, orientation, velocity and other factors.

.. list-table:: Key Estimation Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``integrationMethod``
     - The navigation integration method to use.
     - Integration ID
   * - ``accelerometerConfig``
     - The accelerometer profile to use for estimating errors.
     - Filepath
   * - ``gyroscopeConfig``
     - The gyroscope profile to use for estimating errors.
     - Filepath
   * - ``altimeterConfig``
     - The altimeter profile to use for estimating errors.
     - Filepath
   * - ``clockConfig``
     - The clock profile to use for estimating errors.
     - Filepath
   * - ``gpsFusionMethod``
     - The GPS fusion method to use for processing GPS data.
     - GPS Method ID
   * - ``useVirtualMemory``
     - Should disk space be used to hold estimation results?
     - Yes *or* No

.. note:: The following INS integration methods are supported:

    Integration Methods:
        ``Numerical``, ``Kalman``, ``Runge_Kutta``, ``Adams_Bashforth``


Altimeter Settings
..................
The **Altimeter** section controls all altimeter related features. This include the altimeter frequency,
error profile and fusion method.

.. list-table:: Key Altimeter Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``altimeterEnabled``
     - Should the altimeter be used in the simulation?
     - Yes *or* No
   * - ``altimeterFusion``
     - Should the altimeter be used in the simulation?
     - See Below
   * - ``altimeterGainAmount``
     - The fixed gain amount to use for fixed-gain filtering.
     - Float
   * - ``altimeterAlphaAmount``
     - The alpha value to use for alpha-beta filtering.
     - Float
   * - ``altimeterBetaAmount``
     - The beta value to use for alpha-beta filtering.
     - Float

.. note:: The following altimeter fusion methods are supported:

    Integration Methods:
        ``fixed_gain`` or ``alpha_beta``


GPS Settings
.............
The **GPS** section controls all GPS related features. This includes if GPS is to be used in a simulation, the GPS
satellite data to use and areas where GPS signals are available.

.. list-table:: Key GPS Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``gpsEnabled``
     - Should the GPS be used in the simulation?
     - Yes *or* No
   * - ``gpsFusionMethod``
     - The GPS fusion method to use
     - See Below
   * - ``gpsTimeYear``
     - The 4-digit year to use for the satellite information.
     - Integer
   * - ``gpsTimeMonth``
     - The 2-digit month to use for the satellite information.
     - Integer
   * - ``gpsTimeYear``
     - The 2-digit day to use for the satellite information.
     - Integer
   * - ``gpsAutoDownload``
     - Can GPS information be automatically downloaded?
     - Yes *or* No


.. note:: The following GPS fusion methods are supported:

    GPS Fusion Methods:
        ``Fixed``, ``Loose``, ``Tight``


Quantum Settings
................
The **Quantum** section controls all Quantum Sensor related features. This covers both the cold-atom quantum
inertia and gravity gradient sensors and fusion methods.

.. list-table:: Key Quantum Section Variables
   :header-rows: 1
   :width: 100%
   :widths: 1 2 1

   * - Variable Name
     - Description
     - Type
   * - ``quantumImuEnabled``
     - Should the quantum IMU be used in the simulation?
     - Yes *or* No
   * - ``quantumImuSensor``
     - The type of quantum inertia sensor to use.
     - concept *or* realistic
   * - ``quantumImuFusion``
     - The fusion method to use for the quantum inertia sensor.
     - basic, particle_filter *or* misaligned
   * - ``quantumImuConfig``
     - The quantum IMU sub-configuration file.
     - Filepath
   * - ``quantumGravityEnabled``
     - Should the quantum gravity map-matcher be used?
     - Yes *or* No
   * - ``quantumGravitySensor``
     - The type of quantum gravity sensor to use.
     - dual_interferometer
   * - ``quantumGravityFusion``
     - The fusion method to use for the quantum gravity map-matching.
     - particle_filter
   * - ``quantumGravityConfig``
     - The quantum gravity map-matching sub-configuration file.
     - Filepath


Holonomic Settings
..................
The **Holonomics** section controls the optional use of holonomic constraints for preventing numerical
sensitivities impacting simulation accuracy. Essentially these are included to combat unwanted numerical rounding
errors in simulations.

.. list-table:: Key Holonomics Section Variables
   :header-rows: 1
   :width: 100%

   * - Variable Name
     - Description
     - Type
   * - ``useHolonomics``
     - Should holonomic constraints be used?
     - Yes *or* No
   * - ``correctionFreq``
     - The update rate at which corrections are applied (in Hz).
     - Float
   * - ``positionGain``
     - Holonomic gain for position.
     - Float
   * - ``positionAxis``
     - Holonomic axis percentages for position.
     - CSV (3-values)
   * - ``velocityGain``
     - Holonomic gain for velocity.
     - Float
   * - ``velocityAxis``
     - Holonomic axis percentages for position.
     - CSV (3-values)
   * - ``accelerationGain``
     - Holonomic gain for acceleration.
     - Float
   * - ``accelerationAxis``
     - Holonomic axis percentages for acceleration.
     - CSV (3-values)
   * - ``attitudeGain``
     - Holonomic gain for attitude.
     - Float
   * - ``attitudeAxis``
     - Holonomic axis percentages for attitude.
     - CSV (3-values)
   * - ``angleRateGain``
     - Holonomic gain for angle rates.
     - Float
   * - ``angleRateAxis``
     - Holonomic axis percentages for angle rates.
     - CSV (3-values)


Random Settings
...............
The **Random** section controls the generation of all random numbers within waypoint generation, object initialisation
and during simulations. Essentially this section allows random aspects such as noise, drift, sensor errors etc. to be
managed via random seeds. If the same random seed is reused, the randomness will always be the same.

.. list-table:: Key Random Section Variables
   :header-rows: 1
   :width: 100%

   * - Variable Name
     - Description
     - Type
   * - ``initialRandomSeed``
     - The random seed to use during initialisation.
     - Integer
   * - ``autoGenerateInitialSeed``
     - Should the value be autogenerated?
     - Yes *or* No
   * - ``waypointRandomSeed``
     - The random seed to use during waypoint creation.
     - Integer
   * - ``autoGenerateWaypointSeed``
     - Should the value be autogenerated?
     - Yes *or* No
   * - ``simulationRandomSeed``
     - The random seed to use during simulations.
     - Integer
   * - ``autoGenerateSimulationSeed``
     - Should the value be autogenerated?
     - Yes *or* No


Misc Settings
.............
The **Misc** section contains the remaining miscellaneous settings that do not fit into any of the above sections.
These namely control the extraction process of values from configuration files and whether any automatic
behaviour is to be displayed.

.. list-table:: Key Misc Section Variables
   :header-rows: 1
   :width: 100%

   * - Variable Name
     - Description
     - Type
   * - ``allowConfigOverloading``
     - Should sub-config values overload those in the main config?
     - Yes *or* No
   * - ``displayFallbackWarnings``
     - Should warnings about missing variables be displayed?
     - Yes *or* No

|

Sub-configuration Files
-----------------------
To reduce the complexity of the configuration file and to allow for a more modular design, some sections will
reference sub-configuration files which contain variables to use. For example, the variable ``vehicleConfig`` points
to the vehicle sub-configuration file to use, allowing pre-created profiles to be re-used. For instance,
``vehicleConfig`` can be used to load values belonging to different vehicle models. This has many advantages
including allowing the changing of multiple parameters by only changing a single value.

When a sub-configuration file is loaded, the values extracted from it will be overwritten by pre-existing values in
the main configuration file. For instance, if ``vehicleName`` is set to "**Cargo Plane**" in the main configuration
file, but is set to "**Commercial Plane**" in the vehicle sub-configuration file, "**Cargo Plane**" will be used.

.. note::
    This overwriting priority behaviour can be altered by changing the value of ``allowConfigOverloading``
    (set to ``false`` by default) located in the ``Misc`` section.

