
Example Usage
=============
This section covers and describes the examples included with the project. Examples are group into categories,
with a handful of cases being given for each.


Changing INS Grades
-------------------
The sensor grades for the INS can be configured under the ``Measurement`` section. Specifically, by changing the
path values of ``accelerometerConfig`` and ``gyroscopeConfig``, or by changing the contents of the
sub-configuration files. The examples included cover all of the typical sensors grades ranging from
theoretically perfect down to basic consumer level grade.

.. code-block:: console

   python3 run.py 'examples/example_1/perfect.ini'
   python3 run.py 'examples/example_1/marine.ini'
   python3 run.py 'examples/example_1/aviation.ini'
   python3 run.py 'examples/example_1/intermediate.ini'
   python3 run.py 'examples/example_1/tactical.ini'
   python3 run.py 'examples/example_1/consumer.ini'


Changing INS Integration
------------------------
The INS integration method can be configured under the ``Estimation`` section. Specifically, by changing the value of
``integrationMethod`` to either *numerical*, *runge_kutta*, *adams_bashforth* or *kalman*. If the Kalman Filter is
selected, the variable ``estimatedStateModel`` must also be changed to *kalman* as well. The example include
covering these four integration methods against perfect and aviation grade sensors.

.. code-block:: console

   python3 run.py 'examples/example_2/perfect_numerical.ini'
   python3 run.py 'examples/example_2/perfect_runge_kutta.ini'
   python3 run.py 'examples/example_2/perfect_adams_bashforth.ini'
   python3 run.py 'examples/example_2/perfect_kalman.ini'
   python3 run.py 'examples/example_2/aviation_numerical.ini'
   python3 run.py 'examples/example_2/aviation_runge_kutta.ini'
   python3 run.py 'examples/example_2/aviation_adams_bashforth.ini'
   python3 run.py 'examples/example_2/aviation_kalman.ini'

Holonomic Constraints
---------------------
Holonomic constrains can be enabled to combat the impact of unwanted numerical sensitivities during simulations. To do
this, the option ``useHolonomics`` under the ``Holonomics`` section should be enabled. These examples includes
repeating the previously used trajectory with perfect sensor and other very long trajectories, all of
which show varying degrees of round errors. More examples are then used to repeat these with holonomic constraints,
removing the effect.

.. code-block:: console

   python3 run.py 'examples/example_3/perfect_numerical.ini'
   python3 run.py 'examples/example_3/20deg_perfect_without_hc.ini'
   python3 run.py 'examples/example_3/20deg_perfect_with_hc.ini'
   python3 run.py 'examples/example_3/5deg_perfect_without_hc.ini'
   python3 run.py 'examples/example_3/5deg_perfect_with_hc.ini'
   python3 run.py 'examples/example_3/60deg_perfect_without_hc.ini'
   python3 run.py 'examples/example_3/60deg_perfect_with_hc.ini'
   python3 run.py 'examples/example_3/lwpl_perfect_without_hc.ini'
   python3 run.py 'examples/example_3/lwpl_perfect_with_hc.ini'


Vehicle Models
--------------
The selected vehicle model for forming the trajectory can be configured under the ``Vehicle`` section. Specifically, a
sub-configuration for the desired vehicle type can be selected using ``vehicleConfig``, or a custom one can be
generated. Each vehicle model describes the expected speeds, turning rates, acceleration profiles and drift delays
expected. These configurations also can contain additional noise, vibration and oscillation profiles. The examples
included cover various land, sea and aerial based vehicles.

.. code-block:: console

   python3 run.py 'examples/example_4/'perfect_custom.ini'
   python3 run.py 'examples/example_4/'perfect_large_plane.ini'
   python3 run.py 'examples/example_4/'perfect_large_ship.ini'
   python3 run.py 'examples/example_4/'perfect_large_van.ini'
   python3 run.py 'examples/example_4/'perfect_submarine.ini'


Configuring Trajectories
------------------------
The shape of the trajectory can be described in a handful of ways. Firstly, by setting ``waypointMode`` to *config*
under the ``Waypoints`` section, a trajectory can be defined by setting a series of latitude, longitude an altitude
points with the variables ``waypointLat``, ``waypointLon`` and ``waypointAlt``. Alternatively, the waypoint mode can
be set to *csv* to read these values from a CSV file. Lastly, by settings the waypoint to *racetrack*, an
arithmetically generated looping trajectory will be used, as configured under the ``Racetrack`` section. The examples
included cover a few cases for each of these:

.. code-block:: console

   python3 run.py 'examples/example_5/'perfect_lwpl_csv.ini'
   python3 run.py 'examples/example_5/'perfect_lwpl_csv_2.ini'
   python3 run.py 'examples/example_5/'perfect_racetrack_1.ini'
   python3 run.py 'examples/example_5/'perfect_racetrack_2.ini'
   python3 run.py 'examples/example_5/'perfect_s3k119200_qt_csv.ini'
   python3 run.py 'examples/example_5/'perfect_trajeuler_csv.ini'


Altimeter Sensors and Fusion
----------------------------
An altimeter can be enabled to provide corrections to the estimated vertical position, configured under the
``Altimeter`` section. This allows not only the selection of sensor grade, but also whether the fusion method
*fixed_gain* or *alpha_beta* should be used, selected using the ``altimeterFusion`` option. The examples covers
a few scenarios:

.. code-block:: console

   python3 run.py 'examples/example_6/'aviation_alpha_beta.ini'
   python3 run.py 'examples/example_6/'aviation_fixed_gain.ini'
   python3 run.py 'examples/example_6/'aviation_only.ini'
   python3 run.py 'examples/example_6/'perfect_alpha_beta.ini'
   python3 run.py 'examples/example_6/'perfect_fixed_gain.ini'
   python3 run.py 'examples/example_6/'perfect_only.ini'


GPS Sensors and Fusion
----------------------
The involvement of GPS can be configured under the ``GPS`` section. This allows the enabling of the sensor, stating if
GPS data can be automatically downloaded, the configuration of noise and the involvement of active and inactive zones.
These examples cover the three fusion methods *fixed_gain*, *loose* and *tight* that can be selected using
``gpsFusionMethod``.

.. code-block:: console

   python3 run.py 'examples/example_7/'aviation_gps_fixed.ini'
   python3 run.py 'examples/example_7/'aviation_gps_fixed_with_zones.ini'
   python3 run.py 'examples/example_7/'aviation_gps_loose.ini'
   python3 run.py 'examples/example_7/'aviation_gps_loose_with_zones.ini'
   python3 run.py 'examples/example_7/'aviation_gps_tight.ini'
   python3 run.py 'examples/example_7/'aviation_gps_tight_with_zones.ini'
   python3 run.py 'examples/example_7/'aviation_only.ini'
   python3 run.py 'examples/example_7/'perfect_only.ini'
   python3 run.py 'examples/example_7/'tactical_only.ini'


Inertial Quantum Sensors
------------------------
The use of cold-atom quantum sensor can be configured under the ``Quantum`` section. This covers which sensors are
to be enabled (inertia and/or gravity), their implementation, their configured error profiles and their selected
fusion methods. These examples explore the options available for ``quantumImu``:

.. code-block:: console

   python3 run.py 'examples/example_8/'aviation_only.ini'
   python3 run.py 'examples/example_8/'aviation_quantum_concept.ini'
   python3 run.py 'examples/example_8/'aviation_quantum_realistic.ini'
   python3 run.py 'examples/example_8/'perfect_only.ini'
   python3 run.py 'examples/example_8/'perfect_quantum_concept.ini'
   python3 run.py 'examples/example_8/'perfect_quantum_realistic.ini'
   python3 run.py 'examples/example_8/'tactical_only.ini'
   python3 run.py 'examples/example_8/'tactical_quantum_concept.ini'
   python3 run.py 'examples/example_8/'tactical_quantum_realistic.ini'


.. Gravity Quantum Sensors
   -----------------------
