
Sensor Processing
=================
To acquire information from the ground truth and produce measurement of control noise and errors, the project presents
a handful of sensors. Each of these implement the base sensor class and feature methods for generating measurements,
updating internal states and indicating when their next measurement is due. These sensors are categorised below:


Classic Inertia
---------------
Classes representing conventional inertial based sensors are the accelerometer and gyroscope. These are implemented
as generic sensors, as they are reused internally by other sensors within the toolbox. These are responsible for
obtaining  acceleration and angle rate measurements respectively.

#. :ref:`Generic Accelerometer <accelerometer.py>` (``Accelerometer``)
#. :ref:`Generic Gyroscope <gyroscope.py>` (``Gyroscope``)


Altitude
--------
For obtaining altitude measurements, an altimeter sensor has been implemented. This simple obtains the measured
vertical position, following configured noise profiles.

#. :ref:`Generic Altimeter <altimeter.py>` (``Altimeter``)


GPS/GNSS
--------
For simulating GPS measurements two classes have been implemented. The first class is satellite, which represents a
satellite in orbit. While not specifically a sensor class, multiple satellite instances are used by the GPS sensor
class, which is responsible for producing measurements. Measurement information include clock times, pseudo range
information and estimated position and velocity data.

#. :ref:`GPS Satellite <satellite.py>` (``Satellite``)
#. :ref:`GPS Sensor <gps/sensor.py>` (``GPSSensor``)

Quantum Inertia
---------------
Two cold-atom quantum based inertia sensors have been implemented. The first one is a basic concept, simply extending
the use of the generic altimeter by simulating fairly accurate average inertia measurements over a duty cycle. The
second one improves on this by simulating and processing the movement of atoms to gain measurements.

#. :ref:`Concept Quantum IMU <base.py>` (``ConceptQuantumImu``)
#. :ref:`Realistic Quantum IMU <realistic.py>` (``QuantumIMU``)

Quantum Gravity
---------------
One cold-atom gravity gradient sensor has been implemented. This simulates a dual-interferometer, acquiring
measurement signals from the releasing of two vertically stacked clouds of atoms that share a phase. A series of
signals collected across multiple duty-cycles can be acquired to form an ellipse, used to estiamted the
vertical gravity gradient.

#. :ref:`Quantum Gravity Gradiometer <base.py>` (``GravityGradiometer``)
