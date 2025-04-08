
Fusion Methods
==============
To utilise measurements acquired from the sensors described in the previous section, a collection of fusion methods
have been implemented. These are grouped and categorised as before.


Classic Inertia
---------------
Numerical integration methods for the INS include: Simple Numerical, 4th Order Runge-Kutta, Adams Bashforth and
Kalman Filter Integration. For certain cases, Kalman Filter Integration will be required to leverage its covariance
matrices. For example, some GPS and Quantum Sensor fusion methods will require access to these.

#. :ref:`Simple Numerical <ins.py>` (``NumericalINS``)
#. :ref:`Runge Kutta <ins.py>` (``RungeKuttaINS``)
#. :ref:`Adams Bashforth <ins.py>` (``AdamsBashforthINS``)
#. :ref:`Kalman Filter <ins.py>` (``KalmanINS``)


Altitude
--------
For processing altimeter measurements two fusion methods have been implemented. The first is simple fixed-gain, which
updates the current estimated altitude by using a weighted average between the measurement and current state. The
second is alpha-beta fusion, which extends on the previous by also correcting the vertical velocity in a similar way.

#. :ref:`Fixed-Gained Fusion <altimeter.py>` (``FixedGainAltimeter``)
#. :ref:`Alpha-Beta Fusion <altimeter.py>` (``AlphaBetaAltimeter``)


GPS/GNSS
--------
For processing GPS measurements three fusion methods are available. Firstly, fixed gain simple updates the estimated
position and velocity by using a fixed weighted gain for the measurements produced by the satellites. Secondly, Loosely
coupled fusion improves on this by using a Kalman filter to integrate INS parameters with the received GPS position
and velocity measurements. Lastly, Tightly coupled fusion further extends this by operating on the raw GPS measurements
from the receiver, such as the satellite pseudo-ranges and Doppler observables.

#. :ref:`GPS Fixed Gain <fusion.py>` (``GpsFixedGainFusion``)
#. :ref:`GPS Loosely Coupled <fusion.py>` (``GpsLooseFusion``)
#. :ref:`GPS Tightly Coupled <fusion.py>` (``GpsTightFusion``)


Quantum Inertia
---------------
For processing cold-atom quantum inertia measurements three fusion methods are implemented. The first method is
called Concept Fusion, which mostly serves as a base class. From the collected measurements this simple calculates
the believed sensor biases and corrects the estimated state. The second method is a particle filter method, which
applies a particle filter approach for estimating the acceleration and angle rate biases independently. The third
method extends on this further by also accounting for any sensor misalignment.

#. :ref:`Concept Quantum Fusion <base.py>` (``ConceptQuantumFusion``)
#. :ref:`Particle Filter Quantum Fusion <realistic.py>` (``QuantumPFFusion``)
#. :ref:`Particle Filter and Misaligned Quantum Fusion <realistic.py>` (``QuantumPFMFusion``)


Quantum Gravity
---------------
Currently, there is only one fusion method for processing the gravity gradient sensor, by performing gravity
gradient map matching. This employs a particle filter approach, exploring a search space for best fitting
position, velocity and attitude candidates.

#. :ref:`Quantum Gravity Gradient Fusion <gravity_gradient.py>` (``GravityGradientPF``)
