
Software Components
===================
The functionality of the produced software has been logically categorised into Python sub-packages. This allows means
for grouping related elements together and organising the project into manageable components (rather than large amount
of files and resources). The modules belonging to each sub-package of the project have been outlined below.

Earth
-----
Modules relating to core calculations involving the earth:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    transformations.py <../modules/util/transformations>
    dted.py <../modules/earth/dted>
    geoid.py <../modules/earth/geoid>
    utm.py <../modules/earth/utm>

Estimation
----------
Modules related to state estimation:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    state.py <../modules/estimation/state>

Fusion
------
Modules related to standard sensor fusion methods:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    ins.py <../modules/fusion/ins>
    kalman.py <../modules/fusion/kalman>
    altimeter.py <../modules/fusion/altimeter>


GPS
---
Modules relating to GPS sensor and fusion methods:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    ephemeris.py <../modules/gps/ephemeris>
    fusion.py <../modules/gps/fusion>
    satellite.py <../modules/gps/satellite>
    sensor.py <../modules/gps/sensor>
    zones.py <../modules/gps/zones>

Gravity
-------
Modules relating to gravitational calculations:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    base.py <../modules/gravity/base>
    simple.py <../modules/gravity/simple>
    somigliana.py <../modules/gravity/somigliana>
    nima.py <../modules/gravity/nima>
    map.py <../modules/gravity/map>
    egm.py <../modules/gravity/egm>
    geosat.py <../modules/gravity/geosat>
    marine.py <../modules/gravity/marine>
    ggm_plus.py <../modules/gravity/ggm_plus>
    srtm2gravity.py <../modules/gravity/srtm2gravity>
    irish.py <../modules/gravity/irish>

Input
-----
Modules related to reading in input data:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    config.py <../modules/input/config>
    config_handler.py <../modules/input/config_handler>

Measurement
-----------
Modules related to standard sensors for measurements:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    sensor.py <../modules/measurement/sensor>
    platform.py <../modules/measurement/platform>
    error_properties.py <../modules/measurement/error_properties>
    accelerometer.py <../modules/measurement/accelerometer>
    gyroscope.py <../modules/measurement/gyroscope>
    altimeter.py <../modules/measurement/altimeter>
    clock.py <../modules/measurement/clock>

Output
------
Modules related to exporting output data:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    data.py <../modules/output/data>
    formats.py <../modules/output/formats>
    figures.py <../modules/output/figures>
    summary.py <../modules/output/summary>

Quantum
-------
Modules relating to quantum sensor and fusion methods:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    base.py <../modules/quantum/base>
    realistic.py <../modules/quantum/realistic>
    particle_filter.py <../modules/quantum/particle_filter>
    misalignment.py <../modules/quantum/misalignment>
    gravity_gradient.py <../modules/quantum/gravity_gradient>
    dummy.py <../modules/quantum/dummy>


Simulation
----------
Modules relating to simulation loop handling:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    simulation.py <../modules/simulation/simulation>
    scheduling.py <../modules/simulation/scheduling>
    holonomic.py <../modules/simulation/holonomic>
    results.py <../modules/simulation/results>
    display.py <../modules/simulation/display>


Util
----
Modules relating to commonly used utility functions:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    constants.py <../modules/util/constants>
    connections.py <../modules/util/connections>
    system.py <../modules/util/system>
    rng.py <../modules/util/rng>


Waypoints
---------
Modules related to trajectory generation:

.. toctree::
    :maxdepth: 2
    :titlesonly:

    generation.py <../modules/waypoints/generation>
    trajectory.py <../modules/waypoints/trajectory>
    vehicle.py <../modules/waypoints/vehicle>
    tracks.py <../modules/waypoints/tracks>
