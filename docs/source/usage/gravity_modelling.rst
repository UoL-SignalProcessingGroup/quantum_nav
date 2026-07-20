
Gravity Modelling
=================
The project include a small collection of gravity models. Each is equipped with their own methods for calculating
various values, such as the vector acceleration, vertical gravity gradient and expected anomaly or disturbance values
for given positions.

Gravity models are grouped into two sets, being functions and correction maps. Gravity functions serve as a simple
and quick approximation for the gravity at a position. These can be applied for any location and are used as base
values. Extending from these are the gravity correction maps, which as suggested apply a correction on top of a
selected base gravity function. These provide correction records, but can only be applied in areas they cover. If
data for the requested position is unavailable, no correction to the value produced by the base function will be
applied.

.. note::
    For simple usage, only a basic gravity function will be needed. However, for applying realistic gravity behaviour,
    which is required for some forms of sensing, a high resolution correction map will also be needed.


Gravity Functions
-----------------

Fixed Constant Value
....................
Essentially not a function, serving a fixed value of 9.81 used for vertical acceleration at any position.
This is primarily used for debugging.

Simple Uniform Sphere
.....................
Approximates the Earth as uniform sphere with height extrapolation. A slight improvement over the Fixed Constant model,
but is again primarily used for debugging.

Somigliana Formula
..................
The standardised “Normal” gravity function with height extrapolation (named after Carlo Somigliana (1860–1955). Also
known as the International Gravity Formula 1980. This is far more accurate than the previous models, and also prooduces
east component values in calculated gravity vectors.

NIMA’s WGS-84 Gravity
......................
Calculations provided by National Imagery and Mapping Agency (NIMA) in "Department of Defense World Geodetic System
1984, Its Definition and Relationship with Local Geodetic Systems.". This shares many similarities with the Somigliana
model. This was namely included due to it being used in software such as MATLAB as standard.


Gravity Correction Maps
-----------------------

EGM Models
..........
Application of 7 Earth Gravitation Models defining geoid undulations in raster form. These essentially provide the
geoid height correction on top of the WGS-84 reference ellipsoid. Supported EGM models include:

* EGM-1984 (in 30 and 15 arc-minute resolutions)
* EGM-1996 (in 15 and 5 arc-minute resolutions)
* EGM-2008 (in 5, 2.5 and 1 arc-minute resolutions)

.. note::
    While EGM-2020 has been technically released, it is currently unavailable for public access.

Geosat-44
.........
the Geosat-44 gravity model provided data collected in 1999 from satellite altimeter measurements, which contain
precise geoid and gravity anomaly profiles constructed from the average of 44 repeat cycles. In total, this dataset
contains 987,755 ascending records and 991,313 descending records between the latitude bounds +/- 72 degrees
north/south. This model is represented as scatter points, which are triangulated and used in interpolation.

Marine Gravity Map
..................
The marine gravity map provides gravity anomaly values for acquired from satellite altimeter data in 2013. It is based
on previous data from CryoSat-2 and Jason-1 satellite projects. The resulting model is a 9,600 x 21,600 raster grid,
between the latitude bounds +/- 80 degrees north, with interpolation performed for land space.

GGM Plus
........
The Global Gravity Map Plus (GGM Plus) model is the result of a joint research initiative by Curtin University and
TU Munich University in 2013. It provides records of gravitational acceleration and distributable at a 7.2 arc-second
(approximately 200 metres) spatial resolution, for all land and near-coastal areas. The resulting model consists of
3,062,677,383 points, partitioned into 5 x 5 degree tiles, covering latitude bounds +/- 60 degrees north/south.


SRTM2Gravity
............
The SRTM2Gravity model extends from GGM Plus by performing the conversion of Shuttle Radar Topography Mission-based
digital elevation data to implied gravity effects. It provides records for full-scale and residual gravity at 3
arc-second (approximately 90 metres) spatial resolution, for all land and near-coastal areas. The resulting model is
partitioned into 1 x 1 degree tiles, covering latitude bounds +/- 60 degrees north/south.

Irish Sea Model
...............
A bespoke marine gravity anomaly map covering an area a north western region Irish sea. Support for this model
was included by request.

Custom Gravity Maps
...................
User-supplied rectilinear gravity maps can be loaded from CSV or MATLAB
files. A separate INI file defines the map's coordinate reference system,
grid layout, vector frame, units, source fields, interpolation method, and
out-of-coverage behaviour. QNav transforms WGS-84 trajectory positions into
the map CRS before interpolation.

Custom maps provide full North-East-Down acceleration residuals. They may
contain those residuals directly, or QNav can derive them by subtracting a
configured reference vector field from a total vector field. Before
subtraction, QNav converts geodetic, geocentric-local, ECEF, ENU, or constant
custom axes to WGS-84 geodetic NED. It also normalises gravitational
attraction by adding centrifugal acceleration when the source does not
already contain it. Residuals are then added to the selected base gravity
function.

Custom maps are two-dimensional surface corrections. The base function
retains responsibility for altitude dependence, consistent with
SRTM2Gravity residual handling. Configured height and vertical-datum
information is used to define the local vector frame and centrifugal
acceleration at each map node; it does not create a three-dimensional
interpolation volume.
