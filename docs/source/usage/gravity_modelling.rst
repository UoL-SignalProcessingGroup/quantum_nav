
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
User-supplied rectilinear gravity maps can be loaded from CSV, general
delimited text, MATLAB, GeoTIFF, and NetCDF/GMT GRD files. GeoTIFF and
NetCDF readers are optional and can be installed with
``pip install "qnav[maps]"``. A separate INI file defines the map's
coordinate reference system, grid layout, physical quantity, vector frame,
units, source fields, interpolation method, and out-of-coverage behaviour.
QNav transforms WGS-84 trajectory positions into the configured map CRS
before interpolation; a map is not assumed to be WGS-84 merely because it
also contains latitude and longitude columns.

Custom maps can provide full North-East-Down acceleration residuals or a
scalar vertical gravity disturbance/free-air anomaly. A scalar is converted
to the NED down component using its explicit ``verticalDirection``. QNav can
also derive vector residuals by subtracting a configured reference vector
field from a total vector field. Before
subtraction, QNav converts geodetic, geocentric-local, ECEF, ENU, or constant
custom axes to WGS-84 geodetic NED. It also normalises gravitational
attraction by adding centrifugal acceleration when the source does not
already contain it. Residuals are then added to the selected base gravity
function.

``gravity_disturbance`` is the preferred scalar map-matching signal because
it is directly the difference between actual and normal gravity at the same
position. ``free_air_anomaly`` is also supported, but requires a geoid model
so QNav can apply its existing anomaly/disturbance conversion. Generic
``gravity_anomaly``, Bouguer anomaly, and isostatic anomaly products are
rejected: their reductions are not interchangeable with a physical
acceleration correction without extra terrain, density, and reference-model
metadata.

Custom maps are two-dimensional surface corrections. The base function
retains responsibility for altitude dependence, consistent with
SRTM2Gravity residual handling. Configured height and vertical-datum
information is used to define the local vector frame and centrifugal
acceleration at each map node; it does not create a three-dimensional
interpolation volume.

Large maps should use a ``[Subset]`` section and retain at least one padding
cell for linear interpolation. Bounds can be expressed in the map grid or in
WGS-84 and are transformed with a densified boundary. ``maxCells`` (five
million by default) stops an accidentally global or misconfigured selection
before field arrays are allocated. GeoTIFF windows, NetCDF indexes, and a
shared delimited grid/field file are subset during reading.

Examples of compatible public products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The custom interface is intentionally provider-neutral. The following
products are useful compatibility examples but remain subject to their
providers' documentation, citation, and licence terms:

* The `ICGEM Calculation Service
  <https://icgem.gfz-potsdam.de/calcgrid>`_ can calculate regular
  ellipsoidal grids of gravity disturbance. Its ASCII output can be described
  with ``format = delimited`` and the provider's actual header/index layout;
  choose gravity disturbance rather than an ambiguously named anomaly.
* The `NCEI gravity catalogue
  <https://www.ncei.noaa.gov/products/gravity-data>`_ includes regional
  free-air grids and documented ASCII products. A complete regular grid can
  use the delimited template. Point surveys and irregular flight lines must
  first be gridded; QNav does not silently interpolate scattered data.
* The `4D Antarctica compilation
  <https://ftp.space.dtu.dk/pub/RF/4D-ANTARCTICA/4D_antarctica_gravity-grid.pdf>`_
  documents free-air/gravity-disturbance fields in ASCII and GeoTIFF and an
  underlying polar-stereographic grid. The appropriate projected CRS and the
  chosen physical field must be configured explicitly. Bouguer and isostatic
  fields from such products are deliberately not accepted as acceleration
  corrections.

The templates ``custom_delimited_free_air.ini``,
``custom_geotiff_scalar.ini``, and ``custom_netcdf_scalar.ini`` demonstrate
the supported layouts without bundling or downloading provider data.
