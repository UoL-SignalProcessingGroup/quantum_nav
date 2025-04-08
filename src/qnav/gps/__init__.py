"""
All GPS functionality: Satellites, Solvers and Estimation Fusion
This module contains all GPS related functionality added to the project,
encompassing all RINEX file processing, satellite functions and
estimation fusion methods. In summary this module provides the following:
 * Downloading and extracting of RINEX data.
 * Parsing of RINEX/Ephemeris files.
 * Extracting of satellite data.
 * Processing for collections of satellites.
 * Pseudo Range calculation and estimation.
 * Fixed gain fusion with any INS.
 * Loose-coupled fusion with Kalman Filter.
 * Tightly-coupled fusion with Kalman Filter.
"""