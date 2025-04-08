"""
Generates waypoints and trajectory data for various vehicle platforms.
This module is responsible for handling waypoint generation. In essence,
this is used for generating the true simulation data that is used within
navigation experiments. In summary, this module provides the following:
* Vehicle model simulation (with movement noise and limitations).
* Default racetrack generation for rapid experimentation.
* High frequency waypoint interpolation.
* Waypoint importing from CSV file.
* Trajectory data calculations.
* Importing/Exporting of trajectory data.
"""