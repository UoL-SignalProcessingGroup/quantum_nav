"""
Provides objects for simulation measurement hardware.
This module is responsible for providing classes that represent typical
hardware instruments that produce (noisy) measurements. The quality of each of
these can be fine-tuned by the user. As of version 1.0 of the toolbox this
module contains the following virtual hardware classes:
* Accelerometers
* Altimeters
* Clocks
* Gyroscopes

External hardware (such as GPS/GNSS satellites) are contained within
other modules.
"""