
# Python Navigation Toolbox
## Seperated Sensor Branch
### Last Updated 4th July 2025

This software is a Position, Navigation and Timing (PNT) enumeration testing 
environment design for navigation study. This toolbox supplies features for 
in-depth simulations involving multi-source fusion. It has been designed 
to be compatible with any modern  platform, allowing even low-end hardware 
to fully utilise all available  features. The software can be used as either
a standalone application or as  a well documented API for advanced
experimentation. 

In summary, the toolbox provides the following key features:
- Easy of use
  - Single command execution
  - Self describing user input 
- Realistic waypoint generation:
  - At selectable resolution
  - Featuring vehicle movement modelling
  - Trajectory data generation (with variable noise)
  - Importing/Exporting functionalities
- Configurable virtual hardware:
  - Accelerometers, gyroscopes, altimeters etc.
  - A wide selection of measurement quality settings
  - Controlled errors and randomisation 
- Selectable Integration System
  - Various INS solutions 
  - Supports class inheritance 
  - Including Kalman filter solutions
- GPS simulation 
  - Downloading of RINEX/ephemeris data
  - Simulation of satellites and pseudo ranges
  - Various fusion methods including tight-coupling
- Simulation results exporting 
  - Supports many universal out formats
  - Summary PDF report generation 


## Execution
A simulation can be simply executed from the command line (employing the
default configuration file) by executing the following command:
```bash
python run.py
```

## Configuration
On execution, the software will extract its settings and user preferences from
 a provided configuration file. A requested configuration can be state by 
 supplying its name (and location) as the first argument to the run script. 
 If a configuration file is not provided, the default configuration 
 `default_settings.ini` will be used  automatically. An example of using a 
 custom configuration file (in this case `custom_config.ini`) has been 
 generated below:
```bash
python run.py custom_config.ini
```

## Run Series
The toolbox can also execute a series of simulations simultaneously via 
 the `run_all.py` script. By presenting the path to a directory, the script
 will recursively find all configuration files and execute them in a 
 parallel pool. The number of threads to use can also be passed as the 
 second command line argument (or left blank to auto assign). For example:
```bash
python run_all.py path/to/config/dir/ 10
```


