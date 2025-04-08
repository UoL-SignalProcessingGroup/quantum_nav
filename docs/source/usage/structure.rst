
Project Structure
=================

This project contains a handful of files, directories and sub-directories; each with their own purpose.
This section outlines the role of all project folders and files. Each are grouped and outlined below:

Project Folders
---------------

/src/
    All source code files that the toolbox are composed of. This directory is identified as a "Python Directory"
    and houses all the developed code. This folder contains the project ``qnav`` and its series of sub-packages.

/test/
    All pytest test functions and classes. This can be executed to re-perform all automated tests used to evaluate
    the correctness and robustness of the project.

/docs/
    A collection of scripts and configuration files used for auto-generating the user documentation
    (in HTML offline webpages, LaTeX and PDF formats). This folder mostly consists of ReStructuredText (.rst)
    files which describe the information to be included within the documentation.

/config/
    A series of sub-configuration profiles that can be imported by main configuration profiles (such as
    ``default_settings.ini``). The purpose for these is to reduce the complexity of the main configuration script
    by partitioning sets of frequently used variables into sub-profiles. This folder contains sub-profiles for various
    aspects such as vehicle types, hardware models and waypoint trajectories.

/examples/
    A collection of helpful example scripts providing useful sample code that can be executed to demonstrate the usage of
    various features contained within the toolbox. These examples including: generating trajectories,
    loading and altering configuration files and using various databases.


/output/
    The default output directory produced by the toolbox. Here output results (data and report figures) are saved
    to auto-generated folders named by their time and date of creation. Check this folder after running a simulation
    to see the results produced.

|

Project Files
--------------

default_settings.ini
    The default configuration settings automatically used by the project if a configuration file of choice is not
    presented. When making custom configuration files it is recommended to create a copy of ``default_settings.ini``
    to use as a fully commented template.

README.md
    A short description about the purpose of the project and how to use it. Only minor details are covered to provide
    a quick-start guide. This file is automatically detected and previewed by some software.

pyproject.toml
    A standardised configuration file for the project, namely specifying build system requirements. Included to solve
    build-tool dependency problems and install necessary build tools in a virtual environment. Also required for
    configuring various development tools.

requirements.txt
    A list of all the required Python packages along with the minimum version number. This can be used for
    automatically installing all requirements via applications such as `PIP <https://docs.python.org/3/installing/>`_.

LICENSE
    Standard MIT license included with the project
