
System Requirements
===================
The finalised toolbox requires various hardware and software requirements to be efficiently used.
The minimum and suggested requirements have been acquired based on experimentation and auditing of the toolbox.


Hardware Requirements
---------------------
The hardware requirements are subject to the requested input. The resources needed can be reduced by decreasing the
requested frequency/resolution and increasing the output down-sampling via the configuration file.


Minimum Specifications
......................
The minimum requirements to use the toolbox are outlined in the table below. The host system must at a minimum have
these hardware specifications to use the toolbox at a basic (limited) level. Updates and potential additions to the
toolbox may increase these minimum requirements.

+---------------------------+---------------------------------------------------+
|Hardware                   |Minimum Resources Required                         |
+===========================+===================================================+
|Processor (CPU)            | x86 32-bit processor 1.00 GHz (or better)         |
+---------------------------+---------------------------------------------------+
|Physical Memory (RAM)      | 2.00 GB of memory (or larger)                     |
+---------------------------+---------------------------------------------------+
|Disk Storage Space (HDD)   | 20.00 GB available (or more)                      |
+---------------------------+---------------------------------------------------+


Recommended Specifications
..........................
The recommended requirements to fully use the toolbox are outlined in the table below. If the host system has these
hardware specifications or higher it is guaranteed to use the software to its full extent.

+---------------------------+---------------------------------------------------+
|Hardware                   |Recommended Resources Required                     |
+===========================+===================================================+
|Processor (CPU)            | x86 64-bit processor 2.00 GHz (or better)         |
+---------------------------+---------------------------------------------------+
|Physical Memory (RAM)      | 8.00 GB of memory (or larger)                     |
+---------------------------+---------------------------------------------------+
|Disk Storage Space (HDD)   | 100.00 GB available (or more)                     |
+---------------------------+---------------------------------------------------+


Note About Memory and Storage
.............................

Executing simulations can demand a large amount of RAM. This is especially true if high frequencies and/or a large
amount of waypoint data is used. For example, a single simulation at 1,000 Hz is expected to require around 1 GB of RAM
for each simulated hour due to the amount of data that needs to be produced and recorded. To significantly reduce this
resource requirement various settings can be changed.

Firstly, virtual memory can be used for offloading waypoint and estimation results to disk. This is enabled under
the **Waypoint** and **Estimation** sections in the configuration file via the ``useVirtualMemory`` options. In
addition, an ``outputDownSampleRate`` can be specified under the **Output** section, meaning that only a sub-sampled
amount of the estimation results will be collected and saved. Lastly, at a worst case scenario the waypoint and
measurement frequencies can be reduced.

.. warning:: The use of virtual memory on 32-bit systems is limited to at most 2 GB. This means that waypoints and
    simulation results cannot use more than 2 GB if the use of virtual memory is enabled on 32-bit systems. If this
    limitation causes significant problems requested updates to the toolbox can be made to overcome it.

In the event that the output data needs to be saved in multiple formats, but there is insufficient storage space,
a single format (such as simple binary) can used which can later be converted. Using the read and write modules inside
the output sub-package, the single generated file can be read and its content can then be written into the other
needed formats.

|

Software Requirements
---------------------

Operating System
................
The toolbox has been developed with platform independence constantly in mind, taking fully advantage of Python's
flexibility.

The handling of file paths and managing local resources has been done without replaying on any operating system
specific features/formats. Likewise, the implementation of multi-processing functions has been done in a way that
acknowledges adapts to the different ways operating system handle communication between multiple threads/processes.
As a result, the same project build can be used on any major platform without any alteration to its source code.

The toolbox has been tested and proved to function flawlessly on Windows, Mac and Linux systems. Specifically:

* Windows 10 - Enterprise
* Linux - Ubuntu LTS 22.04
* MacOS - Sonoma 14.6.1
* Raspberry Pi OS - 2022-09-22 (32-bit)


Python
......
The toolbox was developed using the `latest stable release of Python <https://www.python.org/downloads//>`_.
To benefit from this, the latest Python features have been employed. Therefore, the toolbox requires the installation
of **Python version 3.11 or higher**. During development, the use of using future safe functions/packages was strongly
considered. While older versions of Python *may* be able to run the toolbox, the correct behaviour/output cannot
be guaranteed.


Required Packages
.................
Where possible, the use of dependencies has been eliminated by implementing the majority of the software manually.
However, this could not be achieved for instances where the required functionality is not available in base Python.
As a result, the toolbox requires the use of the following packages.

* `Mypy <https://pypi.org/project/mypu/>`_ (version 3.12.1+)
* `NumPy <https://pypi.org/project/numpy/>`_ (version 2.1.1+)
* `SciPy <https://pypi.org/project/scipy/>`_ (version1.14.1+)
* `Pandas <https://pypi.org/project/pandas/>`_ (version 2.2.3+)
* `MatplotLib <https://pypi.org/project/matplotlib/>`_ (version 3.9.2+)
* `Plotly <https://pypi.org/project/plotly/>`_ (version 5.24.1+)
* `H5py <https://pypi.org/project/h5py/>`_ (version 1.11.2+)


If `PIP <https://docs.python.org/3/installing/>`_ is installed on the host machine, all the above required packages for
their target version can be installed automatically by utilising the included ``requirements.txt`` file by using the
following command:

.. code-block:: sh
    :caption: Automatically install required packages (via PIP)

    pip install -r requirements.txt


Sometimes PIP may not be installed as a standalone application, or it may be associated with another version of
Python that is installed (i.e. an older version). In this case PIP can be use through Python.

.. code-block:: sh
    :caption: Automatically install required packages

    python -m pip install -r requirements.txt


.. note:: If the toolbox is installed as a package via PIP, all packages and dependencies will also be installed
    automatically along side the software. You can check which Python packages are installed using the
    ``pip list`` command.


For installing and handling Python packages in general, please see
`this guide <https://packaging.python.org/tutorials/installing-packages/>`_.
