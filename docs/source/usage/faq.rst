
Frequently Asked Questions
==========================

This section lists all the commonly asked questions about the toolbox. Each block lists a question followed by a short
but descriptive answer.


Questions and Answers
---------------------

Can I extend the functionality of the toolbox?
    *Yes*. All Python classes implemented in the toolbox can be extended/inherited by custom children classes. These
    custom children classes should be supported by all modules in the toolbox. For example, you may choose to implement
    a custom Kalman filter than extends from the toolbox's.

_______


Can I run the toolbox on a Raspberry Pi?
    *Yes*. Many modifications have been made to the toolbox to allow it to be used efficiently on systems with limited
    resources. It is recommended that waypoints are generated a head of time on a more powerful platform, then exported
    so that they can be imported by other devices. The use of virtual memory is strongly recommended for systems with
    limited RAM.

    .. warning::
        While 32-bit system are supported, it is recommended that a 64-bit OS is used due to the amount of memory
        (virtual or real) required by the toolbox.

_______


The toolbox is using too much memory. Can I reduce this?
    *Yes*. The easiest way the reduce memory usage is to enable the ``useVirtualMemory`` options in the configuration
    file. This will use temporary disk space to offload large arrays from memory.

_______

Can the software be install in an isolated environment?
    *Yes*. The software, Python and the required packages can be installed in an isolated environment without
    Internet access. This will require the above to be downloaded on a machine with internet access of similar
    platform, then copied onto the isolated machine (i.e. through the use of write-only media).

    For example, the Python installer must be downloaded on a machine with the same operating system as the isolated
    environment. Using the same machine, the required packages (listed in ``requirements.txt``) must be downloaded to a
    local folder (not installed) via PIP. Once performed the Python installer, downloaded packages and toolbox source
    code must then be moved to the isolated environment. Once Python is installed on the isolated environment, the
    python packages can be installed locally. The toolbox can then be used on the isolated machine.

    There are many guides online for installing Python packages offline for various platforms. If you need assistance,
    please email us and we will be happy to help.

_______

Can multi-threading be used to speed up a simulation?
    *No*. Unfortunately the calculations in one iteration are always required by the next. Because of this
    multi-threading cannot be used efficiently to improve runtime. However, vectorisation has been used where possible
    to improve processing times for large arrays, namely during waypoint generation.




