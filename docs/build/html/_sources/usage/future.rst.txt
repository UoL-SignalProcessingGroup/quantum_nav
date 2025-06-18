
Potential Future Work
=====================

The produced PNT toolbox is very capable of providing facilities for conducting a wide range of navigation based
experiments. However, it can always be improved and additional features can be added anytime. For this reason, this
section lists some of the possible ways the toolbox can be extended given a potential project extension.

Possible Extensions
-------------------

More Sensors and Estimation Algorithms
    If needed additional virtual hardware classes could be added, including non-classical sensors. On top of this,
    more advanced estimation algorithms can be implemented. For example, cold-atom quantum sensors could be introduced
    to the toolbox along with new algorithms to fuse their measurements with the current estimated states.

-----------

Concurrent Simulations
    In some cases it is required than experiments are executed to cover a pre-set range of parameters. If needed, the
    toolbox could be adjusted so that multiple simulations are executed to cover a requested range of configuration
    values or cover all permutations automatically.

    Moreover, since the simulation execution cannot efficiently benefit from multithreading, the independent
    simulations could be conducted in parallel with each using a separate CPU core. In theory, this would allow the
    range of configuration settings to be covers in a fraction of the time.

-----------

Multi-Vehicle Cooperation
    The toolbox is built for simulating a single platform following a set trajectory, taking measurements and using
    them to update its estimated states. For more advanced experiments, multiple platforms may need to cooperate with
    one another and collaborate to share estimates. While the toolbox is very capable of doing this (when used as a
    Python library) there is no set module, function or configuration to conduct it immediately. Nevertheless, the
    resources to conduct such experiments could be implemented if requested.

-----------

Graphical User Interface
    If requested, a custom Graphical User Interface (GUI) could be implemented. This would aid the user with
    configuring experiments, showing produced trajectories and displaying simulation results in real-time. In
    principle, this would allow the software to be utilised much more easily as a standalone application.

    The user interface could be implemented as desktop application using a standard Python framework such as
    `Tkinter <https://docs.python.org/3/library/tkinter.html>`_. Alternatively, a web-browser based interface could be
    made, allowing the software to be interfaced with across a local network. This would be more beneficial if the
    software is to be executed on system that does not feature a display or operating system with a desktop
    environment (i.e. command-line based interface). By using a web-browser, the interfacing system will not required
    any software or resources to be installed (including Python).

-----------

Numba Optimisation
    If faster processing times are required, the toolbox can utilise `Numba <https://numba.pydata.org/>`_. This would
    employ a JIT (Just-In-Time) compiler to convert the Python code to low-level code, obtaining performance comparable
    to that produced by C-code. In addition, this would utilise hardware such as multiple CPU cores and GPU devices
    to further accelerate calculations.
