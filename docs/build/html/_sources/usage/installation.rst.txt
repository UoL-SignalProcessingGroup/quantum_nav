
Installation Guide
==================

The software is in the format of a conventional Python library and can be installed directly from the command-line.
Doing this allows the PNT toolbox's modules to be used in any Python project (sharing the same environment).

.. note:: If you intend to use the toolbox as a standalone application, interacting with it purely through
    user configuration files, then the installation steps can be skipped. However, the
    :ref:`required dependencies <Software Requirements>` must be installed.

Installing the library is optional and only recommended if you wish to employ its features within Python.

|

Installing via PIP
------------------

To install the PNT library via `PIP <https://docs.python.org/3/installing/>`_ (Python's Preferred Installer Program)
firstly ensure that PIP is installed. If not, it can be invoked through python using ``python -m pip``. Next navigate
to the **project's root directory** (the main project folder containing the files ``README.md``,
``requirements.txt`` and ``setup.py``). Then execute the following command:

.. code-block:: sh
    :caption: Install the Toolbox via PIP

    pip install .

Once executed the ``setup.py`` script will be used and the toolbox will now be installed under the
package name ``nav_toolbox``.

.. important:: It is recommended that you use a `virtual environment <https://docs.python.org/3/library/venv.html>`_
    for Python so that the interpreter, libraries and scripts installed are isolated. Doing this has many benefits
    including avoiding issues with conflicting packages and removing the need to use administrator privileges.

|

Uninstalling via PIP
--------------------

Conversely, the toolbox can be uninstalled using simular steps. To remove the PNT library via PIP the
following command can be used:

.. code-block:: sh
    :caption: Uninstall the Toolbox via PIP

    pip uninstall nav_toolbox

Once executed the toolbox will be uninstalled, however its dependencies will remain.

|

Handling Dependencies
---------------------

The Python packages used by the toolbox will automatically be install during the installation process. These
dependencies and their required versions are listed within ``requirements.txt``. The listed packages can be
installed manually via PIP using the following command:

.. code-block:: sh
    :caption: Install all Toolbox Dependencies via PIP

    pip install -r requirements.txt

|

Similarly to uninstalling the toolbox, these packages can be uninstalled individually. To uninstall all
dependencies use the following command:

.. code-block:: sh
    :caption: Uninstall all Toolbox Dependencies via PIP

    pip uninstall -r requirements.txt