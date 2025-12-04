.. _installation:

Installation
============

`pyRTX` requires a two-step installation process to handle both its Python and C++ dependencies. The use of a virtual environment (e.g., Conda or venv) is strongly recommended.

**Note:** The installation has been tested and is currently supported on Linux only.

Step 0: Create an environment
-----------------------------
Create an environment specific to pyRTX. 
``conda create --name=pyRTX-env python=3.8``

After the environment is created, activate it
``conda activate pyRTX-env``

Step 1: System and C++ Dependencies
-----------------------------------

Before installing the Python package, you must install the necessary C++ libraries.

1.  **Install System Prerequisites:**
    `pyRTX` requires the `GEOS` library for the `basemap` package. On Debian-based systems like Ubuntu, you can install this with:

    .. code-block:: bash

        sudo apt-get update
        sudo apt-get install libgeos-dev

2.  **Run the C++ Dependency Installer:**
    The repository includes a script to download and build the C++ ray tracing libraries (Embree). Run this script from the root of the `pyRTX` directory:

    .. code-block:: bash

        python install_deps.py

Step 2: Python Package Installation
-----------------------------------

Once the C++ dependencies are in place, you can install the `pyRTX` Python package and its dependencies.

1.  **Install `basemap`:**
    Install the `basemap` package separately using pip:

    .. code-block:: bash

        pip install basemap

2.  **Install `pyRTX`:**
    Install the `pyRTX` package and its remaining Python dependencies using pip:

    .. code-block:: bash

        pip install .

After completing these steps, the `pyRTX` library will be fully installed and ready to use.
