.. _installation:

Installation
============

`pyRTX` requires a Conda environment to manage its dependencies, particularly the C++ ray tracing library, Embree.


Step 1: Install Dependencies with Conda
---------------------------------------

Set up a Conda environment and install the required dependencies from the `conda-forge` channel.
This step is needed to ensure that the most complex dependencies are properly managed.

.. code-block:: bash

    conda create --name pyRTX-env -c conda-forge python=3.8  embree3 python-embree basemap
    conda activate pyRTX-env

NOTE: The above steps are compatible with a Linux installation. If you are using a macOS system, you should check if your system is using an Intel or ARM (Apple Silicon) processor. If Intel, you can follow the same steps as for Linux. Otherwise, you must ensure that your conda environment enforces the Intel architecture to remain compatible with python-embree:

Create the environment for Intel architecture:

.. code-block:: bash

    CONDA_SUBDIR=osx-64 conda create --name pyRTX-env -c conda-forge python=3.8

Activate and lock the architecture for this environment:

.. code-block:: bash

    conda activate pyRTX-env
    conda config --env --set subdir osx-64

Install the specific packages:

.. code-block:: bash

    conda install -c conda-forge embree3 python-embree basemap

Step 2: Install pyRTX
---------------------

Once the main dependencies are installed via Conda, you can install the `pyRTX` package from this repository using `pip`:

.. code-block:: bash

    pip install -r requirements.txt .

After completing these steps, the `pyRTX` library will be fully installed and ready to use.


Quickstart and installation testing
-----------------------------------
Download the data required for running the examples running in the `examples` folder:

.. code-block:: bash

    python download_lro_kernels.py

Once the test data (SPICE kernels) is downloaded you can test the installation.
From the ``tests`` folder run

.. code-block:: bash

    pytest
