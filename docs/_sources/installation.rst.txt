.. _installation:

Installation
============

`pyRTX` requires a Conda environment to manage its dependencies, particularly the C++ ray tracing library, Embree.


Step 1: Install Dependencies with Conda
---------------------------------------

Set up a Conda environment and install the required dependencies from the `conda-forge` channel.
This step is needed to ensure that the most complex dependencies are properly managed.


**Linux and macOS (Intel)**
"""""""""""""""""""""""""""""

For Linux and Intel-based macOS systems, you can create the conda environment with the following command:

.. code-block:: bash

    conda create --name pyRTX-env -c conda-forge python=3.8  embree3 python-embree basemap
    conda activate pyRTX-env

.. note::
   The Linux installation instructions should also work for Windows Subsystem for Linux (WSL), but this has not been extensively tested.


**macOS (ARM, Apple Silicon)**
""""""""""""""""""""""""""""""

If you are using a macOS system with an ARM-based processor (Apple Silicon), you must ensure that your conda environment enforces the Intel architecture to remain compatible with `python-embree`.

Create the environment for the Intel architecture:

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


Step 3: Testing the Installation
--------------------------------

``pyRTX`` comes with a suite of tests to verify the successful installation.
First of all, download the necessary files (SPICE kernels, needed also for the examples)
by entering the ``examples`` folder and running

.. code-block:: bash

   python download_lro_kernels.py

Then enter the ``tests`` folder and run

.. code-block:: bash

   pytest
