.. _installation:

Installation
============

`pyRTX` requires a Conda environment to manage its dependencies, particularly the C++ ray tracing library, Embree.

**Note:** The installation has been tested and is currently supported on Linux only.

Step 1: Install Dependencies with Conda
---------------------------------------

Set up a Conda environment and install the required dependencies from the `conda-forge` channel.
This step is needed to ensure that the most complex dependencies are properly managed.

.. code-block:: bash

    conda create --name pyRTX-env -c conda-forge python=3.8  embree3 python-embree basemap
    conda activate pyRTX-env

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
