Installation
============

Requirements
------------

* Python >= 3.8
* NumPy
* SciPy
* Trimesh
* Intel Embree (for ray tracing)

From Source
-----------

.. code-block:: bash

   git clone https://github.com/gaelccc/pyRTX.git
   cd pyRTX
   python install_deps.py
   pip install -e .

The ``install_deps.py`` script will:

1. Install Python build dependencies (Cython, NumPy, etc.)
2. Install main Python dependencies
3. Download and install Intel Embree
4. Build python-embree bindings

Dependencies
------------

Build Dependencies
~~~~~~~~~~~~~~~~~

* cython
* numpy
* setuptools
* wheel

Runtime Dependencies
~~~~~~~~~~~~~~~~~~~

* numba
* matplotlib
* pyembree
* trimesh
* scipy
* spiceypy
* (see requirements.txt for complete list)
