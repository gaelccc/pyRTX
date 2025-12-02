![resized](https://github.com/gaelccc/pyRTX/assets/74771467/137f6c0a-197c-4139-862c-07b7d9a3ee78)
# pyRTX v0.0.2

A collection of functions, classes and tools for the computation of non-gravitational acceleration on space probes leveraging ray tracing techniques.

This library is thought to help scientists and engineers working in orbit determination, 
navigation, GNC, and similar applications, by providing a framework for precise computation
of non-gravitational forces. 

Main features of pyRTX currently supported:
### Spacecraft modeling
Support for basic and complex, static and moveable spacecraft shapes. 
The spacecraft shape can be directly imported from the main 3D file formats. 
Flexible definition of the thermo-optical properties of every spacecraft surface.
([Example 1](Notebooks/lro_visualization.ipynb), [Example 2](Notebooks/full_visualization.ipynb))
### Solar radiation pressure 
Precise computation of the solar radiation pressure force and acceleration on the spacecraft.
Automatic computation of self-shadowing, secondary reflections and diffusive effects. 
Eclipse times computation using user-defined planet shapes and solar limb darkening.  ([Example 1](examples/lro_srp_complete.py))
### Planetary radiation pressure
Albedo and thermal infrared pressure computations based on user-defined planetary properties. 
Easy implementation of planetary characteristics maps (e.g.,n albedo and temperature). Possibility of
using planetary shapes based on digital terrain models for maximum accuracy. ([Example 1](examples/lro_planetary_radiation.py))
### Atmospheric drag
Precise computation of effective area. User defined density models. Plug-in structure allowing to use complex
density models (e.g., VenusGRAM, MCD, etc.). [(Example 1)](examples/lro_drag.py)
### Lookup tables generation and handling
Handful classes for computing, storing and reading lookup tables for improved computational performance. ([Example 1](examples/generate_lro_accel_lookup.py),[Example 2](examples/generate_crossection_lut.py))

# Installation

`pyRTX` requires a two-step installation process to handle both its Python and C++ dependencies. The use of a virtual environment (e.g., Conda or venv) is strongly recommended.

**Note:** The installation has been tested and is currently supported on Linux only.

### Step 1: System and C++ Dependencies

Before installing the Python package, you must install the necessary C++ libraries.

1.  **Install System Prerequisites:**
    `pyRTX` requires the `GEOS` library for the `basemap` package. On Debian-based systems like Ubuntu, you can install this with:
    ```bash
    sudo apt-get update
    sudo apt-get install libgeos-dev
    ```

2.  **Run the C++ Dependency Installer:**
    The repository includes a script to download and build the C++ ray tracing libraries (Embree). Run this script from the root of the `pyRTX` directory:
    ```bash
    python install_deps.py
    ```

### Step 2: Python Package Installation

Once the C++ dependencies are in place, you can install the `pyRTX` Python package and its dependencies.

1.  **Install `basemap`:**
    Install the `basemap` package separately using pip:
    ```bash
    pip install basemap
    ```

2.  **Install `pyRTX`:**
    Install the `pyRTX` package and its remaining Python dependencies using pip:
    ```bash
    pip install .
    ```

After completing these steps, the `pyRTX` library will be fully installed and ready to use.


# Quickstart and installation testing
Download the data required for running the examples running in the `examples` folder:

`python download_lro_kernels.py` 



# [Documentation](https://gaelccc.github.io/pyRTX)
The API documentation can be found [here](https://gaelccc.github.io/pyRTX)  
The user is strongly advised to look at the files contained in the `examples` folder and at the Notebooks contained in the `Notebooks` folder


# Change log
Version 0.0.2 implements the same functionalities as v0.0.1 but the code structure has been heavily restructured. Backwards compatibility is guaranteed for functions and classes call signs but not for imports syntax.
