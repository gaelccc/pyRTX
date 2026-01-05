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

`pyRTX` requires a Conda environment to manage its dependencies, particularly the C++ ray tracing library, Embree.

**Note:** The installation has been tested and is currently supported on Linux only.

### Step 1: Install Dependencies with Conda

Set up a Conda environment and install the required dependencies from the `conda-forge` channel. This is the simplest way to install `embree3`, its Python wrapper `python-embree`, and other libraries like `basemap`.

```bash
conda create --name pyRTX-env
conda activate pyRTX-env
conda install -c conda-forge embree3 python-embree
```

### Step 2: Install pyRTX

Once the main dependencies are installed via Conda, you can install the `pyRTX` package from this repository using `pip`:

```bash
pip install -r requirements.txt .
```

After completing these steps, the `pyRTX` library will be fully installed and ready to use.


# Quickstart and installation testing
Download the data required for running the examples running in the `examples` folder:

```bash
python download_lro_kernels.py`
```

Once the test data (SPICE kernels) is downloaded you can test the installation.
From the ``tests`` folder run

```bash
pytest
```



# [Documentation](https://gaelccc.github.io/pyRTX)
The API documentation can be found [here](https://gaelccc.github.io/pyRTX)  
The user is strongly advised to look at the files contained in the `examples` folder and at the Notebooks contained in the `Notebooks` folder
