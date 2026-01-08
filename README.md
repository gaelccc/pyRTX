![resized](https://github.com/gaelccc/pyRTX/assets/74771467/137f6c0a-197c-4139-862c-07b7d9a3ee78)
# pyRTX v1.0

A collection of functions, classes and tools for the computation of non-gravitational acceleration on space probes leveraging ray tracing techniques.

This library is thought to help scientists and engineers working in orbit determination, 
navigation, GNC, and similar applications, by providing a framework for precise computation
of non-gravitational forces. 

For a quick glance at `pyRTX` capabilities look at the gallery of [tutorials](Notebooks/)

Main features of pyRTX currently supported:
### Spacecraft modeling
Support for basic and complex, static and moveable spacecraft shapes. 
The spacecraft shape can be directly imported from the main 3D file formats. 
Flexible definition of the thermo-optical properties of every spacecraft surface.

### Solar radiation pressure 
Precise computation of the solar radiation pressure force and acceleration on the spacecraft.
Automatic computation of self-shadowing, secondary reflections and diffusive effects. 
Eclipse times computation using user-defined planet shapes and solar limb darkening.  
### Planetary radiation pressure
Albedo and thermal infrared pressure computations based on user-defined planetary properties. 
Easy implementation of planetary characteristics maps (e.g.,n albedo and temperature). Possibility of
using planetary shapes based on digital terrain models for maximum accuracy. 
### Atmospheric drag
Precise computation of effective area. User defined density models. Plug-in structure allowing to use complex
density models (e.g., VenusGRAM, MCD, etc.). 
### Lookup tables generation and handling
Handful classes for computing, storing and reading lookup tables for improved computational performance. 

# Installation

`pyRTX` requires a Conda environment to manage its dependencies, particularly the C++ ray tracing library, Embree.

### Step 0: Download/clone the code

Download the latest version of the code 

```bash
git clone git@github.com:gaelccc/pyRTX.git
```

### Step 1: Install Dependencies with Conda

Set up a Conda environment and install the required dependencies from the `conda-forge` channel. 
This step is needed to ensure that the most complex dependencies are properly managed.

```bash
conda create --name pyRTX-env -c conda-forge python=3.8  embree3 python-embree basemap
conda activate pyRTX-env
```

NOTE: The above steps are compatible with a Linux installation. If you are using a macOS system, you should check if your system is using an Intel or ARM (Apple Silicon) processor. If Intel, you can follow the same steps as for Linux. Otherwise, you must ensure that your conda environment enforces the Intel architecture to remain compatible with python-embree:

Create the environment for Intel architecture:

```bash
CONDA_SUBDIR=osx-64 conda create --name pyRTX-env -c conda-forge python=3.8
```

Activate and lock the architecture for this environment:


```bash
conda activate pyRTX-env
conda config --env --set subdir osx-64
```
Install the specific packages:

```bash
conda install -c conda-forge embree3 python-embree basemap
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
The installation instructions, quickstart guide, several tutorials and examples and the full API documentation can be found [here](https://gaelccc.github.io/pyRTX)  

# Contributing
We welcome contributions ton `pyRTX`. Feel free to open an issue on github or send us an email at gaelc@umbc.edu

# Citing
If you've used `pyRTX` in your work please reference our manuscript [TBD]
