![resized](https://github.com/gaelccc/pyRTX/assets/74771467/137f6c0a-197c-4139-862c-07b7d9a3ee78)

# pyRTX v0.0.2

A collection of functions, classes and tools for the computation of 
non-gravitational acceleration on space probes leveraging ray tracing techniques.

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
Installation in a new environment called "py38" with Anaconda (suggested).

1) Download pyRTX folder
2) conda create --name py38 python=3.8 --file requirements.txt --channel default --channel anaconda
3) conda activate py38
4) pip install "PATH_TO_MAIN_FOLDER" (e.g., pip install ~/username/pyRTX)

Several examples are provided in the examples folder. See the README.txt in the examples folder for a description of the various examples.

# [Documentation](https://gaelccc.github.io/pyRTX)
(work in progress)


# Change log
Version 0.0.2 implements the same functionalities as v0.0.1 but the code structure has been heavily restructured. Backwards compatibility is guaranteed for functions and classes call signs but not for imports syntax.
