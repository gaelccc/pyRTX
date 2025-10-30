pyRTX Documentation
===================

**pyRTX** is a Python library for non-gravitational forces modelling for deep space probes using ray tracing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   tutorials/index
   api/index
   examples

Introduction
------------

pyRTX provides high-performance ray tracing capabilities for modeling:

* Solar radiation pressure
* Thermal radiation
* Surface reflections (specular and diffuse)
* Multi-bounce light transport

The library uses Intel Embree for fast ray-triangle intersections and supports
complex spacecraft geometries.

Features
--------

* **Fast Ray Tracing**: Intel Embree backend for performance
* **Multi-Bounce**: Support for multiple reflection bounces
* **Diffuse Scattering**: Lambert cosine distribution for rough surfaces
* **Solar Modeling**: Eclipse calculations and solar flux variations
* **Flexible Geometry**: Support for complex 3D mesh models

Quick Example
-------------

.. code-block:: python

   from pyRTX.classes import SunShadow
   import spiceypy as sp
   
   # Create shadow calculator
   shadow = SunShadow(
       spacecraft=my_spacecraft,
       body='Moon',
       bodyRadius=1737.4
   )
   
   # Compute eclipse ratio
   epoch = sp.str2et('2024-01-01T12:00:00')
   flux_ratio = shadow.compute(epoch)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
