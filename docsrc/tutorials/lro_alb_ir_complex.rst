LRO Albedo and Thermal IR (Complex)
====================================

Overview
--------

This example demonstrates advanced computation of albedo and thermal infrared 
accelerations using the object-oriented interface of the pyRTX library.

Key Features
------------

* **Spatially-varying albedo**: Uses a grid of albedo values across the planet surface
* **Spatially-varying temperature**: Incorporates temperature variations with location
* **Digital elevation model**: Represents planetary topography for accurate shadowing
* **OBJ file format**: Loads detailed shape models from standard mesh files

This advanced approach is essential for high-precision orbit determination and 
propagation where planetary radiation pressure effects need to be accurately modeled.

When to Use This Approach
-------------------------

Use this complex method when:

* High-fidelity force modeling is required
* Surface property data is available
* Topographic effects are significant
* Comparing with the simple uniform model shows non-negligible differences

Code
----

.. literalinclude:: lro_alb_ir_complex.py
   :language: python
   :linenos:
   :caption: lro_alb_ir_complex.py
