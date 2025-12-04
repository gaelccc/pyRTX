LRO Albedo and Thermal IR (Simple)
===================================

This example demonstrates how to compute albedo and thermal infrared accelerations
using the object-oriented interface of the pyRTX library.

In this simplified case, we use single uniform values for:

* Planet emissivity
* Planet albedo  
* Planet surface temperature

This provides a quick way to compute planetary radiation pressure effects without
requiring detailed surface property maps.

.. literalinclude:: lro_alb_ir_simple.py
   :language: python
   :linenos:
   :caption: lro_alb_ir_simple.py
