Tutorials
=========

These tutorials demonstrate how to use pyRTX for various applications.

Note that in order to have most tutorial properly running you'll need to 
download some LRO-specific files. 
You can do this by running ``python download_lro_kernels.py`` in the
``examples`` folder.

Tutorial 1: First Steps with pyRTX
-----------------------------------

Learn the basics of ray tracing and solar pressure modeling.

.. toctree::
   :maxdepth: 1
   
   Notebook1

Tutorial 2: Advanced SRP Calculations
--------------------------------------

Advanced solar radiation pressure techniques.

.. toctree::
   :maxdepth: 1
   
   Notebook2

Tutorial 3: Lookup Tables
--------------------------

Using pre-computed lookup tables for fast calculations.

.. toctree::
   :maxdepth: 1
   
   Notebook3


Tutorial 4: Planetary Radiation
-------------------------------

Computing albedo and thermal infrared pressure

.. toctree::
   :maxdepth: 1
   
   Notebook4

Tutorial 5: Atmospheric Drag
-------------------------------

Computing atmospheric drag on complex spacecraft shapes. 

.. toctree::
   :maxdepth: 1
   
   Notebook5


Next Steps
___________
Where to go from here? Check the examples in the github repository. 
They show how to treat more complex cases, where the spacecraft has 
moveable appendages. 


.. toctree::
   :maxdepth: 1

   lro_mass.py
   compute_lut.py
   lro_srp.py
   lro_srp_with_lut.py
   lro_alb_ir_simple.py
   lro_alb_ir_complex.py
   
