Quick Start Guide
=================

Basic Usage
-----------

1. **Import the library**

.. code-block:: python

   from pyRTX.classes import SunShadow, SolarPressure
   from pyRTX.classes import Spacecraft
   import spiceypy as sp

2. **Set up your spacecraft**

.. code-block:: python

   # Load your spacecraft geometry
   spacecraft = Spacecraft(
       name='MySpacecraft',
       mesh_file='path/to/mesh.obj',
       mass=1000.0  # kg
   )

3. **Calculate solar radiation pressure**

.. code-block:: python

   # Create SRP calculator
   srp = SolarPressure(
       spacecraft=spacecraft,
       baseflux=1361.5  # W/m² at 1 AU
   )
   
   # Compute at an epoch
   epoch = sp.str2et('2024-01-01T12:00:00')
   acceleration = srp.compute(epoch)

4. **Calculate eclipses**

.. code-block:: python

   # Create shadow calculator
   shadow = SunShadow(
       spacecraft=spacecraft,
       body='Moon',
       bodyRadius=1737.4
   )
   
   # Compute shadow ratio (0=total eclipse, 1=full sun)
   shadow_ratio = shadow.compute(epoch)

Next Steps
----------

* See the :doc:`api/index` for detailed API documentation
* Check the :doc:`examples` for more complex use cases
