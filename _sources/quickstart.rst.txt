Quick Start Guide
=================

This guide provides a basic walk-through of pyRTX to compute the Solar Radiation Pressure (SRP) on a spacecraft.

1. **Load SPICE Kernels**

All calculations requiring geometry or astrodynamic data rely on the SPICE toolkit. You must provide a "metakernel" that lists all the necessary SPICE kernels (trajectory, attitude, clock, etc.).

.. code-block:: python

   import spiceypy as sp

   METAKR = 'path/to/your/metakernel.tm'
   sp.furnsh(METAKR)

2. **Define the Spacecraft**

A spacecraft is defined by its components (bus, solar arrays, antennas), each with its own 3D model, reference frame, and material properties.

.. code-block:: python

   from pyRTX.classes.Spacecraft import Spacecraft

   # Component models are stored in a dictionary
   model = {
       'BUS': {
           'file': 'path/to/bus.obj',
           'frame_type': 'Spice',
           'frame_name': 'SC_BUS_FRAME',
           'center': [0.0, 0.0, 0.0],
           'diffuse': 0.1,
           'specular': 0.3,
       },
       'SOLAR_ARRAY': {
           'file': 'path/to/solar_array.obj',
           'frame_type': 'Spice',
           'frame_name': 'SC_SA_FRAME',
           'center': [0.0, 0.0, 0.0],
           'diffuse': 0.0,
           'specular': 0.3,
       },
   }

   # Create the spacecraft object
   sc = Spacecraft(
       name='MySpacecraft',
       base_frame='SC_BUS_FRAME',
       spacecraft_model=model
   )

3. **Set Up the Ray Tracing Environment**

To calculate SRP, you need to define a source for the sun's rays (a ``PixelPlane``) and a ``RayTracer`` to track their paths.

.. code-block:: python

   from pyRTX.classes import PixelPlane, RayTracer

   # Define the sun's ray source
   rays = PixelPlane(
       spacecraft=sc,
       mode='Dynamic',
       distance=100,  # meters from spacecraft
       source='Sun',
       width=10,      # meters
       height=10,     # meters
       ray_spacing=0.01 # meters
   )

   # Configure the ray tracer
   rtx = RayTracer(
       spacecraft=sc,
       pixel_plane=rays,
       bounces=1
   )

4. **Calculate Solar Radiation Pressure**

Finally, create a ``SolarPressure`` object and use it to compute the acceleration over a desired time range.

.. code-block:: python

   from pyRTX.classes import SolarPressure
   from pyRTX.core.analysis_utils import epochRange
   import numpy as np

   # Define time settings
   start_epoch = "2024 JAN 01 12:00:00"
   duration = 3600  # seconds
   step = 60      # seconds
   epochs = epochRange(start_epoch, duration, step)

   # Create the SRP calculator
   srp = SolarPressure(
       spacecraft=sc,
       ray_tracer=rtx,
       baseflux=1361.5  # W/m^2
   )

   # Compute the acceleration [km/s^2]
   acceleration = srp.compute(epochs, n_cores=1)


Next Steps
----------

* See the :doc:`api/index` for detailed API documentation.
* Check the :doc:`tutorials/index` for more complex use cases.
