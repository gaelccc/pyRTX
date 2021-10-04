# LRO SC object building

import trimesh as tm
import numpy as np
import pickle as pkl
import trimesh.transformations as tmt
from pyRTX.scClass import Spacecraft
from pyRTX.pixelPlaneClass import pixelPlane
from pyRTX.rayTracerClass import rayTracer
from pyRTX.srpClass import solarPressure
import spiceypy as sp
import matplotlib.pyplot as plt
from pyRTX import utils_rt
from pyRTX.shadowClass import SunShadow
from pyRTX.genericClasses import Planet
from pyRTX import constants
import logging


# Example purpose:
# Show the object-oriented interface of the pyRTX library
#
# Example case:
# Compute the SRP acceleration for LRO spacecraft, using the SPICE trajectory and frames
#


# Load the metakernel containing references to the necessary SPICE frames
METAKR = '../example_data/LRO/metakernel_lro.tm'
sp.furnsh(METAKR)

# Define a basic epoch
epc = "2010 may 10 08:25:00"
epc_et0 =  sp.str2et( epc )


# Define spacecraft properties 
mass = 2000

# Define the Spacecraft Object (Refer to the class documentation for further details)
obj_path = '../example_data/LRO/'
lro = Spacecraft( name = 'LRO',
					base_frame = 'LRO_SC_BUS', 					# Name of the spacecraft body-fixed frame
					spacecraft_model = {						# Define a spacecraft model
					'LRO_BUS': { 
							 'file' : obj_path + 'bus_rotated.obj',		# .obj file of the spacecraft component
							 'frame_type': 'Spice',				# type of frame (can be 'Spice' or 'UD'
							 'frame_name': 'LRO_SC_BUS',			# Name of the frame
							 'center': [0.0,0.0,0.0],			# Origin of the component
							 'diffuse': 0.1,				# Diffuse reflect. coefficient
							 'specular': 0.3,				# Specular reflect. coefficient
							 },

					'LRO_SA': {	
							'file': obj_path + 'SA_recentred.obj',
							'frame_type': 'Spice',
							'frame_name': 'LRO_SA',
							'center': [-1,-1.1, -0.1],
							 'diffuse': 0,
							 'specular': 0.3,
							},


					'LRO_HGA': { 	
							'file': obj_path + 'HGA_recentred.obj',
							'frame_type': 'Spice',
							'frame_name': 'LRO_HGA',
							'center':[-0.99,    -0.3,  -3.1],
							 'diffuse': 0.2,
							 'specular': 0.1,
					},


					}
					
					)



# Define the Moon object
moon = Planet( 
                fromFile = None,
                radius = 1737.4,
                name = 'Moon',
                bodyFrame = 'MOON_PA',
                #bodyFrame = 'GSE_MOON',
                sunFixedFrame = 'GSE_MOON',
                units = 'km',
                subdivs = 5,
                )


# Define the Sun rays object

rays = pixelPlane( 
			spacecraft = lro,   # Spacecraft object 
			mode = 'Dynamic',   # Mode: can be 'Dynamic' ( The sun orientation is computed from the kernels), or 'Fixed'
			distance = 100,	    # Distance of the ray origin from the spacecraft
			source = 'Sun',     # Source body (used to compute the orientation of the rays wrt. spacecraft)
			width = 10,	    # Width of the pixel plane
			height = 10,        # Height of the pixel plane
			ray_spacing = 0.01, # Ray spacing (in m)
		)



# Define the ray tracer
rtx = rayTracer(        
                        lro,                    # Spacecraft object
                        rays,                   # pixelPlane object
                        kernel = 'Embree',      # The RTX kernel to use
                        bounces = 2,            # The number of bounces to account for
                        diffusion = False,       # Account for secondary diffusion
                        ) 

## Define the solar pressure

# Define the shadow function object
shadow = SunShadow( 
                        spacecraft = lro,
                        body = 'Moon',
                        bodyShape = moon,
                        limbDarkening = 'Eddington',
                        )


# Define the solar pressure object
srp = solarPressure(
                        lro, 
                        rtx,
                        baseflux = 1361,  # Here we use the None option to obtain the generalized geometry vector, used also for the computation of albedo and thermal infrared
                        shadowObj = shadow,
                        )



# Compute the SRP acceleration in different epochs and plot it
increment = 100
steps = 100
accel = np.zeros((steps, 3))
mass = 2000
epochs = [i*increment for i in range(steps)]



# Managing Error messages from trimesh
# (when concatenating textures, in this case, withouth .mtl definition, trimesh returns a warning that
#  would fill the stdout. Deactivate it for a clean output)
log = logging.getLogger('trimesh')
log.disabled = True

for i in range(len(epochs)):
        print(f'{i}/{len(epochs)}')
        epoch = epc_et0 + i*increment
        f_srp = np.array(srp.compute(epoch))
        accel[i,:] = f_srp/mass


log.disabled = False

# Always unload the SPICE kernels
sp.unload(METAKR)


fig, ax = plt.subplots(3,1, sharex = True)
ax[0].plot(epochs,accel[:,0])
ax[0].set_ylabel('X [m/s^2]')
ax[1].plot(epochs,accel[:,1])
ax[1].set_ylabel('Y [m/s^2]')
ax[2].plot(epochs, accel[:,2])
ax[2].set_ylabel('Z [m/s^2]')
ax[2].set_xlabel('Seconds past {}'.format(epc))


plt.show()


