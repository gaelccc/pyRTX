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
import matplotlib.pyplot as plt
import matplotlib

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


### Axes visualization
# This code block shows a generic way of plotting arrays in trimesh
# This can be used to represent forces, directions, etc. 
xaxis = np.array([1,0,0])
yaxis = np.array([0,1,0])
zaxis = np.array([0,0,1])
origin = np.array([0,0,0])
xaxis = tm.load_path(np.hstack(( origin, origin + xaxis*0.01)).reshape(-1, 2, 3))
yaxis = tm.load_path(np.hstack(( origin, origin + yaxis*0.01)).reshape(-1, 2, 3))
zaxis = tm.load_path(np.hstack(( origin, origin + zaxis*0.01)).reshape(-1, 2, 3))

xaxis.colors = np.full((1,4),matplotlib.colors.to_rgba_array('red')*255)
yaxis.colors = np.full((1,4),matplotlib.colors.to_rgba_array('green')*255)
zaxis.colors = np.full((1,4),matplotlib.colors.to_rgba_array('blue')*255)

# Dumping the spacecraft mesh at a specific epoch (since the relative
# position of the parts depend on SPICE frames)
mesh = lro.dump(epc_et0) 

scene = tm.Scene([mesh, xaxis, yaxis, zaxis])
scene.show()


plt.show()


