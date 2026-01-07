# LRO SC object building

import trimesh as tm
import numpy as np
import spiceypy as sp
import matplotlib.pyplot as plt
import matplotlib
from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.visual.utils import plot_mesh






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
# Dumping the spacecraft mesh at a specific epoch (since the relative
# position of the parts depend on SPICE frames)
mesh = lro.dump(epc_et0) 

# Plot the mesh using plot_mesh
fig, ax = plot_mesh(mesh, title="LRO Spacecraft")

# Add axes visualization
origin = [0, 0, 0]
axis_len = 2.5
ax.quiver(origin[0], origin[1], origin[2], axis_len, 0, 0, color='r', arrow_length_ratio=0.1, label='X-axis')
ax.quiver(origin[0], origin[1], origin[2], 0, axis_len, 0, color='g', arrow_length_ratio=0.1, label='Y-axis')
ax.quiver(origin[0], origin[1], origin[2], 0, 0, axis_len, color='b', arrow_length_ratio=0.1, label='Z-axis')


plt.show()


