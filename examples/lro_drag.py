import spiceypy as sp
import pickle as pkl
import os, sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh.transformations as tmt


from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Drag import Drag
from pyRTX.core.analysis_utils import epochRange2


# Purpose
# Show the way of computing the atmospheric drag with a custom 
# density function.
# Here for the sake of simplicity we compute the Drag on LRO (although it does not make much sense)
# to leverage the already downloaded kernel

# Load the metakernel containing references to the necessary SPICE frames
METAKR = '../example_data/LRO/metakernel_lro.tm'
sp.furnsh(METAKR)

# Define the spacecraft
identity = tmt.identity_matrix()
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




# Here we will first compute the cross-section lookup table (LUT) and use it to compute the atmospheric drag

if not os.path.exists('lro_crossection.pkl'):
	print('ERROR: the cross-section lookup table must be generated first. Run generate_lro_crossection.py')
	sys.exit(0)

with open('lro_crossection.pkl', 'rb') as f:
	lut = pkl.load(f)


# The computation of the drag requires to specify a density function
# The density function must have a call sign like: dens = dens(h), where h is the height
# Here we define a dummy exponential function.
# More complex models can be defined through the classes in pyRTX.classes.Atmosphere
def density(h):
	return 1e6*np.exp(-h/100)

# Define the CD
CD = 2.2

# Define the drag object
drag = Drag(lro, lut, density, CD, 'Moon')



# Define a basic epoch and a time span
epc = "2010 may 10 09:25:00"
epc_et0 =  sp.str2et( epc )
duration = 10000 # Seconds
epc_et1 = epc_et0 + duration
tspan = epochRange2(startEpoch = epc_et0, endEpoch = epc_et1, step = 100)

accels = np.zeros(( 3,len(tspan)))

for i,ep in enumerate(tspan):
	accels[:,i], _ = drag.compute(ep, frame = 'MOON_PA')

fig, ax = plt.subplots()
for i in range(3):
	ax.plot(tspan, accels[i, :])
plt.show()



