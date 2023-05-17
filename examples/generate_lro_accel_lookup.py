import spiceypy as sp
import numpy as np
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Planet import Planet
from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.classes.RayTracer import RayTracer
from pyRTX.classes.SRP import SolarPressure
from pyRTX.classes.Radiation import Albedo, Emissivity
from pyRTX.core.analysis_utils import epochRange2
from pyRTX.core.analysis_utils import LookupTableND as LT


# Generate the normalized optical response lookup table for LRO



# Load the metakernel containing references to the necessary SPICE frames
METAKR = '../example_data/LRO/metakernel_lro.tm'
sp.furnsh(METAKR)


# Define a basic epoch and a time span
# NOTE: Here we compute the lookup table at a specific epoch (for simplicity).
# Since LRO has moving appendages, this lookup table will be valid
# only for computing accelerations when the spacecraft is in the same
# configuration (orientation of the solar arrays and HGA)
# This aspect can be tackled by generating a higher dimensional lookup table
# with additional axes reflecting the geometry drivers (e.g., gimbal angles)
epc = "2010 may 10 09:25:00"
epc_et0 =  sp.str2et( epc )


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


rays = PixelPlane(
		spacecraft = lro,
		mode = 'Fixed',
		width = 15,
		height = 15,
		ray_spacing = 0.1,
		lon = 0,
		lat = 0,
		distance = 30)



sampling = 20
conv = np.pi/180
RA = np.linspace(0,360, int(360/sampling))*conv
DEC = np.linspace(-90,90, int(360/sampling))*conv

lut = np.zeros((len(RA), len(DEC), 3))

for i, ra in tqdm(enumerate(RA), total = len(RA)):
	for j, dec in enumerate(DEC):
			rays.update_latlon(lon = ra, lat = dec)
			rtx = RayTracer( lro, rays, kernel = 'Embree', bounces = 1, diffusion = False,)
			srp = SolarPressure( lro, rtx, baseflux = None)
			f = srp.compute(epc_et0)
			lut[i,j,:] = f


# Create the LUT object
lookup = LT(axes = (RA, DEC), values = lut, )


# Interrogate the lookup table at an off-grid point
ra = 21.112
dec = 55.402
value = lookup[ra*conv, dec*conv]
print(f'The value at [{ra},{dec}] deg is {value} ')

# Save the lookup table for later use
with open('lro_lut.pkl', 'wb') as f:
	pkl.dump(lookup, f)

# Plot the lookup table
fig, ax = plt.subplots(2,2, sharex = True, sharey = True)
print(np.shape(lookup[:,:].T))
ax[0,0].contourf(RA/conv, DEC/conv, lookup[:,:].T[0,:,:])
ax[0,0].set_title('X')
ax[0,0].set_ylabel('DEC')

ax[0,1].contourf(RA/conv, DEC/conv, lookup[:,:].T[1,:,:])
ax[0,1].set_title('Y')

ax[1,0].contourf(RA/conv, DEC/conv, lookup[:,:].T[2,:,:])
ax[1,0].set_title('Z')
ax[1,0].set_xlabel('RA')
ax[1,0].set_ylabel('DEC')


ax[1,1].contourf(RA/conv, DEC/conv, np.linalg.norm(lookup[:,:].T, axis = 0))
ax[1,1].set_title('Magnitude')
ax[1,1].set_xlabel('RA')
plt.show()