import spiceypy as sp
import pickle as pkl
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Planet import Planet
from pyRTX.classes.Radiation import Albedo, Emissivity
from pyRTX.core.analysis_utils import epochRange2

# Example purpose:
# Show how to compute albedo and thermal-ir accelerations
#




# Load the metakernel containing references to the necessary SPICE frames
METAKR = '../example_data/LRO/metakernel_lro.tm'
sp.furnsh(METAKR)


# Define a basic epoch and a time span
epc = "2010 may 10 09:25:00"
epc_et0 =  sp.str2et( epc )
duration = 10000 # Seconds
epc_et1 = epc_et0 + duration
tspan = epochRange2(startEpoch = epc_et0, endEpoch = epc_et1, step = 100)


# Define spacecraft properties 
mass = 2000
sunFlux = 1380

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

# Set the albedo and emissivity values
# Here we use dummy values and assume that
# albedo and emissivity are constant over the whole planet
# and set a different dayside and nightside temperature
# pyRTX supports also gridded values for albedo and emissivity


moon.albedo = 0.3
moon.emissivity  = 0.8 
moon.dayside_temperature = 300
moon.nightside_temperature = 200



# Create the albedo object
albedo = Albedo(
		Planet = moon,
		spacecraftName = 'LRO',
		spacecraftFrame = 'LRO_SC_BUS'
	)

thermal_ir = Emissivity(
		Planet = moon,
		spacecraftName = 'LRO',
		spacecraftFrame = 'LRO_SC_BUS'

		)

# Both the albedo and emissivity objects have a .compute() method.
# This method returns the normalized fluxes, direction and albedo values
# for each of the planet faces contributing to the computation
# The general formula for computing the acceleration of an elementary face is:
#
# acc_i = L * albedo_value/mass * norm_flux
# where L is the normalized optical response of the spacecraft which can be computed
# with raytracing setting a unitary radiance of the impacting rays
# Due to the very high number of rays involved in these computations
# it is useful to precompute L in the form of a lookup table.
# Here we import the lookup table for LRO which can be computed using the example script
# 'generate_lro_accel_lookup.py'

if not os.path.exists('lro_lut.pkl'):
	print('ERROR: You need to generate first the lookup table. To do so: run generate_lro_accel_lookup.py')
	sys.exit(0)
else:
	with open('lro_lut.pkl', 'rb') as f:
		LUT = pkl.load(f)


ALB = np.zeros((3, len(tspan)))
EMI = np.zeros_like(ALB)
for i, ep in tqdm(enumerate(tspan), total = len(tspan)):

	norm_fluxes, dirs, alb_values = albedo.compute(ep)
	lll = LUT[dirs[:,0], dirs[:,1]]
	norm_fluxes = np.expand_dims(norm_fluxes, axis = 1)
	alb_values = np.expand_dims(alb_values, axis = 1)

	ALB[:,i] = np.sum(alb_values * sunFlux/mass * norm_fluxes * lll, axis = 0)



	norm_fluxes, dirs, emi_values = thermal_ir.compute(ep)
	lll = LUT[dirs[:,0], dirs[:,1]]
	norm_fluxes = np.expand_dims(norm_fluxes, axis = 1)
	emi_values = np.expand_dims(emi_values, axis = 1)

	EMI[:,i] = np.sum(emi_values * sunFlux/mass *norm_fluxes* lll, axis = 0)


fig, ax = plt.subplots(2,1, sharex = True)

labels = ['x','y','z']
for i in range(3):
	ax[0].plot(tspan, ALB[i,:], label = labels[i])
	ax[1].plot(tspan, EMI[i,:], label = labels[i])
ax[0].legend()
ax[0].set_ylabel('Albedo Acceleration [km/s]')
ax[1].set_ylabel('Thermal IR Acceleration [km/s]')
plt.show()




