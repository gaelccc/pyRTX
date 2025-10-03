### ------------------------------------------------------------------------------------------------------- ###

# Example purpose:
# Show the object-oriented interface of the pyRTX library
#
# Example case:
# Show how to compute albedo and thermal-ir accelerations
# Here we use an albedo and temperature grid for the planet. We also represent the shape of the planet
# with a digital elevation model stored in a .obj file.

### ------------------------------------------------------------------------------------------------------- ###
### IMPORTS

import numpy as np
import spiceypy as sp
import xarray as xr
import matplotlib.pyplot as plt
import timeit

from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Planet import Planet, TemperatureGrid, AlbedoGrid
from pyRTX.classes.Radiation import Albedo, Emissivity
from pyRTX.classes.Precompute import Precompute
from pyRTX.classes.LookUpTable import LookUpTable
from pyRTX.core.analysis_utils import epochRange2

from numpy import floor, mod

### ------------------------------------------------------------------------------------------------------- ###
### INPUTS

ref_epc		= "2010 may 10 09:25:00"
duration    = 10000  									  # seconds
timestep    = 100
METAKR      = '../example_data/LRO/metakernel_lro.tm'     # metakernel
obj_path    = '../example_data/LRO/'				      # folder with .obj files
lutfile     = 'luts/lro_accel_lut.nc'					  # lookup table file
base_flux   =  1361.5
ref_radius  =  1737.4
n_cores     =  10

# The spacecraft mass can be a float, int or a xarray with times and values [kg]
sc_mass = xr.open_dataset('mass/lro_mass.nc')
sc_mass.load()
sc_mass.close()

# Planet representation (set None for representing the planet as a simple sphere.
# Load an obj for representing it with digital elevation models)
fromFile = 'moon_obj/fib_1e4_v2.obj'	

# Albedo grids
alb_grid = 'grids/bond_albedo.npy'  
alb_lon  = 'grids/ldam_4_lon.npy'
alb_lat  = 'grids/ldam_4_lat.npy'

# Temperature grids
temp_grid = 'grids/temp.npy'  
temp_lon  = 'grids/temp_lon.npy'
temp_lat  = 'grids/temp_lat.npy'

### ------------------------------------------------------------------------------------------------------- ###
### OBJECTS DEFINITION

# Time initialization
tic = timeit.default_timer()

# Load the metakernel containing references to the necessary SPICE frames
sp.furnsh(METAKR)

# Define epochs
epc_et0  = sp.str2et( ref_epc )
epc_et1  = epc_et0 + duration
epochs   = epochRange2(startEpoch = epc_et0, endEpoch = epc_et1, step = timestep)
 
# Define the Spacecraft Object (Refer to the class documentation for further details)
lro = Spacecraft( name = 'LRO',
                 
				  base_frame = 'LRO_SC_BUS', 					     # Name of the spacecraft body-fixed frame
      
                  mass = sc_mass,
      
				  spacecraft_model = {						         # Define a spacecraft model
                          
					'LRO_BUS': { 
							 'file' : obj_path + 'bus_rotated.obj',	 # .obj file of the spacecraft component
							 'frame_type': 'Spice',				     # type of frame (can be 'Spice' or 'UD'
							 'frame_name': 'LRO_SC_BUS',			 # Name of the frame
							 'center': [0.0,0.0,0.0],			     # Origin of the component
							 'diffuse': 0.1,				         # Diffuse reflect. coefficient
							 'specular': 0.3,				         # Specular reflect. coefficient
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
moon = Planet(  fromFile = fromFile,
                radius = ref_radius,
                name = 'Moon',
                bodyFrame = 'MOON_PA',
                sunFixedFrame = 'GSE_MOON',
                units = 'km',
                subdivs = 5,
                )

# Define the Albedo grid object
Lon = np.load(alb_lon)
Lat = np.load(alb_lat)

ALB = AlbedoGrid(
		radius      = ref_radius,
		frame       = 'MOON_PA',
		planet_name = 'Moon',
		from_array  = alb_grid,
		axes        = (Lon, Lat),
		)

# Define the temperature grid object
Lon = np.load(temp_lon)
Lat = np.load(temp_lat)

TEMP = TemperatureGrid(
		radius      = ref_radius,
		frame       = 'GSE_MOON',
		planet_name = 'Moon',
		from_array  = temp_grid,
		axes        = (Lon, Lat),
		)

# Set thermal properties
moon.emissivity          = 0.9
moon.albedo              = ALB
moon.gridded_temperature = TEMP

# Load the Look up table
LUT  = LookUpTable(lutfile)

# Precomputation object
prec = Precompute(epochs = epochs,)
prec.precomputePlanetaryRadiation(lro, moon, LUT.moving_frames, correction='CN')
prec.dump()

# Create the albedo object
albedo = Albedo(lro, LUT, moon, precomputation  = prec, baseflux  = base_flux,)

# Create the thermal infrared object
thermal_ir = Emissivity(lro, LUT, moon, precomputation  = prec, baseflux  = base_flux,)

### ------------------------------------------------------------------------------------------------------- ###
### COMPUTATIONS

# Both the albedo and emissivity objects have a .compute() method.
# This method returns the normalized fluxes, direction and albedo values
# for each of the planet faces contributing to the computation
# The general formula for computing the acceleration of an elementary face is:
#
# acc_i = L * albedo_value/mass * norm_flux
#
# where L is the normalized optical response of the spacecraft which can be computed
# with raytracing setting a unitary radiance of the impacting rays
# Due to the very high number of rays involved in these computations
# it is mandatory to precompute L in the form of a lookup table.
# Here we import the lookup table for LRO which can be computed using the example script
# 'compute_lut.py'

# Parallel computations
alb_accel = albedo.compute(epochs, n_cores = n_cores)[0] * 1e3 
ir_accel  = thermal_ir.compute(epochs, n_cores = n_cores)[0] * 1e3 

# Always unload the SPICE kernels
sp.unload(METAKR)

### ... Elapsed time
toc = timeit.default_timer()
time_min = int(floor((toc-tic)/60))
time_sec = int(mod((toc-tic), 60))
print("")
print("\t Elapsed time: %d min, %d sec" %(time_min, time_sec))
print("")

### ------------------------------------------------------------------------------------------------------- ###
### PLOT

epochs  = [float( epc - epc_et0 )/3600 for epc in epochs]

# ALBEDO 
fig, ax = plt.subplots(3, 1, figsize=(14,8), sharex = True)

ax[0].plot(epochs, alb_accel[:,0], linewidth = 2, color = "tab:blue")
ax[0].set_ylabel('X [m/s^2]')
ax[1].plot(epochs, alb_accel[:,1], linewidth = 2, color = "tab:blue")
ax[1].set_ylabel('Y [m/s^2]')
ax[2].plot(epochs, alb_accel[:,2], linewidth = 2, color = "tab:blue")
ax[2].set_ylabel('Z [m/s^2]')
ax[2].set_xlabel('Hours from CA')
fig.suptitle('Albedo in S/C body frame')
plt.tight_layout()

# IR
fig, ax = plt.subplots(3, 1, figsize=(14,8), sharex = True)

ax[0].plot(epochs, ir_accel[:,0], linewidth = 2, color = "tab:blue")
ax[0].set_ylabel('X [m/s^2]')
ax[1].plot(epochs, ir_accel[:,1], linewidth = 2, color = "tab:blue")
ax[1].set_ylabel('Y [m/s^2]')
ax[2].plot(epochs, ir_accel[:,2], linewidth = 2, color = "tab:blue")
ax[2].set_ylabel('Z [m/s^2]')
ax[2].set_xlabel('Hours from CA')
fig.suptitle('IR in S/C body frame')
plt.tight_layout()

plt.show()

### ------------------------------------------------------------------------------------------------------- ###