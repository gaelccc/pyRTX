### ------------------------------------------------------------------------------------------------------- ###

# Example purpose:
# Show the object-oriented interface of the pyRTX library
#
# Example case:
# Compute the drag acceleration for LRO spacecraft, using the SPICE trajectory and frames and the crossection
# values stored in a lookup table.

### ------------------------------------------------------------------------------------------------------- ###
### IMPORTS

import spiceypy as sp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import timeit
from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Drag import Drag
from pyRTX.classes.LookUpTable import LookUpTable
from pyRTX.classes.Precompute import Precompute
from pyRTX.core.analysis_utils import epochRange2

import warnings
warnings.filterwarnings('ignore')

### ------------------------------------------------------------------------------------------------------- ###
### INPUTS

ref_epc 	=  "2010 may 10 09:25:00"
duration    =  10000  									   # seconds
timestep    =  100
METAKR      =  '../example_data/LRO/metakernel_lro.tm'     # metakernel
obj_path    =  '../example_data/LRO/'				       # folder with shape .obj files
accel_frame =  'MOON_PA'
body	    =  'Moon'
lutfile     =  'luts/lro_cross_lut.nc'					   # lookup table file
n_cores	    =  10

# The spacecraft mass can be a float, int or a xarray with times and values [kg]
sc_mass = xr.open_dataset('mass/lro_mass.nc')
sc_mass.load()
sc_mass.close()

### ------------------------------------------------------------------------------------------------------- ###
### OBJECTS DEFINITION

# Time initialization
tic = timeit.default_timer()

# Load the metakernel containing references to the necessary SPICE frames
sp.furnsh(METAKR)

# Define a basic epoch
epc_et0 =  sp.str2et( ref_epc ) 
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


# Load the Look up table
# NOTE: before running this script you should generate the lutfile running the 
# example 'compute_lut.py' using type = 'cross-section' instead of 'accel'.
LUT  = LookUpTable(lutfile)

# The computation of the drag requires to specify a density function [kg/m**3]
# The density function must have a call sign like: dens = dens(h), where h is the height
# Here we define a dummy exponential function.
# More complex models can be defined through the classes in pyRTX.classes.Atmosphere
def density(h):
	return (1e-6)*np.exp(-h/100)	# kg/m**3

# Define the CD
CD = 2.2

# Precomputation object
prec = Precompute(epochs = epochs,)
prec.precomputeDrag(lro, body, LUT.moving_frames, accel_frame)
prec.dump()

# Define the drag object
drag = Drag(lro, LUT, density, CD, body, precomputation = prec)

# Compute drag acceleration
accel = drag.compute(epochs, frame = accel_frame, n_cores = 3)[0] * 1e3

### ------------------------------------------------------------------------------------------------------- ###
### PLOT

epochs  = [float( epc - epc_et0)/3600 for epc in epochs]

fig, ax = plt.subplots(3, 1, figsize=(14,8), sharex = True)

ax[0].plot(epochs, accel[:,0], linewidth = 2, color = "tab:blue")
ax[0].set_ylabel('X [m/s^2]')
ax[1].plot(epochs, accel[:,1], linewidth = 2, color = "tab:blue")
ax[1].set_ylabel('Y [m/s^2]')
ax[2].plot(epochs, accel[:,2], linewidth = 2, color = "tab:blue")
ax[2].set_ylabel('Z [m/s^2]')
ax[2].set_xlabel('Hours from t0')
fig.suptitle('Drag in S/C body frame')

plt.tight_layout()
plt.show()

### ------------------------------------------------------------------------------------------------------- ###

