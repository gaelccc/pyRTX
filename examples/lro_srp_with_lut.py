### ------------------------------------------------------------------------------------------------------- ###

# Example purpose:
# Show the object-oriented interface of the pyRTX library
#
# Example case:
# Compute the srp acceleration for LRO spacecraft, using the values stored in a lookup table.

### ------------------------------------------------------------------------------------------------------- ###
### IMPORTS

import spiceypy as sp
import matplotlib.pyplot as plt
import logging, timeit

from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Planet import Planet
from pyRTX.classes.SRP import SunShadow, SolarPressure 
from pyRTX.classes.Precompute import Precompute
from pyRTX.classes.LookUpTable import LookUpTable
from pyRTX.core.analysis_utils import epochRange2
import logging

from numpy import floor, mod

import warnings
warnings.filterwarnings('ignore')

### ------------------------------------------------------------------------------------------------------- ###
### INPUTS

ref_epc 	= "2010 may 10 09:25:00"
duration    = 10000  									  # seconds
sc_mass		= 2000  									  # can be a float, int or xarray [kg]
timestep    = 100
METAKR      = '../example_data/LRO/metakernel_lro.tm'     # metakernel
obj_path    = '../example_data/LRO/'				      # folder with .obj files
lutfile     = 'luts/lro_accel_lut.nc'					  # lookup table file
base_flux   =  1361.5
ref_radius  =  1737.4

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


# Define the Moon object
moon = Planet(  fromFile      = None,
                radius        = ref_radius,
                name          = 'Moon',
                bodyFrame     = 'MOON_PA',
                sunFixedFrame = 'GSE_MOON',
                units         = 'km',
                subdivs       = 5,
                )


# Precomputation object. This object performs all the calls to spiceypy before 
# calculating the acceleration. This is necessary when calculating the acceleration
# with parallel cores.
prec = Precompute(epochs = epochs,)
prec.precomputeSolarPressure(lro, moon, correction='LT+S')
prec.dump()

# Define the shadow function object
shadow = SunShadow( spacecraft     = lro,
				    body           = 'Moon',
				    bodyShape      = moon,
				    limbDarkening  = 'Eddington',
        			precomputation = prec,
				    )

# Load the Look up table
LUT  = LookUpTable(lutfile)

# Define the solar pressure object (LUT mode)
srp = SolarPressure( lro, 
				     rayTracer      = None,
				     baseflux       = base_flux,   
				     shadowObj      = shadow,
					 precomputation = prec,
					 lookup         = LUT,
				     )

# Managing Error messages from trimesh
# (when concatenating textures, in this case, withouth .mtl definition, trimesh returns a warning that
#  would fill the stdout. Deactivate it for a clean output)
log = logging.getLogger('trimesh')
log.disabled = True

# Compute the SRP acceleration at different epochs and plot it
accel = srp.lookupCompute(epochs) * 1e3
        
log.disabled = False

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

epochs  = [float( epc - epc_et0)/3600 for epc in epochs]

fig, ax = plt.subplots(3, 1, figsize=(14,8), sharex = True)

ax[0].plot(epochs, accel[:,0], linewidth = 2, color = "tab:blue")
ax[0].set_ylabel('X [m/s^2]')
ax[1].plot(epochs, accel[:,1], linewidth = 2, color = "tab:blue")
ax[1].set_ylabel('Y [m/s^2]')
ax[2].plot(epochs, accel[:,2], linewidth = 2, color = "tab:blue")
ax[2].set_ylabel('Z [m/s^2]')
ax[2].set_xlabel('Hours from t0')

plt.tight_layout()
plt.show()

### ------------------------------------------------------------------------------------------------------- ###


