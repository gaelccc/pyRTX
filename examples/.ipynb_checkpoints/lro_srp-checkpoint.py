### ------------------------------------------------------------------------------------------------------- ###

# Example purpose:
# Show the object-oriented interface of the pyRTX library
#
# Example case:
# Compute the srp acceleration for LRO spacecraft, using the SPICE trajectory and frames

### ------------------------------------------------------------------------------------------------------- ###
### IMPORTS

import spiceypy as sp
import xarray as xr
import matplotlib.pyplot as plt
import logging, timeit

from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Planet import Planet
from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.classes.RayTracer import RayTracer
from pyRTX.classes.SRP import SunShadow, SolarPressure 
from pyRTX.classes.Precompute import Precompute
from pyRTX.core.analysis_utils import epochRange2
import logging

from numpy import floor, mod

import warnings
warnings.filterwarnings('ignore')

### ------------------------------------------------------------------------------------------------------- ###
### INPUTS

ref_epc 	=  "2010 may 10 09:25:00"
duration    =  10000  									  # seconds
timestep    =  100
spacing	    =  0.01
METAKR      =  '../example_data/LRO/metakernel_lro.tm'     # metakernel
obj_path    =  '../example_data/LRO/'				       # folder with shape .obj files
base_flux   =  1361.5
ref_radius  =  1737.4
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
epc_et0 = sp.str2et( ref_epc ) 
epc_et1 = epc_et0 + duration
epochs  = epochRange2(startEpoch = epc_et0, endEpoch = epc_et1, step = timestep)

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


# Define the Sun rays object
rays = PixelPlane( spacecraft  = lro,         # Spacecraft object 
			       mode        = 'Dynamic',   # Mode: can be 'Dynamic' ( The sun orientation is computed from the kernels), or 'Fixed'
			       distance    = 100,	      # Distance of the ray origin from the spacecraft
			       source      = 'Sun',       # Source body (used to compute the orientation of the rays wrt. spacecraft)
			       width       = 10,	      # Width of the pixel plane
			       height      = 10,          # Height of the pixel plane
			       ray_spacing = spacing,     # Ray spacing (in m)
				   )


# Define the ray tracer
rtx = RayTracer( lro,                    # Spacecraft object
                 rays,                   # pixelPlane object
                 kernel = 'Embree3',     # The RTX kernel to use
                 bounces = 2,            # The number of bounces to account for
                 diffusion = False,      # Account for secondary diffusion
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

# Define the solar pressure object
srp = SolarPressure( lro, 
				     rtx,
				     baseflux       = base_flux,    # Here we use the None option to obtain the generalized geometry vector, used also for the computation of albedo and thermal infrared
				     shadowObj      = shadow,
					 precomputation = prec,
				     )

# Managing Error messages from trimesh
# (when concatenating textures, in this case, withouth .mtl definition, trimesh returns a warning that
#  would fill the stdout. Deactivate it for a clean output)
log = logging.getLogger('trimesh')
log.disabled = True

### ------------------------------------------------------------------------------------------------------- ###
### COMPUTATIONS

# Compute the SRP acceleration 
accel = srp.compute(epochs, n_cores = n_cores) * 1e3
        
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


