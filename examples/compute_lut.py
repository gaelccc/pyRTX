### -------------------------------------------------------------------- ###

#					    LOOK UP TABLE OBJECT BUILDING

# Example case:
# Generate the normalized optical response lookup table for LRO.

# The look up table is designed to store the "geometry vector" of a spacecraft.
# This vector is representative of how the spacecraft’s shape and surface 
# properties interact with radiation coming from a specific direction. 

# The lookup table helps to speed up the computation of non-gravitational
# accelerations and is MANDATORY for albedo, thermal infrared and drag
# acceleration.

# To compute a lookup table, we need to define a grid of right ascensions
# and declinations that represents the directions from which the radiation 
# is coming in the spacecraft body-fixed frame. 

# Also, we need to specify every frame that is not fixed with respect to 
# the body-fixed frame. These frames are stored in the variable 'moving_frames'. 
# If a frame is not specified inside 'moving_frames', it is represented 
# fixed with respect to the spacecraft body frame. For every moving_frame, 
# a range of euler angles in a specific euler set must be defined.

# The LUT have two computational mode:
# - If units is 'km/s**2', the LUT computes the acceleration for a 
#   normalized radiation flux. The LUT values will be 3x1 vectors. This
#	mode is for srp, albedo and thermal infrared acceleration.
# - If units is 'km**2', the LUT computes the cross-section for the
#	drag acceleration. The LUT values will be single float. 

# NOTE: Here we compute the lookup table for the Lunar Redonnaissance orbiter
# by varying the orientation of the solar array. The High Gain Antenna
# is considered fixed with respect to the bus. 

### -------------------------------------------------------------------- ###
### IMPORTS

import time
import xarray as xr
import spiceypy as sp
import numpy as np
import pickle as pkl

from numpy import floor, mod
from math import ceil

from pyRTX.classes.SRP import SolarPressure
from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.classes.RayTracer import RayTracer
from pyRTX.core.analysis_utils import epochRange2

import multiprocessing as mproc

import timeit, os, itertools, logging

### -------------------------------------------------------------------- ###
### INPUTS & LUT CONFIG
			  	
n_cores    =  10							      # number of cores for parallel computation
grid_res   =  20							      # angular resolution for RA, DEC
angle_res  =  10						          # angular resolution for moving parts
spacing	   =  0.01								  # spacing between rays 

ref_epc	    = "2010 may 10 09:25:00"	          # reference epoch
duration    = 10000  							  # seconds
timestep    = 100

sc_mass 	  = 1								  # the sc mass must be 1 for LUT computation
base_frame    = 'LRO_SC_BUS'				      # sc body-fixed frame
moving_frames = ['LRO_SA',]					      # frames that are not fixed wrt the sc base frame

eul_set = (2,1,3)								  # euler representation for moving frames attitude

obj_path = '../example_data/LRO/'		          # path for 3D shape elements

METAKR = '../example_data/LRO/metakernel_lro.tm'  # metakernel

units  = 'km/s**2'	                              # units for lookup table values: km/s**2 or km**2

lutfile =  'luts/lro_accel_lut.nc'	    		  # output file

### --------------------------------------------------------------------------- ###
### LUT LIMITS

sp.furnsh(METAKR)

# List of right ascension and declination values for the incoming rays
RA   = np.linspace(0, 360, int(360/grid_res) + 1) * np.pi / 180
DEC  = np.linspace(-90, 90, int(180/grid_res) + 1) * np.pi / 180

###  Here we find the limits for every moving frame, in terms of euler angles  ###

# Define timespan
eul_limits = {frame: {e: [] for e in eul_set} for frame in moving_frames}
epc_et0  = sp.str2et( ref_epc )
epc_et1  = epc_et0 + duration
epochs   = epochRange2(startEpoch = epc_et0, endEpoch = epc_et1, step = timestep)

# Find euler limits
for frame in moving_frames:
    
	EUL = np.zeros((len(epochs),3))

	for i, epc in enumerate(epochs):

		rot = sp.pxform(frame, base_frame, epc)
		EUL[i,:] = np.array(sp.m2eul(rot, *eul_set)) * 180 / np.pi
 
	for i, e in enumerate(eul_set):
  
		if max(abs(EUL[:,i].max()), abs(EUL[:,i].min())) >= grid_res:
			eul_limits[frame][e] = [EUL[:,i].min() - 0.05, EUL[:,i].max() + 0.05]

sp.unload(METAKR)

### --------------------------------------------------------------------------- ###
### OBJECT DEFINITION

# Define a spacecraft model
spacecraft_model = {						        
                          
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
 
 # Elements with moving frames must have an user defined rotation
for elem in spacecraft_model.keys():
    if any([spacecraft_model[elem]['frame_name'] == frame for frame in moving_frames]):
        spacecraft_model[elem]['frame_type'] = 'UD'
        spacecraft_model[elem]['UD_rotation'] = np.identity(4)

# Define the Spacecraft Object (Refer to the class documentation for further details)
lro = Spacecraft( name = 'LRO',     
				  base_frame = 'LRO_SC_BUS', 					     # Name of the spacecraft body-fixed frame
                  mass = sc_mass,									 # The mass should be 1 for LUT computation 
				  spacecraft_model = spacecraft_model,
				)

# Define the Sun rays object
rays = PixelPlane( spacecraft = lro,
				   mode = 'Fixed',
				   width = 15,
				   height = 15,
				   ray_spacing = spacing, 
				   lon = 0,
				   lat = 0,
				   distance = 30)

# Define the Ray Tracer
rtx = RayTracer( lro,                    # Spacecraft object
				 rays,                   # pixelPlane object
				 kernel = 'Embree3',     # The RTX kernel to use (use Embree 3)
				 bounces = 1,            # The number of bounces to account for
				 diffusion = False,      # Account for secondary diffusion
				 ) 

# Define the Solar Pressure Object 
# NOTE: for LUT computation the baseflux must be set to None.
srp = SolarPressure( lro, rtx, baseflux = None, )

### -------------------------------------------------------------------- ###
### LUT INITIALIZATION

# Time initialization
tic = timeit.default_timer()

# Refresh inputs and output directiories
os.system('rm inputs/*')
os.system('rm outputs/*')

# Save srp object
with open('inputs/srp.pkl', 'wb') as f: pkl.dump(srp, f)

# Deactivate trimesh logging
log = logging.getLogger('trimesh')
log.disabled = True

print('\n *** Calculating dimension ...')

# Build dimensions and axes
axes = []
dims = []
for name in moving_frames:
    for ax in eul_set:
        
        if not len(eul_limits[name][ax]): continue
        lb = eul_limits[name][ax][0]
        ub = eul_limits[name][ax][1]
        
        dims.append('%s%d'%(name,ax))
        axes.append(np.linspace(lb, ub, ceil((ub-lb)/angle_res) + 1)*np.pi/180)	  

# Append ra and dec
dims.append('ra')
dims.append('dec')
dims.append('value')
axes.append(RA) 
axes.append(DEC) 

# Build attribute dictionary for xarray
attrs = {
	'moving_frames': ",".join(moving_frames),
	'base_frame': base_frame,
	'units': units,
	'ref_epoch': ref_epc,
	'eul_set': ",".join([str(e) for e in eul_set]),
	'dims': ",".join(dims),
	}

# Save attributes dictionary
with open('inputs/attrs.pkl', 'wb') as f: pkl.dump(attrs, f)

# Compute shape for xarray
shape = tuple([len(r) for r in axes] + [3])

# Build coordinates
coords = {dims[i]: vals for i, vals in enumerate(axes)}

### -------------------------------------------------------------------- ###
### LUT PARALLEL COMPUTATION 

print(f'\n *** LUT size: {shape} ...')

# Init data array
data  = np.zeros(shape)

# Find all permutations
SEQ    = list(itertools.product(*axes))
IDX    = list(itertools.product(*[range(l) for l in shape[:-1]]))
steps  = [int(i) for i in np.linspace(0,len(SEQ),n_cores)]
SEQ    = [SEQ[i:j] for (i,j) in zip(steps[:-1],steps[1:])]
IDX    = [IDX[i:j] for (i,j) in zip(steps[:-1],steps[1:])]
INPUTS = [(I,S) for I,S in zip(IDX,SEQ)]

# Save inputs file
with open('inputs/inputs.pkl', 'wb') as f: pkl.dump(INPUTS, f)

### -------------------------------------------------- ###
### Multiprocessing target function

def process(ID, METAKR):
	os.system(f'python task_lut.py {ID} {METAKR}')
	return

### -------------------------------------------------- ###

print('\n *** Filling the LUT values ...')

# --------------------------------- #
# PYTHON MULTIPROCESSING

p = [0.] * len(INPUTS)

# Process in parallel
for ID in range(len(INPUTS)):
	p[ID] = mproc.Process( target=process, args=(ID, METAKR,) )

for ID in range(len(INPUTS)):       
	p[ID].start()
	while sum([pi.is_alive() for pi in p]) >= n_cores: time.sleep(0.5)

for ID in range(len(INPUTS)): 
	while(p[ID].is_alive()): p[ID].join(1)

# --------------------------------- #

# Fill LUT values
for ID in range(len(INPUTS)):
    result = np.load(f'outputs/output{ID}.npy')
    for i, idxs in enumerate(INPUTS[ID][0]):
        data[tuple(idxs)] = result[i]

print('\n *** Generating the x-array ...')

# Define X-array LUT
data = xr.Dataset( data_vars = {'look_up_table': (dims, data)}, 
				   coords    = coords,
				   attrs     = attrs,)

print(f'\n *** LUT completed!\n')

# Reactivate trimesh logging
log.disabled = False

# Save
data.to_netcdf(lutfile, encoding = data.encoding.update({'zlib': True, 'complevel': 1}))

### ... Elapsed time
toc = timeit.default_timer()
time_min = int(floor((toc-tic)/60))
time_sec = int(mod((toc-tic), 60))
print("")
print("\t Elapsed time: %d min, %d sec" %(time_min, time_sec))
print("")

### -------------------------------------------------------------------- ###