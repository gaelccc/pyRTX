### --------------------------------------------------------------------------- ###

# 			SINGLE PROCESS FOR THE LOOK UP TABLE PARALLEL COMPUTATION

### --------------------------------------------------------------------------- ###

import sys, logging

import spiceypy as sp
import pickle as pkl
import numpy as np

### --------------------------------------------------------------------------- ###
### READ INPUTS

ID = int(sys.argv[1])
METAKR = (sys.argv[2])

with open('inputs/inputs.pkl', 'rb') as f: INPUTS = pkl.load(f)
with open('inputs/attrs.pkl', 'rb') as f: attrs = pkl.load(f)
with open('inputs/srp.pkl', 'rb') as f: srp = pkl.load(f)

# Get attributes
ref_epoch     =  attrs['ref_epoch']
moving_frames =  attrs['moving_frames'].split(',')
dims		  =  attrs['dims'].split(',')
units	      =  attrs['units']
eul_set       =  tuple([int(e) for e in attrs['eul_set'].split(',')])
eul_idxs      =  {ax: idx for idx, ax in enumerate(eul_set)}
sc_model      =  srp.rayTracer.spacecraft.spacecraft_model
base_frame    =  srp.rayTracer.spacecraft.base_frame
materials     =  srp.rayTracer.spacecraft.materials()	
elements      =  list(materials['props'].keys())

### --------------------------------------------------------------------------- ###
### COMPUTE OUTPUTS

log = logging.getLogger('trimesh')
log.disabled = True

sp.furnsh(METAKR)

ref_epc  = sp.str2et(ref_epoch)

IDX, SEQ = INPUTS[ID]
shape = (len(SEQ),3) if 's**2' in units else (len(SEQ),)
OUTPUT  = np.zeros(shape)

# Loop for every sequence
for n, tup in enumerate(zip(IDX,SEQ)):
	
	idxs, seq = tup

	# Mapping dictionary
	map            = { frame: np.zeros((3,)) for frame in moving_frames }
	map['ra']      = -1
	map['dec']     = -1

	# Extract sequence
	for v, val in enumerate(seq):
		
		# Find axis label
		dim = dims[v]

		# Store value in the dictionary
		if dim[:-1] in moving_frames: 
			eul_idx = eul_idxs[int(dim[-1])]
			map[dim[:-1]][eul_idx] = val
		else: map[dim] = val

	# Update spacecraft model
	for frame in moving_frames:
		
		# Find rotation matrix
		eul     = map[frame]
		rot     = sp.eul2m(*eul, *eul_set)
		tmatrix = np.zeros((4,4))
		tmatrix[:3,:3] = rot

		# Update model
		frame_elms = [elem for elem in elements if sc_model[elem]['frame_name'] == frame ]

		# Dump UD rotation for every element
		for element in frame_elms:
			srp.rayTracer.spacecraft.spacecraft_model[element]['UD_rotation'] = tmatrix
	
	# Extract ra, dec
	ra, dec = (map['ra'], map['dec'])

	# Update rayTracer
	srp.rayTracer.rays.update_latlon(lon = ra, lat = dec)

	# Compute normalized accel
	if 's**2' in units:
		value = np.squeeze(srp.compute(ref_epc))
	else:
		srp.rayTracer.trace(ref_epc)
		hit_rays = srp.rayTracer.index_ray_container
		value = len(hit_rays[0])/srp.rayTracer.rays.norm_factor

	OUTPUT[n] = value

# Save output
np.save(f'outputs/output{ID}.npy', OUTPUT)

log.disabled = False

# Unload the SPICE kernels
sp.unload(METAKR)

### ------------------------------------------------------------------------------------------------------- ###