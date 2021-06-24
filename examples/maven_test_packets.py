import sys
sys.path.append('..')

import trimesh as tm
import pyRTX
from pyRTX import utils_rt
from pyRTX import physical_utils as pu
import numpy as np
import pickle as pkl 
import time

# Complete example for the computation of the SRP on MAVEN
# In this example the 'multiple packets' approach is used
# (see README)


# RAY TRACING
# In this example a complete ray tracing step is performed:
# 1) importing the geometry and setting up the materials
# 2) generating the sun rays
# 3) ray tracing
# 4) SRP force computation



#1) Import the pre-processed mesh and materials
with open('../example_data/maven_trimesh.pkl', 'rb') as f:
	data = pkl.load(f)

mesh = data['mesh']
materials = data['materials']
material_idxs = data['material_idxs']

# Create a dictionary of specular and diffusive properties
props = {'material0': {'specular': 0.5, 'diffuse': 0.1},  # Generic metal
		 'material1': {'specular': 0.5, 'diffuse': 0.1}, # Generic metal
		 'material2': {'specular': 0.01, 'diffuse': 0.1}, # SA back
		 'material3': {'specular': 0.045, 'diffuse': 0.45}, # SA front
		 'material4': {'specular': 0.5, 'diffuse': 0.1}, # Generic metal
		 'material5': {'specular': 0.5, 'diffuse': 0.1}, # Generic metal
		 'material6': {'specular': 0.5, 'diffuse': 0.1}, # Generic
		 'material7': {'specular': 0.5, 'diffuse': 0.1}, # Generic
		 'material8': {'specular': 0.18, 'diffuse': 0.28}, # HGA back
		 'material9': {'specular': 0.5, 'diffuse': 0.04},   # Silver foil
		 }


material_dict = {'idxs': material_idxs, 'props': props}

#2) Generation of the sun rays (or pixel array)
lon = 45.0*np.pi/180.0  # Right ascension of the sun wrt. spacecraft body frame
lat = 0.0               # Declination of the sun wrt. spacecraft body frame
width = 15              # Width (meters) of the pixel array
height = 15		# Height (meters) of the pixel array
d0 = 100		# Distance of the pixel array wrt. origin of the mesh
pixel_spacing = 0.04	# Spacing (meters) of the rays
diffusion = True	# Compute also the diffusion propagation after first bounce
num_diffuse = 120	# Number of diffused rays sampled 
packets = 5		# Number of parts in which to subdivide the sun rays

# Compute the sun rays
ray_origins_p, ray_directions_p = utils_rt.pixel_plane_opt(	
							d0, 
							lon,	
							lat, 
							width = width, 
							height = height, 
							ray_spacing = pixel_spacing,
							packets = packets
							)

# Compute the normalization factor
norm_factor = 1.0/pixel_spacing**2

# Initialize the force array
force = np.array([0,0,0], dtype = 'float64')
for i in range(packets):
	#3) Perform the ray-tracing
	index_tri, index_ray,location, ray_origins, ray_directions, diffusion_pack = utils_rt.RTXkernel(	mesh,  				
														ray_origins_p[i], 
														ray_directions_p[i], 
														kernel = 'Embree',
														bounces = 4, 
														diffusion = diffusion, 
														num_diffuse = num_diffuse)
											



	#3) SRP computation

	flux = 1390.0
	force_temp = pu.compute_srp(
				flux, 				# Solar irradiance (W/m^2)
				mesh, 				# Mesh object
				index_tri, 			# indexes of intersected triangles
				index_ray, 			# indexes of intersecting rays
				location, 			# location of intersections
				ray_origins, 			# ray origins
				ray_directions, 		# ray directions
				norm_factor, 			# normalization factor
				grouped = True, 		# Return the total force (True) or the force for each bounce (False)
				materials = material_dict, 	# Material properties
				diffusion = diffusion, 		# Compute the secondary diffusion after first bounce
				num_diffuse = num_diffuse, 	# Number of diffused ray samples
				diffusion_pack = diffusion_pack # Output of the raytracer containing inputs for the diffusion computation
				)


	force += force_temp



print(force)


