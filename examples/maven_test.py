import sys
import trimesh as tm
import pyRTX
from pyRTX import utils_rt
from pyRTX import physical_utils as pu
import numpy as np
import pickle as pkl 
import time

# Complete example for the computation of the SRP on MAVEN

## Part 1: ##
# GEOMETRY PREPROCESSING
# Import the maven-modified geometry and save references to different materials


# Import the mesh
# Trimesh will import the mesh in different groups based on the different
# material definition in the .obj file
fname = '../example_data/maven_mat.obj'
mesh = tm.load_mesh(fname)
mesh = mesh.dump()

# Save the face idxs for different materials
# to be used in later processing
material_names = ['material%i'%i for i in range(len(mesh))]
counter = 0
store_indxs = []

for i,elem in enumerate(mesh):
	store_indxs.append([counter, counter + len(elem.faces)-1])
	counter += len(elem.faces)


# Concatenate the mesh objects
mesh = mesh.sum()

# Dump the mesh to a pickled binary
# Not strictly necessary in this example since 
# the following steps could directly use 'mesh', 'material_names' and 'store_indxs'
# but the normal usage would imply executing 'Part 1' only once.
# This part is then left here to show how to save the data of the geometry preprocessing.
todump = {'mesh': mesh, 'materials': material_names, 'material_idxs': store_indxs}
with open('../example_data/maven_trimesh.pkl', 'wb') as f:
	pkl.dump(todump, f, protocol = 4)




## PART 2: ##
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

# Compute the sun rays
ray_origins, ray_directions = utils_rt.pixel_plane_opt(	d0, 
							lon,	
							lat, 
							width = width, 
							height = height, 
							ray_spacing = pixel_spacing,
							)


# Compute the normalization factor
norm_factor = 1.0/pixel_spacing**2

	
#3) Perform the ray-tracing
index_tri, index_ray,location, ray_origins, ray_directions, diffusion_pack = utils_rt.RTXkernel(	mesh,  				
													ray_origins, 
													ray_directions, 
													kernel = 'Embree',
													bounces = 4, 
													diffusion = diffusion, 
													num_diffuse = num_diffuse)
										



#3) SRP computation
flux = 1390.0
force = pu.compute_srp(	flux, 				# Solar irradiance (W/m^2)
			mesh, 				# Mesh object
			index_tri, 			# indexes of intersected triangles
			index_ray, 			# indexes of intersecting rays
			location, 			# location of intersections
			ray_origins, 			# ray origins
			ray_directions, 		# ray directions
			norm_factor, 			# normalization factor
			grouped = False, 		# Return the total force (True) or the force for each bounce (False)
			materials = material_dict, 	# Material properties
			diffusion = diffusion, 		# Compute the secondary diffusion after first bounce
			num_diffuse = num_diffuse, 	# Number of diffused ray samples
			diffusion_pack = diffusion_pack # Output of the raytracer containing inputs for the diffusion computation
			)






print(force)


