##################################
# Physics-related Utilities for ray tracing (part of pyRTX module)
# 
# Developed by: Gael Cascioli 2021



import numpy as np 
import trimesh as tm 
import sys
sys.path.append('/home/cascioli/RTX/code/')
from pyRTX.utils_rt import(get_centroids, get_surface_normals_and_face_areas, reflected, reflected)
import time
from pyRTX import constants
from numba import jit




## Optimized version

def preprocess_RTX_geometry(mesh_obj):
	"""
	Preprocess the RTX output to obtain the information required
	"""

	V = np.array(mesh_obj.vertices, dtype=np.float64)
	F = np.array(mesh_obj.faces, dtype=np.int64)

	#P = get_centroids(V, F)
	N, A = get_surface_normals_and_face_areas(V, F)	

	return V, F,  N, A

def preprocess_materials(material_dict):
	"""
	Get the material properties and set up an array for handling
	Parameters:
	material_dict: a dictionary with the shape:
		{'props': dictionary of properties for each material, 'idxs': indexes of faces associated with each material}
	
	Returns:
	prop_container: a (len(mesh), 2) numpy array containing [specular, diffuse] coefficients for each face of the mesh
	"""

	properties = material_dict['props']
	material_names = properties.keys()
	mat_idxs = material_dict['idxs']
	last_idx = mat_idxs[-1][-1]
	prop_container = np.zeros((last_idx+1,2))

	for i,elem in enumerate(material_names):
		spanned_idxs = range(mat_idxs[i][0], mat_idxs[i][1]+1)
		prop_container[spanned_idxs,0] = properties[elem]['specular']
		prop_container[spanned_idxs,1] = properties[elem]['diffuse']

	return prop_container


def srp_core(flux, indexes_tri, indexes_ray, N, S, norm_factor, mesh_obj, materials = 'None', diffusion = False, num_diffuse = None, diffusion_pack = None):
	"""
	Core of SRP computation.
	Highly vectorized version. For explicit algorithm implementation refer to the old version

	Parameters:
	flux: solar flux (float, W/m^2)
	indexes_tri: indexes of intersected triangles
	indexes_ray: indexes of intersecting rays
	N: normals
	S: incident direction vectors
	norm_factor: normalization factor computed from ray spacing (float)
	mesh_obj: trimesh.Trimesh object [Not used for now, will be used when interrogating mesh
				for surface properties]

	Returns:
	force: np.array of SRP force
	"""

	c = constants.c
	if isinstance(materials, str) and materials == 'None':
		rho_s = 0.1  #Hardcoded and used just for the dummy case in which the materials are not provided
		rho_d = 0.1  
	else:
		rho_s = materials[:,0][indexes_tri]
		rho_d = materials[:,1][indexes_tri]

	force = np.array([0,0,0], dtype = 'float64')

	counter = 0

	dA = np.ones(len(indexes_ray))/norm_factor
	s = S[indexes_ray]
	n = N[indexes_tri]
	r = reflected(s,n)


	aa = flux[indexes_ray]*dA/c 


	# When using vectorization, this operation must be done through np.multiply operator
	# bb = (s  - rho_s * r - 2.0/3 * rho_d * n)
	#term_2 = np.multiply(r.T, ni*mi).T
	#term_3 = np.multiply(n.T, ni*(1-mi)).T
	term_2 = np.multiply(r.T, rho_s).T
	term_3 = np.multiply(n.T, rho_d).T

	bb = s - term_2 - 2.0/3*term_3 

	forc = np.multiply(bb.T, aa).T

	force = np.sum(forc, axis = 0)
	newFlux = flux[indexes_ray]*rho_s

	# Handle the secondary diffusions
	if diffusion:
		idx_tri_previous = diffusion_pack[4]
		idx_tri_actual = diffusion_pack[0]
		idx_ray = diffusion_pack[1]
		idx_ray_previous = diffusion_pack[6]
		ray_directions = diffusion_pack[2]
		original_flux = diffusion_pack[5]
		rho_d = np.repeat(materials[:,1][idx_tri_previous], num_diffuse, axis = 0)
		
		original_flux = np.repeat(original_flux[idx_ray_previous], num_diffuse, axis = 0)
		flux = original_flux * rho_d / num_diffuse

		new_rho_d = materials[:,1][idx_tri_actual]
		new_rho_s = materials[:,0][idx_tri_actual]
		dA = np.ones(len(idx_ray))/norm_factor
		aa = flux[idx_ray]*dA/c
		s = ray_directions[idx_ray]
		n = N[idx_tri_actual]
		r = reflected(s,n)
		term_2 = np.multiply(r.T, new_rho_s).T
		term_3 = np.multiply(n.T, new_rho_d).T
	
		bb = s - term_2 - 2.0/3*term_3 
	
		forc2 = np.multiply(bb.T, aa).T

		force2 = np.sum(forc2, axis = 0)
		
		force = force + force2


	return force, newFlux









def compute_srp(flux, mesh_obj, index_tri, index_ray, location, ray_origins, ray_directions, pixel_spacing, materials = 'None', grouped = True,
		diffusion = False, num_diffuse = None, diffusion_pack = None):
	"""
	Compute the SRP force

	Parameters:
	flux: Solar input flux [W/m^2]
	A: areas of the mesh faces
	s: incident ray directions
	r: reflcted ray directions
	n: normal unit vector to the faces


	"""

	# Compute geometric quantities	
	V, F, N, A = preprocess_RTX_geometry(mesh_obj)
	
	# Retrieve material properties
	if materials != 'None':
		properties = preprocess_materials(materials)
	else:
		properties = 'None'


	# Automatically get the number of bounces
	n_bounce = len(index_tri)



	if grouped:
		force = np.array([0,0,0], dtype = 'float64')
	else:
		force = []

	for i in range(n_bounce):
		if i == 0:
			flux = np.full(len(ray_directions[i]), flux)
			if diffusion:
				diffusion_pack.append(index_tri[i]) #  Append the emitting triangle indexes
				diffusion_pack.append(flux)  # Append the original flux
				diffusion_pack.append(index_ray[i])   # not needed??


		idx_tri = index_tri[i]
		idx_ray = index_ray[i]
		S = ray_directions[i]

		if i == 1 and diffusion:
			force_temp, flux = srp_core(flux, idx_tri, idx_ray, N, S, pixel_spacing, mesh_obj, materials = properties, diffusion = diffusion, num_diffuse = num_diffuse, diffusion_pack = diffusion_pack)
		else:
			force_temp, flux = srp_core(flux, idx_tri, idx_ray, N, S, pixel_spacing, mesh_obj, materials = properties)

		if grouped:
			force += force_temp
		else:
			force.append(force_temp)
		

	return force







### NON OPTIMIZED VERSIONS
# Kept here for backward compatibility and possible debuggings
'''
#def preprocess_RTX_geometry(mesh_obj):
#	"""
#	Preprocess the RTX output to obtain the information required
#	"""
#
#	V = np.array(mesh_obj.vertices, dtype=np.float64)
#	F = np.array(mesh_obj.faces, dtype=np.int64)
#
#	P = get_centroids(V, F)
#	N, A = get_surface_normals_and_face_areas(V, F)	
#
#	return V, F, P, N, A
#
#
#def srp_core(flux, indexes_tri, indexes_ray, A, N, S, pixel_spacing, mesh_obj, diffusion = False, num_diffuse = None, diffusion_pack = None):
#
#
#	ni = 0.1  #Hardcoded for now. Must be read from mesh metadata in the future
#	mi = 0.1  #Hardcoded for now. Must be read from mesh metadata in the future
#	c = 3e8
#	force = np.array([0,0,0], dtype = 'float64')
#	Aa = 0
#	counter = 0
#	for i,ind in enumerate(indexes_ray):
#
#
#		A_Norm = pixel_spacing #NOTE: CHECK THIS
#
#		#dA = A[indexes_tri[i]]/A_Norm
#		dA = 1/A_Norm
#		#dA = mesh_obj.area_faces[indexes_tri[i]]/A_Norm
#		n = N[indexes_tri[i]]
#
#		s = S[ind]
#		cosT = -np.dot(s,n)
#		r = reflected(s, n)
#
#
#		Aa += dA #*cosT
#
#
#		force +=  flux*dA/c *(s  - ni * mi * r - 2.0/3 * ni*(1-mi)*n)
#		counter +=1
#
#	
#	return force
#
#
#
#
#
#
#def compute_srp(flux, mesh_obj, index_tri, index_ray, location, ray_origins, ray_directions, pixel_spacing, grouped = True, diffusion = False, num_diffuse = None, diffusion_pack = None):
#	"""
#	Compute the SRP force
#
#	Parameters:
#	flux: Solar input flux [W/m^2]
#	mesh_obj: trimesh class for object
#	index_tri, index_ray, location, ray_origins, ray_directions: output of the raytracer
#	pixel_spacing: normalization factor 
#	diffusion: (bool) Select the first-bounce diffusion computation
#	num_diffuse: (int) number of samples of the Lambert distribution for first-bounce diffuse computations
#	diffusion_pack: "diffusion_pack" output from the raytracer (see RTXkernel docs)
#
#
#	"""
#
#	# Compute geometric quantities
#	V, F, P, N, A = preprocess_RTX_geometry(mesh_obj)
#
#
#
#	# Automatically get the number of bounces
#	n_bounce = np.shape(index_tri)[0]
#
#	if grouped:
#		force = np.array([0,0,0], dtype = 'float64')
#	else:
#		force = []
#	for i in range(n_bounce):
#		idx_tri = index_tri[i]
#		idx_ray = index_ray[i]
#		S = ray_directions[i]
#
#		if grouped:
#			force += srp_core(flux, idx_tri, idx_ray, A, N, S, pixel_spacing, mesh_obj, diffusion = diffusion, num_diffuse = num_diffuse, diffusion_pack = diffusion_pack)
#		else:
#			force.append( srp_core(flux, idx_tri, idx_ray, A, N, S, pixel_spacing, mesh_obj, diffusion = diffusion, num_diffuse = num_diffuse, diffusion_pack = diffusion_pack) )
#
#	return force
#
'''
