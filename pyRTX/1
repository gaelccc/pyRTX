##################################
# Utilities for ray tracing (part of pyRTX module)
# 
# Developed by: Gael Cascioli 2021

import numpy as np
import trimesh
from abc import ABC
import os
import pickle as pkl
from numba import jit
import multiprocessing as mproc
from pyRTX.defaults import dFloat, dInt
try:
	import embree
except:
	pass
#except ImportError:
#	print("""Could not import Embree3 library. \n Be sure that: 
#		\n 1) Embree3 is installed\n 2) you activated embree variables (source /path/to/lib/embree-vars.sh)""")


# No more imports after here
#___________________________________________________________

############################################################
# Define general utils
###########################################################


def parallelize( iterator, function, chunks ):
	"""
	Define a general parallelization framework to speedup computations

	Parameters:
	iterator: the array-like object over which to parallelize
	functiuon: the function that must be called by each subprocess. The function should be of the form y = f(iterator)
	chunks: number of parallel workers

	"""



	with mproc.Pool(chunks) as p:
		result = p.map(function, iterator)

	return result


def chunker(iterator, chunks):
	
	return np.array_split(iterator, chunks)
	

def pxform_convert(pxform):
	"""
	Convert a spice-generated rotation matrix (pxform) to the format required by trimesh
	"""
	pxform = np.array([pxform[0],pxform[1],pxform[2]], dtype = dFloat)

	p = np.append(pxform,[[0,0,0]],0)

	mv = np.asarray(np.random.random(), dtype = dFloat)
	p = np.append(p,[[0],[0],[0],[0]], 1)
	return p


def block_normalize(V):
	"""
	get the unit vectors associated to a block of vectors of shape
	(N,3)

	

	"""

	if V.ndim > 1:
		return  V / np.linalg.norm(V, axis = 1).reshape(len(V), 1)
	else:
		return V / np.linalg.norm(V)


def block_dot(a,b):
	"""
	Perform block dot product between two arrays of shape (N,m), (N,m)

	Parameters
	a, b [np.array (N,m)]

	Returns
	c [np.array (N,)]
	"""

	return np.sum(a*b, axis = 1)


def pixel_plane(d0, lon, lat, width = 1, height = 1, ray_spacing = .1):
	""""Generate a pixel array for raytracing ad defined in Li et al., 2018
	This is the "fully exposed version" to explicitly show the algorithm. 
	Parameters:
	d0: Distance of the pixel array from the center (in meters)
	lat: Latitude of the pixel array center (in rad)
	lon: Longitude of the pixel array center (in rad)
	width: The width of the plane(in meters). Default = 1
	height: the height of the plane(in meters). Default = 1
	ray_spacing: the spacing of the rays (in meters). Default = 0.1

	Returns: 
	locs: Pixel locations as a numpy array
	dirs: the ray directions as a numpy array
	"""



	w2 = width/2
	h2 = height/2
	# Build the direction vector 
	x0 = np.array([-d0*np.cos(lon)*np.cos(lat), -d0*np.sin(lon)*np.cos(lat), -d0*np.sin(lat)])

	# Build the transformation matrix
	R1 = np.array( [[np.cos(lon), -np.sin(lon), 0],
					[np.sin(lon), np.cos(lon), 0],
					[0,0,1]]
		)
	R2 = np.array([[np.cos(-lat), 0, np.sin(-lat)],
					[0,1,0],
					[-np.sin(-lat), 0, np.cos(-lat)]])
	R = R1@R2


	# Build the pixel matrix
	basic_coords = np.zeros(((int(width/ray_spacing)+1)* (int(height/ray_spacing)+1), 3))
	basic_dirs = np.zeros_like(basic_coords)
	counter = 0	
	for i, w in enumerate(np.linspace(-w2, w2, num = int(width/ray_spacing)+1 )):
	 	for j, h in enumerate(np.linspace(-h2, h2, num = int(height/ray_spacing)+1)):
	 		basic_coords[counter, :] = R@np.array([0, w, h]) - x0
	 		basic_dirs[counter, :] = x0/np.linalg.norm(x0)
	 		counter += 1


	# Return the output in the shape required by trimesh
	return basic_coords, basic_dirs








@jit(nopython = True)
def fast_vector_build(linsp1, linsp2, dim1, dim2):
	"""
	Further accelerate the ray vector generation using numba's jit vectorization
	"""
	basic_coords = np.zeros((dim1*dim2, 3))
	counter = 0

	for w in linsp1:
	 	for h in linsp2:
	 		basic_coords[counter, :] = [0, w, h]
	 		counter += 1
	return basic_coords	


def pixel_plane_opt(d0, lon, lat, width = 1, height = 1, ray_spacing = .1, packets = 1):
	""""Generate a pixel array for raytracing ad defined in Li et al., 2018
	This is the "optimized version". To explicitly see the algorithm refer to the function definition
	without _opt extension.
	Parameters:
	d0: [float] Distance of the pixel array from the center (in meters)
	lat: [float] Latitude of the pixel array center (in rad)
	lon: [float] Longitude of the pixel array center (in rad)
	width: [float] The width of the plane(in meters). Default = 1
	height: [float] the height of the plane(in meters). Default = 1
	ray_spacing: [float] the spacing of the rays (in meters). Default = 0.1
	packets: [int] the number of 'ray packets' to return. This is implemented to avoid the segmentation
		 fault triggered by the raytracer when the number of rays is too high

	Returns: 
	locs: [numpy array (n_rays, 3)] Pixel locations as a numpy array
	dirs: [numpy array (n_rays, 3)] the ray directions as a numpy array
	"""



	w2 = width/2
	h2 = height/2
	# Build the direction vector 
	x0 = np.array([-d0*np.cos(lon)*np.cos(lat), -d0*np.sin(lon)*np.cos(lat), -d0*np.sin(lat)])
	x0_unit = x0/np.linalg.norm(x0)
	# Build the transformation matrix
	R1 = np.array( [[np.cos(lon), -np.sin(lon), 0],
					[np.sin(lon), np.cos(lon), 0],
					[0,0,1]]
		)
	R2 = np.array([[np.cos(-lat), 0, np.sin(-lat)],
					[0,1,0],
					[-np.sin(-lat), 0, np.cos(-lat)]])
	R = R1@R2


	# Build the pixel matrix
	
	dim1 = int(width/ray_spacing)+1
	dim2 = int(height/ray_spacing)+1
	basic_coords = np.zeros((dim1*dim2, 3))
	basic_dirs = np.full(basic_coords.shape, x0_unit)

	linsp1 = np.linspace(-w2, w2, num = dim1 )
	linsp2 = np.linspace(-h2, h2, num = dim2)


	basic_coords = fast_vector_build(linsp1, linsp2, dim1, dim2)

	basic_coords = np.dot(np.array(basic_coords), R.T)
	basic_coords -= x0


	# Return the output in the shape required by trimesh

	if not packets == 1:	
		basic_coords = np.array_split(basic_coords, packets)
		basic_dirs = np.array_split(basic_dirs, packets)

	return basic_coords, basic_dirs

def reflected(incoming, normal):
	"""
	Compute the reflected unit vector given the incoming and the normal
	Numpy vectorized version of 'reflected' 
	Parameters:
	incoming: [numpy array (number of rays, 3)]  incoming rays
	normal: [numpy array (number of rays, 3)] surface normals associated to each incoming ray

	Returns:
	reflected: [numpy array (number of rays, 3)] reflected rays

	"""

	return incoming - 2*np.multiply(normal.T,np.einsum('ij,ij->i',incoming, normal)).T

@jit(nopython = True)
def get_orthogonal(v):
	"""
	Get a unit vector orthogonal to v

	Parameters:
	v: [numpy array (3,)]

	Returns: 
	x: [numpy array(3,)]
	"""
	x = np.random.random(3)
	x -= x.dot(v)*v
	x  = x/ np.linalg.norm(x)

	return x

@jit(nopython = True)
def sample_lambert_dist(normal, num = 100):
	"""
	Generate a cloud of vectors following the Lambert cosine distribution

	Parameters: 
	normals: [numpy array] the normal vector to the face
	num: [int] the number of samples required
	
	Returns:
	v: [numpy array (num, 3)] array of the sampled vectors
	"""

	theta = np.arccos(np.sqrt(np.random.random(num)))
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	psi = np.random.random(num)*2*np.pi

	a = sin_theta*np.cos(psi)
	b = sin_theta*np.sin(psi)
	c = cos_theta

	t1 = get_orthogonal(normal)
	t2 = np.cross( normal , t1)


	v =np.zeros((num, 3))
	for i in range(num):
		v[i] = a[i]*t1 + b[i]*t2 + c[i]*normal
	return v

def _core_diffuse(normals, diffuse_directions, num):

	for i,n in enumerate(normals):
		diff_dirs = sample_lambert_dist(n, num = num)	
		
		diffuse_directions[i*num: (i+1)*num] = diff_dirs 
	return diffuse_directions




def diffuse(normals, num = 10):
	"""
	Compute num diffuse reflection directions sampling a lambert cosine distribution
	
	Parameters:
	normals: [numpy array (N, 3)] normal unit vectors
	num: number of samples for each normal

	Returns:
	diffuse_directions: [numpy array (N*num, 3)] 

	"""

	diffuse_directions = np.repeat(normals, num, axis = 0)*0.0  

	for i,n in enumerate(normals):
		diff_dirs = sample_lambert_dist(n, num = num)	
		
		diffuse_directions[i*num: (i+1)*num] = diff_dirs 

	return diffuse_directions

	#return _core_diffuse(normals, diffuse_directions, num)

def compute_secondary_bounce(location, index_tri, mesh_obj, ray_directions, index_ray, diffusion = False, num_diffuse = None):
	"""
	Prepare the quantities required for iterating the raytracer

	Parameters:
	location: UNUSED. This input is maintained just to change the variable name
	index_tri: [numpy array (N,)] indexes of the faces intersected by the rays
	mesh_obj: [trimesh.Trimesh] the mesh object
	ray_directions: [numpy array (L, 3)] the directions of the incoming rays
	index_ray: [numpy array (M,)] the indexes of the rays that effectively intersected the N faces
	diffusion: [bool] boolean flag to select wether to compute the secondary diffusion rays (Default: False)
	num_diffuse: [None or int] number of samples for the diffusion computation (Default: None)

	Returns:
	location: same as location in input
	reflect_dirs: [numpy array (L,3)] the specularly reflected directions of the rays
	diffuse_dirs [numpy array (L*num_diffuse, 3) or -1] (if requested) the direction of the diffused rays



	"""
	reflect_dirs = np.zeros_like(location)
	normals = mesh_obj.face_normals



	reflect_dirs = reflected(ray_directions[index_ray], normals[index_tri])

	if diffusion: 
		diffuse_dirs = diffuse(normals[index_tri], num = num_diffuse)
	
	else:
		diffuse_dirs = -1  # Dummy variable for return values management
	
	return location, reflect_dirs, diffuse_dirs


####################################################################################################
# Saving utilities

def save_for_visualization(outputFilePath, mesh, ray_origins, ray_directions, location, index_tri, diffusion_pack ):
	"""
	Save a pickled dictionary useful for visualization scripts (see visual_utils)

	Parameters:
	outputFilePath: [str] output file for saving (should end with .pkl)
	mesh: [trimesh.Trimesh] mesh object
	ray_origins: [numpy array (bounce_number, N, 3)] output of the raytracer
	ray_directions: [numpy array (bounce_number, N, 3)] output of the raytracer
	location: [numpy array (bounce_number, N, 3)] locations of intersection points
	index_tri: [numpy array (bounce_number, M, 3)] indexes of intersected triangles
	diffusion_pack: [list] output of the raytracer with the same name

	Returns:
	None

	"""
	outdict = {'mesh': mesh, 'ray_origins': ray_origins, 'ray_directions': ray_directions, 'locations': location, 'index_tri': index_tri, 'diffusion_pack': diffusion_pack}

	with open(outputFilePath, 'wb') as f:
		pkl.dump(outdict, f, protocol = 4)



def exportEXAC(satelliteID, data,tstep, startTime, endTime, outFileName):
	"""
	GEODYN-EXAC file exporter
	Parameters:
	satelliteID: [int] satellite identifier
	data: [np.array (N, 3)] acceleration data to be written
	tstep: [int or float] time step (in seconds)
	startTime: [datetime.datetime] start time of data
	endTime: [datetime.datetime] end time of data
	outFileName: [str] name of output file
	"""
	from scipy.io import FortranFile
	import datetime

	satid = satelliteID
	date0 = startTime
	date1 = endTime
	dt = tstep
	deltatime = datetime.timedelta(seconds = dt)
	outfile = FortranFile(outFileName, 'w')
	
	# General Header
	masterhdr = np.array([-6666666.0, 1, 1, 0, 0, 0, 0, 0, 0])
	outfile.write_record(masterhdr)

	# Satellite specific header
	sathdr=np.array([-7777777.0, 1, satid, dt, float(date0.strftime('%Y%m%d%H%M%S')[2:]), float(date0.strftime('%f')), float(date1.strftime('%Y%m%d%H%M%S')[2:]), float(date1.strftime('%f')), 0])
	outfile.write_record(sathdr)

	# Data records
	date = date0-deltatime
	for d_elem in data:
		date = date + deltatime
		datarec = np.array([float(date.strftime('%Y%m%d%H%M%S')[2:]), float(date.strftime('%f')), d_elem[0], d_elem[1], d_elem[2], 0, 0, 0, 0])
		outfile.write_record(datarec)


#####################################################################################################
# Define utils specific to Embree3 implementation
# (Provided by Sam Potter)

def Embree3_init_geometry(mesh_obj):

	"""
	Perform initial task for geometry initialization for the Embree3 kernel
	Parameters:
	mesh_obj: the mesh object provided via trimesh

	Returns:
	scene: embree.Scene object
	context: embree.Context object
	V:  mesh vertices
	F: mesh faces
	"""
	V = np.array(mesh_obj.vertices, dtype=np.float64)
	F = np.array(mesh_obj.faces, dtype=np.int64)

	P = get_centroids(V, F)
	N, A = get_surface_normals_and_face_areas(V, F)

	device = embree.Device()
	scene = TrimeshShapeModel(V,F,N=N, A=A).get_scene()
	
		
	context = embree.IntersectContext()

	return scene, context, V, F

def Embree3_init_rayhit(ray_origins, ray_directions):
	"""
	Initialize the rayhit object of Embree3

	Parameters:
	ray_origins: [np.array (N,3)]  ray_origins
	ray_directions: [np.array (N,3)]  ray_directions

	Returns: 
	rayhit: the initialized embree.rayhit object

	"""
	nb = np.shape(ray_origins)[0] # Number of tracked rays
	rayhit = embree.RayHit1M(nb)

	# Initialize the ray structure
	#rayhit.tnear[:] = 0.001 #Avoid numerical problems
	rayhit.tnear[:] = 0.00 #Avoid numerical problems
	rayhit.tfar[:] = np.inf
	rayhit.prim_id[:] = embree.INVALID_GEOMETRY_ID
	rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
	rayhit.org[:] = ray_origins
	rayhit.dir[:] = ray_directions

	return rayhit

def Embree3_dump_solution(rayhit, V, F):
	"""
	Process the output of the embree ray intersector kernel

	Parameters: 
	rayhit: embree.rayhit object
	V: vertices
	F: faces

	Returns:
	hits: indexes of hit faces
	nhits: number of hits
	idh: indexez of hitting rays
	Ph: hit points on the mesh

	"""
	ishit=rayhit.prim_id!=embree.INVALID_GEOMETRY_ID
	idh=np.nonzero(ishit)[0]
	hits=rayhit.prim_id[idh]
	nhits=hits.size

	if nhits>0:
		p = V[F[hits]]
		v1=p[:,0]
		v2=p[:,1]
		v3=p[:,2]
		u=rayhit.uv[idh,0]
		v=rayhit.uv[idh,1]
		Ph = v1 + (v2-v1)*u[:,None] + (v3-v1)*v[:,None]

		return hits, nhits, idh, Ph

	else:
		return -1, -1, -1, -1
		



def get_centroids(V, F):
    return V[F].mean(axis=1)
 
def get_cross_products(V, F):
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C
 
 
def get_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    A = C_norms/2
    return A
 
 
def get_surface_normals(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    return N

def get_surface_normals_and_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A



class ShapeModel(ABC):
    pass


class TrimeshShapeModel(ShapeModel):
    """A shape model consisting of a single triangle mesh."""

    def __init__(self, V, F, N=None, P=None, A=None):
        """Initialize a triangle mesh shape model. No assumption is made about
        the way vertices or faces are stored when building the shape
        model except that V[F] yields the faces of the mesh. Vertices
        may be repeated or not.
        Parameters
        ----------
        V : array_like
            An array with shape (num_verts, 3) whose rows correspond to the
            vertices of the triangle mesh
        F : array_like
            An array with shape (num_faces, 3) whose rows index the faces
            of the triangle mesh (i.e., V[F] returns an array with shape
            (num_faces, 3, 3) such that V[F][i] is a 3x3 matrix whose rows
            are the vertices of the ith face.
        N : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            mesh face normals. Can be passed to specify the face normals.
            Otherwise, the face normals will be computed from the cross products
            of the face edges (i.e. np.cross(vi1 - vi0, vi2 - vi0) normalized).
        P : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            centroids. Can be optionally passed to avoid recomputing.
        A : array_like, optional
            An array of shape (num_faces,) containing the triangle areas. Can
            be optionally passed to avoid recomputing.
        """

        self.dtype = V.dtype

        self.V = V
        self.F = F

        if N is None and A is None:
            N, A = get_surface_normals_and_face_areas(V, F)
        elif A is None:
            if N.shape[0] != F.shape[0]:
                raise Exception(
                    'must pass same number of surface normals as faces (got ' +
                    '%d faces and %d normals' % (F.shape[0], N.shape[0])
                )
            A = get_face_areas(V, F)
        elif N is None:
            N = get_surface_normals(V, F)

        self.P = get_centroids(V, F)
        self.N = N
        self.A = A

        assert self.P.dtype == self.dtype
        assert self.N.dtype == self.dtype
        assert self.A.dtype == self.dtype

        self._make_scene()
        

    def _make_scene(self):
        '''Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.
        '''
        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        scene = device.make_scene()
	
        
        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex, # buf_type
            0, # slot
            embree.Format.Float3, # fmt
            3*np.dtype('float32').itemsize, # byte_stride
            self.V.shape[0], # item_count
        )
        vertex_buffer[:] = self.V[:]
        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index, # buf_type
            0, # slot
            embree.Format.Uint3, # fmt
            3*np.dtype('uint32').itemsize, # byte_stride,
            self.F.shape[0]
        )
        index_buffer[:] = self.F[:]
        geometry.commit()
        scene.attach_geometry(geometry)
        geometry.release()
        scene.commit()

        # This is the only variable we need to retain a reference to
        # (I think)
        self.scene = scene
        self.device = device
    def __reduce__(self):
        return (self.__class__, (self.V, self.F, self.N, self.P, self.A))

    @property
    def num_faces(self):
        return self.P.shape[0]


    def get_scene(self):
    	return self.scene




# Main definition of the kernel wrapper
#-----------------------------------------------------------------------------------------------------#

def RTXkernel(mesh_obj, ray_origins, ray_directions, bounces = 1,  kernel = 'Embree', diffusion = False, num_diffuse = None, errorMsg = True):
	"""
	Wrapper for trimesh RTX kernel
	Parameters:
	mesh_obj: Mesh (or geometry) object 
	ray_origins: numpy array of ray origins (n, 3)
	ray_directions: numpy array of ray directions (does not need to be normalized) (n,3)
	bounces: (int) number of bounces to compute
	kernel: one of Embree, Native or Embree3. To chose wether to use the Intel Embree kernel or the native python kernel
	diffusion: (bool) Boolean flag to activate diffused raytracing for the first bounce
	num_diffuse: (int) number of samples for first-bounce diffuse computation
	errorMsg: (bool) flag to control wether to pring the warning when no bounces are found

	Returns:
	index_tri:  Mesh triangle indexes
	index_ray:  Ray indexes
	location:   Location of intersect points
	"""


	ray_origins_container = []
	ray_directions_container = []
	locations_container = []
	index_tri_container = []
	index_ray_container = []


	# Set variables for diffusion computation
	diffusion_directions = 0
	diffusion_pack = []
	diffusion_control = False


	# Select the kernel
	if kernel in ['Embree', 'Native']:
		for i in range(bounces):

			ray_origins_container.append(ray_origins)


			if kernel == 'Embree':
				intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_obj)

			elif kernel == 'Native':
				intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_obj)


			# Avoid numerical problems
			ray_origins = ray_origins + 1e-3*ray_directions 


			# If computing bounce number 1 and the diffusion computation has been requested
			# do a separate raytracing also for the diffused rays
			# and pack results in the variable: diffusion pack
			if i == 1 and diffusion:
			
				ray_origins_diffusion = np.repeat(ray_origins, num_diffuse, axis = 0) # this should be correct
				ray_directions_diffusion = diffuse_directions

				index_tri_diffusion, index_ray_diffusion, location_diffusion = intersector.intersects_id( ray_origins = ray_origins_diffusion,
															  ray_directions = ray_directions_diffusion,
															  multiple_hits = False,
															  return_locations = True)
				diffusion_pack = [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,  location_diffusion]



			# Main Raytracer
			index_tri, index_ray, location = intersector.intersects_id(
														ray_origins = ray_origins,
														ray_directions = ray_directions,
														multiple_hits = False,
														return_locations = True)
			# Get the number of hits
			n_hits = len(index_tri)

			# Manage the possibility of no hits
			if n_hits == 0 and errorMsg:
				print ('No intersections found for bounce {}. Results provided up to bounce {}'.format(i+1, i))
				break
			else:
				locations_container.append(location)
				index_tri_container.append(index_tri)
				index_ray_container.append(index_ray)
				ray_directions_container.append(ray_directions)
			


				if i != bounces -1:
					# If at bounce number 1 compute the diffused directions:
					if diffusion and i == 0:
						diffusion_control = True
	
					ray_origins, ray_directions, diffuse_directions = compute_secondary_bounce(location, index_tri, mesh_obj, ray_directions, index_ray, diffusion = diffusion_control, num_diffuse = num_diffuse)

					# Set back to false the diffusion computation control flag
					diffusion_control = False
					



	elif kernel == 'Embree3':
		
		# Initialize the geometry
		scene, context, V, F = Embree3_init_geometry(mesh_obj)

		for i in range(bounces):
			ray_origins_container.append(ray_origins)

			# Initialize the rayhit object
			ray_origins = ray_origins +1e-3*ray_directions
			rayhit = Embree3_init_rayhit(ray_origins, ray_directions)

			# Run the intersector
			scene.intersect1M(context, rayhit)

			# Post-process the results
			index_tri, n_hits, index_ray, location = Embree3_dump_solution(rayhit, V, F)

			#embree.Device().release()
			# Handle: not bounces found
			if n_hits == -1:
				print ('No intersections found for bounce {}. Results provided up to bounce {}'.format(i+1, i))
				break


			# Otherwise append results and proceed with next bounce
			else: 
				locations_container.append(location)
				index_tri_container.append(index_tri)
				index_ray_container.append(index_ray)
				ray_directions_container.append(ray_directions) 

				if i != bounces-1:
					ray_origins, ray_directions = compute_secondary_bounce(location, index_tri, mesh_obj, ray_directions, index_ray)





	else:
		print('No Recognized kernel')




	# Manage output variables
	if diffusion:
		return index_tri_container, index_ray_container, locations_container, ray_origins_container, ray_directions_container, diffusion_pack
	else:
		return index_tri_container, index_ray_container, locations_container, ray_origins_container, ray_directions_container, None
		












if __name__ == "__main__":
	d0 = 2
	lat = 90*np.pi/180
	lon = 45*np.pi/180
	locs, dirs = pixel_plane(d0, lon, lat, width = 1, height = 1, ray_spacing = .5)
	


	import matplotlib.pyplot as plt 
	from mpl_toolkits import mplot3d
	ax = plt.axes(projection = '3d')
	ax.scatter3D(locs[:,0], locs[:,1], locs[:,2])
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	ax.quiver(0, 0, 0, 1, 0, 0, 
 arrow_length_ratio=0.1, color = 'r')
	ax.quiver(0, 0, 0, 0, 1, 0, 
 arrow_length_ratio=0.1, color = 'g')
	ax.quiver(0, 0, 0, 0, 0, 1, 
 arrow_length_ratio=0.1, color = 'b')
	plt.show()
