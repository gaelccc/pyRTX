import numpy as np
from pyRTX.utils_rt import fast_vector_build
import spiceypy as sp
from pyRTX import constants

class pixelPlane():


	def __init__(self, spacecraft = None, 
			   source = None,
			   mode = 'Fixed', 
			   distance = 1.0, 
			   lon = None, 
			   lat = None, 
			   width = None, 
			   height = None, 
			   ray_spacing = .1, 
			   packets = 1,
			   units = 'm'):


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

		conversion_factor = constants.unit_conversions[units]

		self.mode = mode
		self.spacecraft = spacecraft
		self.d0 = distance * conversion_factor
		self.lon = lon
		self.lat = lat
		self.packets = packets
		self.ray_spacing = ray_spacing * conversion_factor
		self.norm_factor = 1.0/self.ray_spacing**2  # THIS HAS BEEN MOD
		self.source = source

		if width is not None and height is not None:
			self.width = width*conversion_factor
			self.height = height*conversion_factor

		else:
			width, height = self.compute_hw() # TODO
		
		
		# Instantiate the ray data at __init__ to gain time. Unless when instantiating the object for shadow computation
		self._core_dump(instantiate = True)


		


	def dump(self, epoch = None):

		if self.mode == 'Fixed':

			basic_coords, basic_dirs = self._core_dump()

		elif self.mode == 'Dynamic':
			if isinstance(epoch, str):
				et = sp.str2et( epoch )
			else:
				et = epoch

			sourcedir = sp.spkezr(self.source, epoch, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name )
			sourcedir = sourcedir[0][0:3]
			sourcedir = np.array(sourcedir)/np.linalg.norm(sourcedir)
			_, self.lon, self.lat = sp.recrad(sourcedir)
			
			basic_coords, basic_dirs = self._core_dump()

		return basic_coords, basic_dirs



	def _core_dump(self, instantiate = False):


		if instantiate:
			w2 = self.width/2
			h2 = self.height/2
			lon = 0
			lat = 0
			d0 = self.d0
			packets = self.packets
			ray_spacing = self.ray_spacing




			# Build the direction vector
			x0 = np.array([-d0*np.cos(lon)*np.cos(lat), -d0*np.sin(lon)*np.cos(lat), -d0*np.sin(lat)])
			self.x0 = x0
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

			dim1 = int(self.width/ray_spacing)+1
			dim2 = int(self.height/ray_spacing)+1
			basic_coords = np.zeros((dim1*dim2, 3))
			basic_dirs = np.full(basic_coords.shape, x0_unit)

			linsp1 = np.linspace(-w2, w2, num = dim1 )
			linsp2 = np.linspace(-h2, h2, num = dim2)

			
			basic_coords = fast_vector_build(linsp1, linsp2, dim1, dim2)

			self.basic_coords0 = basic_coords
			
			basic_coords = np.dot(np.array(basic_coords), R.T)
			basic_coords -= x0


			# Return the output in the shape required by trimesh

		else:
			w2 = self.width/2
			h2 = self.height/2
			lon = self.lon
			lat = self.lat
			d0 = self.d0
			packets = self.packets
			ray_spacing = self.ray_spacing
			x0 = np.array([-d0*np.cos(lon)*np.cos(lat), -d0*np.sin(lon)*np.cos(lat), -d0*np.sin(lat)])
			self.x0 = x0
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
			

			basic_coords = np.dot(np.array(self.basic_coords0), R.T)
			basic_coords -= x0
			basic_dirs = np.full(basic_coords.shape, x0_unit)

		if not packets == 1:
			basic_coords = np.array_split(basic_coords, packets)
			basic_dirs = np.array_split(basic_dirs, packets)

		return basic_coords, basic_dirs




	def update_latlon(self,lon = None, lat = None):
		self.lon = lon
		self.lat = lat


