import numpy as np
from pyRTX.core.utils_rt import fast_vector_build
import spiceypy as sp
from pyRTX import constants
import trimesh as tm
import shapely 

class PixelPlane():


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
			self.width, self.height = self.compute_hw() # TO CHECK


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

			lon = self.lon
			lat = self.lat
			d0 = self.d0
			packets = self.packets
			ray_spacing = self.ray_spacing


			# Build the direction vector
			x0 = np.array([-d0*np.cos(lon)*np.cos(lat), -d0*np.sin(lon)*np.cos(lat), -d0*np.sin(lat)])
			self.x0 = x0
			x0_unit = x0/np.linalg.norm(x0)

			# Build the pixel matrix

			# Projecting polygon in the direction of rays
			polygonProjected=tm.path.polygons.projected(self.spacecraft.dump(),x0_unit,x0)
			print(self.spacecraft.dump())
			# Finding rectangular area around the polygon 
			minx, miny, maxx, maxy = polygonProjected.bounds
			nx = int((maxx - minx)/ray_spacing)
			ny = int((maxy - miny)/ray_spacing)
			minx += ray_spacing/2
			miny += ray_spacing/2
			maxx -= ray_spacing/2
			maxy -= ray_spacing/2
			gx, gy = np.linspace(minx,maxx,nx), np.linspace(miny,maxy,ny)
			firstTrue = True
			# Finding equidistant points inside the shape, excluding the border 
			for i in range(len(gx)):
				for j in range(len(gy)):
					point = np.array([gx[i],gy[j]])
					if  shapely.geometry.Point(point).within(polygonProjected):
						if firstTrue == True:
							basic_coords = np.array( np.hstack((0,point)))
						else:
							basic_coords = np.vstack( (basic_coords, np.hstack((0,point))) )
						firstTrue = False
			
			# Return the output in the shape required by trimesh
			basic_coords -= x0
			self.basic_coords0 = basic_coords
			basic_dirs = np.full(basic_coords.shape, x0_unit)
		else:			
			basic_coords = self.basic_coords0
			basic_coords -= self.x0
			x0_unit = self.x0/np.linalg.norm(self.x0)
			basic_dirs = np.full(basic_coords.shape, x0_unit)

		if not self.packets == 1:
			basic_coords = np.array_split(basic_coords, self.packets)
			basic_dirs = np.array_split(basic_dirs, self.packets)

		return basic_coords, basic_dirs




	def update_latlon(self,lon = None, lat = None):
		self.lon = lon
		self.lat = lat


	def compute_hw(self):
		lon = self.lon
		lat = self.lat
		d0 = self.d0
		# Build the direction vector
		x0 = np.array([-d0*np.cos(lon)*np.cos(lat), -d0*np.sin(lon)*np.cos(lat), -d0*np.sin(lat)])
		self.x0 = x0
		x0_unit = x0/np.linalg.norm(x0)
		# Projecting polygon in the direction of rays
		print(self.spacecraft.dump())
		polygonProjected=tm.path.polygons.projected(self.spacecraft.dump(),x0_unit,x0)
		minx, miny, maxx, maxy = polygonProjected.bounds
		width = np.amax(np.abs([minx, maxx]))*2
		height = np.amax(np.abs([miny, maxy]))*2
		return width, height
