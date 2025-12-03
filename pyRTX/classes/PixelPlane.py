import numpy as np
import spiceypy as sp

from pyRTX import constants
from pyRTX.core.utils_rt import fast_vector_build

class PixelPlane():
	"""
	A class to generate a rectangular grid of rays for ray-tracing.

	This class creates a planar grid of ray origins and directions, simulating a
	uniform, parallel light source. It can be configured in a fixed orientation
	or dynamically aligned with a celestial body (e.g., the Sun).
	"""


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
		"""
        Initializes the PixelPlane object.

        Parameters
        ----------
        spacecraft : object, optional
            The spacecraft object, used for dynamic positioning.
        source : str, optional
            The name of the celestial body to track (e.g., 'Sun').
        mode : str, default='Fixed'
            The operational mode ('Fixed' or 'Dynamic').
        distance : float, default=1.0
            The distance of the plane from the origin.
        lon : float, optional
            The longitude of the plane's center in a fixed orientation.
        lat : float, optional
            The latitude of the plane's center in a fixed orientation.
        width : float, optional
            The width of the ray grid.
        height : float, optional
            The height of the ray grid.
        ray_spacing : float, default=0.1
            The spacing between adjacent rays.
        packets : int, default=1
            The number of packets to divide the rays into for processing.
        units : str, default='m'
            The units for all dimensional parameters.
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

		self.sp_data = None


	def dump(self, epoch = None):
		"""
		Generate the ray origins and directions for a given epoch.

		Parameters
		----------
		epoch : float, optional
		    The SPICE ephemeris time for dynamic positioning.

		Returns
		-------
		tuple
		    A tuple containing the ray origins and directions as numpy arrays.
		"""

		if self.mode == 'Fixed':

			basic_coords, basic_dirs = self._core_dump()

		elif self.mode == 'Dynamic':

			if isinstance(epoch, str):
				et = sp.str2et( epoch )
			else:
				et = epoch

			if self.sp_data != None:
				sourcedir = self.sp_data.getPosition(epoch, self.spacecraft.name, self.source, self.spacecraft.base_frame, 'LT+S')
			else:
				sourcedir = sp.spkezr(self.source, epoch, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name )
				sourcedir = sourcedir[0][0:3]
    
			sourcedir = np.array(sourcedir)/np.linalg.norm(sourcedir)
			_, self.lon, self.lat = sp.recrad(sourcedir)
			
			basic_coords, basic_dirs = self._core_dump()

		return basic_coords, basic_dirs


	def _core_dump(self, instantiate = False):
		"""
		Core logic for generating the pixel plane coordinates and directions.

		Parameters
		----------
		instantiate : bool, default=False
		    If True, initializes the base coordinates; otherwise, applies transformations.

		Returns
		-------
		tuple
		    A tuple containing the ray origins and directions as numpy arrays.
		"""

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
		"""
		Update the latitude and longitude for a fixed orientation.

		Parameters
		----------
		lon : float, optional
		    The new longitude value.
		lat : float, optional
		    The new latitude value.
		"""
		self.lon = lon
		self.lat = lat
