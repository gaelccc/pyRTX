# Utilities for data/output analysis
import numpy as np
from scipy import interpolate
import spiceypy as sp
from PIL import Image
from pyRTX.core.utils_rt import block_normalize
from numba import jit
from numpy import ceil

"""
A set of utilities mainly focused on result analysis and data manipulation
"""






class LookupTable():
	"""
	This class is used to store results in the shape aof a lookup table.
	This is mainly used to store the resultas of a set of raytracing results
	example: the solar pressure for a body is computed for a grid of RA/DEC values.
	these values can be stored in the LookupTable object and later retrieved.
	This class offers the possibility of not oly retrieving pre-computed values, but
	aslso interpolating between grid points.

	NOTE: the grid of the lookup table does not need to be regular
	the interpolation is based on numpy griddata method which is able to cope
	with unstructured grids

	The main way of retrieving values is through indexing. The following are implemented:

	LUT[a,b]: if a, b are in the original lookup table, the original values are returned, otherwise they are interpolated
	LUT[:,:] or LUT[a:b, c:d]: return the original lut sliced as requested
	LUT[:,a]: return the original lut (all elements of first axis, integer-indexed elements of second axis)
	LUT[array-like, array-like]: return the lookup table interpolated in the array-like points



	Parameters
	----------
	linspace_x : np.array(N,)
		The x axis of the lookup table
	linspace_y : np.array(M,)
		The y axis of the lookup table
	values : np.ndarray (N,M,1)
		The lookup table values


	"""
	def __init__(self, linspace_x, linspace_y, values):
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.values = values

		self._set_defaults()


	def _set_defaults(self):
		self.interpType = 'cubic'

	def _interpolator(self, x, y):

		x, y = np.meshgrid(x,y)
		meshgrid_x, meshgrid_y = np.meshgrid(self.linspace_x, self.linspace_y)
		#return  interpolate.griddata((meshgrid_x.ravel(), meshgrid_y.ravel()), self.values.T.ravel(),np.array([x,y]).T, method = self.interpType)
		return  interpolate.griddata((meshgrid_x.ravel(), meshgrid_y.ravel()), self.values.T.ravel(),(x,y), method = self.interpType)

	def set_interpType(self, interpType):
		self.interpType = interpType


	def _get_idx(self, ind, search_list):
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, x, y):
		return self._interpolator(x,y)


	def __getitem__(self, idxs):
		"""
		Implement a getitem method.
		Several usages are possible:

		LUT[a,b]: if a, b are in the original lookup table, the original values are returned, otherwise they are interpolated
		LUT[:,:] or LUT[a:b, c:d]: return the original lut sliced as requested
		LUT[:,a]: return the original lut (all elements of first axis, integer-indexed elements of second axis)
		LUT[array-like, array-like]: return the lookup table interpolated in the array-like points

		"""

		x, y = idxs
		if isinstance(x, slice) or isinstance(y, slice):
			return self.values[x, y]
		else:
			return self.interp_point(x,y)
			



	def quickPlot(self, xlabel = None, ylabel = None, title = None, conversion = 1, clabel = None, cmap = 'viridis', saveto = None):

		"""
		Produce a quick plot of the lookup table

		Parameters
		----------
		xlabel : str (Optional)
			label for the x-axis
		ylabel : str (Optional)
			label for the y-axis
		title : str (Optional)
			title of the plot
		conversion : float (Optional, default 1)
			a conversion factor for the plotted values. This method will plot X*conversion, Y*conversion
		clabel : str (Optional)
			label of the color bar
		cmap : str	(Optional, default 'viridis')
			the colormap to use (matplotlib colormaps)
		saveto : str (Optional, default None)
			if not None: the path to save the plot to (the file extension defines the format)

		"""


		import matplotlib.pyplot as plt

		X, Y = np.meshgrid(self.linspace_x, self.linspace_y)
		fig, ax = plt.subplots()
		h = ax.contourf(X*conversion,Y*conversion, self.interp_point(self.linspace_x, self.linspace_y), cmap = cmap)
		c = plt.colorbar(h)
		c.set_label(clabel)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_title(title)

		if saveto is not None:
			plt.savefig(saveto, dpi = 800)

		plt.show()



class LookupTableND():
	"""
	Same concept as the LookupTable class, but allowing for multi-dimensional (>2) tables

	Parameters
	----------
	axes : tuple of np.array
		The axes of the lookup table
	values : np.ndarray(N,M,L,...,1)
	info : str
		A string field to store information about the lookup table. This is set as a class property
		so it can be requested trhough instance.info
	np.array : np.array
		unset

	"""



	def __init__(self, axes = None, values = None, info = None, np_array = None):
		self.axes = axes
		self.values = values
		self.info = info
		self.np_array = np_array


		self._set_defaults()


	def _set_defaults(self):
		self.interpType = 'linear'

	def _interpolator(self, vals):
		
		return interpolate.interpn(self.axes,self.values, vals, method = 'linear') 


	def set_interpType(self, interpType):
		self.interpType = interpType


	def get_idx(self, ind, search_list):
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, vals):
		return self._interpolator(vals)


	def __getitem__(self, idxs):
		"""
		Implement a getitem method.
		Several usages are possible:

		LUT[a,b]: if a, b are in the original lookup table, the original values are returned, otherwise they are interpolated
		LUT[:,:] or LUT[a:b, c:d]: return the original lut sliced as requested
		LUT[:,a]: return the original lut (all elements of first axis, integer-indexed elements of second axis)
		LUT[array-like, array-like]: return the lookup table interpolated in the array-like points

		"""

		
		if np.any([isinstance(x, slice) for x in idxs]) :
			return self.values[idxs]
		else:
			
			outval =  self.interp_point(idxs)
			return np.squeeze(outval)
			

	def axisExtent(self):
		extent = []
		for ax in self.axes:
			extent.append([np.min(ax), np.max(ax)])
		return extent


	def quickPlot(self, xlabel = None, ylabel = None, title = None, conversion = 1, clabel = None, cmap = 'viridis', saveto = None):
		"""
		Produce a quick plot of the lookup table

		Parameters
		----------
		xlabel : str (Optional)
			label for the x-axis
		ylabel : str (Optional)
			label for the y-axis
		title : str (Optional)
			title of the plot
		conversion : float (Optional, default 1)
			a conversion factor for the plotted values. This method will plot X*conversion, Y*conversion
		clabel : str (Optional)
			label of the color bar
		cmap : str	(Optional, default 'viridis')
			the colormap to use (matplotlib colormaps)
		saveto : str (Optional, default None)
			if not None: the path to save the plot to (the file extension defines the format)

		"""


		import matplotlib.pyplot as plt

		X, Y = np.meshgrid(*self.axes)
		fig, ax = plt.subplots()
		h = ax.contourf(X*conversion,Y*conversion, self.interp_point(self.axes), cmap = cmap)
		c = plt.colorbar(h)
		c.set_label(clabel)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.set_title(title)

		if saveto is not None:
			plt.savefig(saveto, dpi = 800)

		plt.show()



class ScatterLookup():
	"""
	A class for dealing with zone-scattered lookup tables
	Intended for creating a lookup table of sets of computed values that lie in distinct, DISJUNCT,
	zones of the axes space.
	Example: the value of a variable has been computed in X = [0,1] Y = [0,1] and X = [3,4] Y = [-2,-1]

	After instantiating the empty class, the different zones are added. Example

	sc = ScatterLookup()
	zone1 = LookupTableND(*args, **kwargs)
	zone2 = LookupTableND(*args, **kwargs)
	sc.addZone(zone1)
	sc.addZone(zone2)



	The value retrieval follows the same rules of indexing as the LookupTable and LookupTableND classes
	"""
	def __init__(self):
		self.zones = []
		self.zonedef = []

	def add_zone(self, ZoneLookup = ''):
		"""
		Add a zone
		Parameters
		----------
		ZoneLookup : pyRTX.core.analysis_utils.LookupTableND

		"""
		if not isinstance(ZoneLookup, LookupTableND):
			raise TypeError('The ZoneLookup argument must be of type class.LookupTableND')

		self.zones.append(ZoneLookup)
		self.zonedef.append(ZoneLookup.axisExtent())

	def __getitem__(self, idxs):

		zone_no = self.zone_determination(idxs)
		return self.zones[zone_no][idxs]
		


	def zone_determination(self,idxs):
		flag = 1
		for i, zonedef in enumerate(self.zonedef):
			#print(f'Zone {zonedef}')  FOR DEBYG
			for j, idx in enumerate(idxs):
				if not zonedef[j][0]<=idx<=zonedef[j][1]:
					flag = 0
					print(idx, zonedef[j][0], zonedef[j][1])
			if flag == 1:
				return i 
			elif i < len(self.zonedef)-1:
				flag = 1
		if flag == 0:
			raise Exception('Interpolation Error: No data correspond to your request')




class TiffInterpolator():
	def __init__(self, axes = None, values = None):
		self.axes = axes
		self.values = values
		


		# Error control
		#if len(self.axes) != len(values.shape[0:-1]):
		#	raise ValueError(f'The number of axes provided does not match with provided data\n Provided data has shape {len(values.shape)} while {len(self.axes)} axes have been provided')



		self._set_defaults()


	def _set_defaults(self):
		self.interpType = 'linear'

	def interpolator(self, vals):
		
		return interpolate.interpn(self.axes,self.values.T, vals) 


	def set_interpType(self, interpType):
		self.interpType = interpType


	def get_idx(self, ind, search_list):
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, vals):
		return self.interpolator(vals)


	def __getitem__(self, idxs):
		"""
		Implement a getitem method.
		Several usages are possible:

		LUT[a,b]: if a, b are in the original lookup table, the original values are returned, otherwise they are interpolated
		LUT[:,:] or LUT[a:b, c:d]: return the original lut sliced as requested
		LUT[:,a]: return the original lut (all elements of first axis, integer-indexed elements of second axis)
		LUT[array-like, array-like]: return the lookup table interpolated in the array-like points

		"""

		
		if np.any([isinstance(x, slice) for x in idxs]) :
			return self.values[idxs]
		else:
			
			outval =  self.interp_point(idxs)
			return np.squeeze(outval)



def getSunAngles(scName = None, scFrame = None, epoch = None, correction = 'LT+S'):
	sunPos = sp.spkezr('Sun', epoch, scFrame, correction, scName)[0][0:3]
	[_, ra, dec] = sp.recrad(sunPos)

	return ra, dec

	

def epochRange(startEpoch = None, duration = None, step = 100):
	if isinstance(startEpoch, str):
		startEp = sp.str2et(startEpoch)
	elif isinstance(startEpoch, float):
		startEp = startEpoch
	else:
		raise ValueError('startEpoch argument must be str or float')
	
	endEp = startEp + duration

	epochList = np.linspace(startEp, endEp, num = int(np.ceil((endEp - startEp)/step)), endpoint = False)
	return epochList


def epochRange2(startEpoch = None, endEpoch = None, step = 100):
	if isinstance(startEpoch, str):
		startEp = sp.str2et(startEpoch)
	elif isinstance(startEpoch, float):
		startEp = startEpoch
	else:
		raise ValueError('startEpoch argument must be str or float')


	if isinstance(endEpoch, str):
		endEp = sp.str2et(endEpoch)
	elif isinstance(endEpoch, float):
		endEp = endEpoch
	else:
		raise ValueError('endEpoch argument must be str or float')
	
	curr = startEp

	epochlist = []
	while curr <= endEp:
		epochlist.append(curr)
		curr += step
	return np.array(epochlist)





def computeRADEC(vecs, periodicity = 360):
	"""
	From a ND array of shape (N, 3), compute RA/DEC

	Parameters:
	vecs: [ndarray (N,3)]
	Returns:
	RA, DEC
	"""

	n = block_normalize(vecs)
	dec = np.arcsin(n[:,2])
	sinB = n[:,1]/np.cos(dec)
	cosB = n[:,0]/np.cos(dec)
	ra = np.arctan2(sinB, cosB)

	if periodicity == 360:
		ra = np.where(ra < 0, 2*np.pi + ra, ra)



	return ra, dec





def convertTIFtoMesh(tifFile = '', latSampling = '', inferSampling = False, lonSampling = '', planet = '', lat0 = -np.pi/2, lat1 = np.pi/2, lon0 = 0, lon1 = 2*np.pi):
	"""
	Convert a TIF map of emissivities to the format needed for assigning values to each face of a planetary 3D mesh

	Parameters:
	tifFile : [str] the path to the TIFF file
	latSampling: [float, in rad] sampling step of latitude
	lonSampling: [float, in rad] sampling step of longitude
	planet: [class.Planet] Planet object containing the mesh and the planetary frames
	inferSampling: [bool] wether the importer shall infer the sampling or not
	Returns:
	emissivityInterpolator: [class.LookupTableND] an interpolator for mapping the temperatures on the mesh faces. This is intended to be passed
				to the Planet class via the setter method: Planet.emissivity = emissivityInterpolator

	"""


	im = Image.open(tifFile)
	
	#arr = np.expand_dims(np.array(im), axis = 2)
	arr = np.array(im)

	#import matplotlib.pyplot as plt
	#plt.figure()
	#print(np.array(im).shape)
	#plt.contourf(np.array(im))

	if inferSampling:
		lonSampling = (lon1-lon0)/arr.shape[1]
		latSampling = (lat1-lat0)/arr.shape[0]

	lats = np.linspace(lat0, lat1, int(ceil((lat1 - lat0) / latSampling )))
	lons = np.linspace(lon0, lon1, int(ceil((lon1 - lon0) / lonSampling )))

	lut = TiffInterpolator(axes = (lons, lats), values = arr)


	return lut




	'''
	newlats = np.linspace(lat0, lat1, 1000)
	newlons = np.linspace(lon0, lon1, 1000)
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(lats,arr[0,:,0], marker = 'o', label = 'orig')
	dd = lut[0, newlats]
	plt.plot(newlats,dd, marker = '.', label = 'interp')
	plt.legend()
	plt.show()
	'''



def convertEpoch(monteEpoch):
	'''
	Convert a Monte epoch string to a spice epoch string

	'''


def get_spacecraft_area(spacecraft, ra = 0.0, dec = 0.0, epoch = None, ray_spacing = 0.1):

	'''
	Compute a pyRTX.Spacecraft apparent area as seen by the direction specified 
	by a pair of right ascension - declination

	Input:
	spacecraft [pyRTX.Spacecraft] 	: the spacecraft object
	ra [float] 						: right ascension (in rad)
	dec [float] 					: declination (rad)
	epoch [float or None]			: epoch for the computation (this is used when moving Spice
										frames are used for the Spacecraft definition)

	Output:
	area [float] 					: the apparent area. The measurement units depend on the units of the
				   						Spacecraft object

	TODO: avoid hardcoded width/height but rather use an automated method

	'''
	from pyRTX.classes.PixelPlane import PixelPlane
	from pyRTX.classes.RayTracer  import RayTracer
	rays = PixelPlane( 
			spacecraft = spacecraft,   # Spacecraft object 
			mode = 'Fixed',   # Mode: can be 'Dynamic' ( The sun orientation is computed from the kernels), or 'Fixed'
			distance = 100,	    # Distance of the ray origin from the spacecraft
			lon = ra,
			lat = dec, 
			ray_spacing = ray_spacing, # Ray spacing (in m)
		)

	rtx = RayTracer(        
                        spacecraft,                    # Spacecraft object
                        rays,                   # pixelPlane object
                        kernel = 'Embree',      # The RTX kernel to use
                        bounces = 1,            # The number of bounces to account for
                        diffusion = False,       # Account for secondary diffusion
                        ) 

	rtx.trace(epoch)
	hit_rays = rtx.index_ray_container
	Area =  len(hit_rays[0])/rays.norm_factor
	return Area
