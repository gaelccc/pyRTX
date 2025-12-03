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
    A class for storing and interpolating 2D lookup tables.

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
		"""
        Initializes the LookupTable object.

        Parameters
        ----------
        linspace_x : numpy.ndarray
            The x-coordinates of the grid.
        linspace_y : numpy.ndarray
            The y-coordinates of the grid.
        values : numpy.ndarray
            The values at the grid points.
		"""
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.values = values

		self._set_defaults()


	def _set_defaults(self):
		"""
        Sets the default interpolation type to 'cubic'.
		"""
		self.interpType = 'cubic'

	def _interpolator(self, x, y):
		"""
        Performs the interpolation.

        Parameters
        ----------
        x : numpy.ndarray
            The x-coordinates at which to interpolate.
        y : numpy.ndarray
            The y-coordinates at which to interpolate.

        Returns
        -------
        numpy.ndarray
            The interpolated values.
		"""

		x, y = np.meshgrid(x,y)
		meshgrid_x, meshgrid_y = np.meshgrid(self.linspace_x, self.linspace_y)
		#return  interpolate.griddata((meshgrid_x.ravel(), meshgrid_y.ravel()), self.values.T.ravel(),np.array([x,y]).T, method = self.interpType)
		return  interpolate.griddata((meshgrid_x.ravel(), meshgrid_y.ravel()), self.values.T.ravel(),(x,y), method = self.interpType)

	def set_interpType(self, interpType):
		"""
        Sets the interpolation type.

        Parameters
        ----------
        interpType : str
            The interpolation method to use (e.g., 'linear', 'cubic').
		"""
		self.interpType = interpType


	def _get_idx(self, ind, search_list):
		"""
        Gets the index of a value in a list.

        Parameters
        ----------
        ind : float
            The value to search for.
        search_list : list or numpy.ndarray
            The list to search in.

        Returns
        -------
        int
            The index of the value.
		"""
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, x, y):
		"""
        Interpolates the lookup table at a single point.

        Parameters
        ----------
        x : float
            The x-coordinate of the point.
        y : float
            The y-coordinate of the point.

        Returns
        -------
        float
            The interpolated value.
		"""
		return self._interpolator(x,y)


	def __getitem__(self, idxs):
		"""
        Allows indexing into the lookup table.

        Parameters
        ----------
        idxs : tuple
            A tuple of indices for each dimension of the lookup table.

        Returns
        -------
        float or numpy.ndarray
            The interpolated value(s) at the given indices.
		"""

		x, y = idxs
		if isinstance(x, slice) or isinstance(y, slice):
			return self.values[x, y]
		else:
			return self.interp_point(x,y)
			



	def quickPlot(self, xlabel = None, ylabel = None, title = None, conversion = 1, clabel = None, cmap = 'viridis', saveto = None):
		"""
        Produces a quick plot of the lookup table.

        Parameters
        ----------
        xlabel : str, optional
            The label for the x-axis.
        ylabel : str, optional
            The label for the y-axis.
        title : str, optional
            The title of the plot.
        conversion : float, default=1
            A conversion factor for the plotted values.
        clabel : str, optional
            The label for the color bar.
        cmap : str, default='viridis'
            The colormap to use.
        saveto : str, optional
            The path to save the plot to.
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
    A class for storing and interpolating N-dimensional lookup tables.

    Same concept as the LookupTable class, but allowing for multi-dimensional (>2) tables

    Parameters
    ----------
    axes : tuple of np.array
        The axes of the lookup table
    values : np.ndarray(N,M,L,...,1)
    info : str
        A string field to store information about the lookup table. This is set as a class property
        so it can be requested trhough instance.info
    np_array : np.array
        unset
	"""



	def __init__(self, axes = None, values = None, info = None, np_array = None):
		"""
        Initializes the LookupTableND object.

        Parameters
        ----------
        axes : tuple of numpy.ndarray, optional
            The axes of the lookup table.
        values : numpy.ndarray, optional
            The values at the grid points.
        info : str, optional
            Information about the lookup table.
        np_array : numpy.ndarray, optional
            Unused.
		"""
		self.axes = axes
		self.values = values
		self.info = info
		self.np_array = np_array


		self._set_defaults()


	def _set_defaults(self):
		"""
        Sets the default interpolation type to 'linear'.
		"""
		self.interpType = 'linear'

	def _interpolator(self, vals):
		"""
        Performs the interpolation.

        Parameters
        ----------
        vals : tuple
            A tuple of coordinates at which to interpolate.

        Returns
        -------
        numpy.ndarray
            The interpolated values.
		"""
		return interpolate.interpn(self.axes,self.values, vals, method = 'linear') 


	def set_interpType(self, interpType):
		"""
        Sets the interpolation type.

        Parameters
        ----------
        interpType : str
            The interpolation method to use (e.g., 'linear', 'cubic').
		"""
		self.interpType = interpType


	def get_idx(self, ind, search_list):
		"""
        Gets the index of a value in a list.

        Parameters
        ----------
        ind : float
            The value to search for.
        search_list : list or numpy.ndarray
            The list to search in.

        Returns
        -------
        int
            The index of the value.
		"""
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, vals):
		"""
        Interpolates the lookup table at a single point.

        Parameters
        ----------
        vals : tuple
            A tuple of coordinates of the point.

        Returns
        -------
        float
            The interpolated value.
		"""
		return self._interpolator(vals)


	def __getitem__(self, idxs):
		"""
        Allows indexing into the lookup table.

        Parameters
        ----------
        idxs : tuple
            A tuple of indices for each dimension of the lookup table.

        Returns
        -------
        float or numpy.ndarray
            The interpolated value(s) at the given indices.
		"""

		
		if np.any([isinstance(x, slice) for x in idxs]) :
			return self.values[idxs]
		else:
			
			outval =  self.interp_point(idxs)
			return np.squeeze(outval)
			

	def axisExtent(self):
		"""
        Returns the extent of each axis.

        Returns
        -------
        list
            A list of [min, max] pairs for each axis.
		"""
		extent = []
		for ax in self.axes:
			extent.append([np.min(ax), np.max(ax)])
		return extent


	def quickPlot(self, xlabel = None, ylabel = None, title = None, conversion = 1, clabel = None, cmap = 'viridis', saveto = None):
		"""
        Produces a quick plot of a 2D slice of the lookup table.

        Parameters
        ----------
        xlabel : str, optional
            The label for the x-axis.
        ylabel : str, optional
            The label for the y-axis.
        title : str, optional
            The title of the plot.
        conversion : float, default=1
            A conversion factor for the plotted values.
        clabel : str, optional
            The label for the color bar.
        cmap : str, default='viridis'
            The colormap to use.
        saveto : str, optional
            The path to save the plot to.
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
    A class for dealing with zone-scattered lookup tables.

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
		"""
        Initializes the ScatterLookup object.
		"""
		self.zones = []
		self.zonedef = []

	def add_zone(self, ZoneLookup = ''):
		"""
        Adds a zone to the lookup table.

        Parameters
        ----------
        ZoneLookup : pyRTX.core.analysis_utils.LookupTableND
            The lookup table for the new zone.
		"""
		if not isinstance(ZoneLookup, LookupTableND):
			raise TypeError('The ZoneLookup argument must be of type class.LookupTableND')

		self.zones.append(ZoneLookup)
		self.zonedef.append(ZoneLookup.axisExtent())

	def __getitem__(self, idxs):
		"""
        Allows indexing into the lookup table.

        Parameters
        ----------
        idxs : tuple
            A tuple of indices for each dimension of the lookup table.

        Returns
        -------
        float or numpy.ndarray
            The interpolated value(s) at the given indices.
		"""

		zone_no = self.zone_determination(idxs)
		return self.zones[zone_no][idxs]
		


	def zone_determination(self,idxs):
		"""
        Determines which zone the given indices belong to.

        Parameters
        ----------
        idxs : tuple
            A tuple of indices for each dimension of the lookup table.

        Returns
        -------
        int
            The index of the zone.
		"""
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
	"""
    A class for interpolating TIFF image data.
	"""
	def __init__(self, axes = None, values = None):
		"""
        Initializes the TiffInterpolator object.

        Parameters
        ----------
        axes : tuple of numpy.ndarray, optional
            The axes of the TIFF image.
        values : numpy.ndarray, optional
            The pixel values of the TIFF image.
		"""
		self.axes = axes
		self.values = values
		


		# Error control
		#if len(self.axes) != len(values.shape[0:-1]):
		#	raise ValueError(f'The number of axes provided does not match with provided data\n Provided data has shape {len(values.shape)} while {len(self.axes)} have been provided')



		self._set_defaults()


	def _set_defaults(self):
		"""
        Sets the default interpolation type to 'linear'.
		"""
		self.interpType = 'linear'

	def interpolator(self, vals):
		"""
        Performs the interpolation.

        Parameters
        ----------
        vals : tuple
            A tuple of coordinates at which to interpolate.

        Returns
        -------
        numpy.ndarray
            The interpolated values.
		"""
		
		return interpolate.interpn(self.axes,self.values.T, vals) 


	def set_interpType(self, interpType):
		"""
        Sets the interpolation type.

        Parameters
        ----------
        interpType : str
            The interpolation method to use (e.g., 'linear', 'cubic').
		"""
		self.interpType = interpType


	def get_idx(self, ind, search_list):
		"""
        Gets the index of a value in a list.

        Parameters
        ----------
        ind : float
            The value to search for.
        search_list : list or numpy.ndarray
            The list to search in.

        Returns
        -------
        int
            The index of the value.
		"""
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, vals):
		"""
        Interpolates the TIFF image at a single point.

        Parameters
        ----------
        vals : tuple
            A tuple of coordinates of the point.

        Returns
        -------
        float
            The interpolated value.
		"""
		return self.interpolator(vals)


	def __getitem__(self, idxs):
		"""
        Allows indexing into the TIFF image.

        Parameters
        ----------
        idxs : tuple
            A tuple of indices for each dimension of the TIFF image.

        Returns
        -------
        float or numpy.ndarray
            The interpolated value(s) at the given indices.
		"""

		
		if np.any([isinstance(x, slice) for x in idxs]) :
			return self.values[idxs]
		else:
			
			outval =  self.interp_point(idxs)
			return np.squeeze(outval)



def getSunAngles(scName = None, scFrame = None, epoch = None, correction = 'LT+S'):
	"""
    Computes the right ascension and declination of the Sun as seen from a
    spacecraft.

    Parameters
    ----------
    scName : str, optional
        The name of the spacecraft.
    scFrame : str, optional
        The reference frame of the spacecraft.
    epoch : float, optional
        The epoch for the calculation.
    correction : str, default='LT+S'
        The aberration correction to use.

    Returns
    -------
    tuple
        A tuple containing the right ascension and declination in radians.
	"""
	sunPos = sp.spkezr('Sun', epoch, scFrame, correction, scName)[0][0:3]
	[_, ra, dec] = sp.recrad(sunPos)

	return ra, dec

	

def epochRange(startEpoch = None, duration = None, step = 100):
	"""
    Generates a range of epochs.

    Parameters
    ----------
    startEpoch : str or float, optional
        The start epoch in a format recognized by SPICE or as a float.
    duration : float, optional
        The duration of the epoch range in seconds.
    step : int, default=100
        The step size in seconds.

    Returns
    -------
    numpy.ndarray
        An array of epochs.
	"""
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
	"""
    Generates a range of epochs between a start and end epoch.

    Parameters
    ----------
    startEpoch : str or float, optional
        The start epoch in a format recognized by SPICE or as a float.
    endEpoch : str or float, optional
        The end epoch in a format recognized by SPICE or as a float.
    step : int, default=100
        The step size in seconds.

    Returns
    -------
    numpy.ndarray
        An array of epochs.
	"""
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
    Computes the right ascension and declination for a set of vectors.

    Parameters
    ----------
    vecs : numpy.ndarray
        An array of shape (N, 3) containing the vectors.
    periodicity : int, default=360
        The periodicity of the right ascension in degrees.

    Returns
    -------
    tuple
        A tuple containing the right ascension and declination in radians.
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
    Converts a TIFF map to a format that can be used to assign values to a
    planetary mesh.

    Parameters
    ----------
    tifFile : str, default=''
        The path to the TIFF file.
    latSampling : float, default=''
        The latitude sampling step in radians.
    inferSampling : bool, default=False
        Whether to infer the sampling from the TIFF file.
    lonSampling : float, default=''
        The longitude sampling step in radians.
    planet : pyRTX.Planet, default=''
        The Planet object containing the mesh.
    lat0 : float, default=-numpy.pi/2
        The minimum latitude.
    lat1 : float, default=numpy.pi/2
        The maximum latitude.
    lon0 : float, default=0
        The minimum longitude.
    lon1 : float, default=2*numpy.pi
        The maximum longitude.

    Returns
    -------
    TiffInterpolator
        An interpolator for mapping the TIFF values to the mesh.
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


# def get_spacecraft_area(spacecraft, ra = 0.0, dec = 0.0, epoch = None):

# 	'''
# 	Compute a pyRTX.Spacecraft apparent area as seen by the direction specified 
# 	by a pair of right ascension - declination

# 	Input:
# 	spacecraft [pyRTX.Spacecraft] 	: the spacecraft object
# 	ra [float] 						: right ascension (in rad)
# 	dec [float] 					: declination (rad)
# 	epoch [float or None]			: epoch for the computation (this is used when moving Spice
# 										frames are used for the Spacecraft definition)

# 	Output:
# 	area [float] 					: the apparent area. The measurement units depend on the units of the
# 				   						Spacecraft object

# 	TODO: avoid hardcoded width/height but rather use an automated method

# 	'''
# 	from pyRTX.classes.PixelPlane import PixelPlane
# 	from pyRTX.classes.RayTracer  import RayTracer
# 	rays = PixelPlane( 
# 			spacecraft = spacecraft,   # Spacecraft object 
# 			mode = 'Fixed',   # Mode: can be 'Dynamic' ( The sun orientation is computed from the kernels), or 'Fixed'
# 			distance = 10000,	    # Distance of the ray origin from the spacecraft
# 			width = 10,	    # Width of the pixel plane
# 			height = 10,        # Height of the pixel plane
# 			lon = ra,
# 			lat = dec, 
# 			ray_spacing = 0.01, # Ray spacing (in m)
# 		)

# 	rtx = RayTracer(        
#                         spacecraft,                    # Spacecraft object
#                         rays,                   # pixelPlane object
#                         kernel = 'Embree',      # The RTX kernel to use
#                         bounces = 1,            # The number of bounces to account for
#                         diffusion = False,       # Account for secondary diffusion
#                         ) 

# 	rtx.trace(epoch)
# 	hit_rays = rtx.index_ray_container
# 	Area =  len(hit_rays[0])/rays.norm_factor
# 	return Area



def get_spacecraft_area(spacecraft, rays, ra = 0.0, dec = 0.0, epoch = None):
	"""
    Computes the apparent area of a spacecraft as seen from a given direction.

    Parameters
    ----------
    spacecraft : pyRTX.Spacecraft
        The spacecraft object.
    rays : pyRTX.PixelPlane
        The pixel plane for raytracing.
    ra : float, default=0.0
        The right ascension of the viewing direction in radians.
    dec : float, default=0.0
        The declination of the viewing direction in radians.
    epoch : float, optional
        The epoch for the computation.

    Returns
    -------
    float
        The apparent area of the spacecraft.
	"""
	from pyRTX.classes.PixelPlane import PixelPlane
	from pyRTX.classes.RayTracer  import RayTracer

	rtx = RayTracer(        
                        spacecraft,                    # Spacecraft object
                        rays,                   # pixelPlane object
                        kernel = 'Embree3',      # The RTX kernel to use
                        bounces = 1,            # The number of bounces to account for
                        diffusion = False,       # Account for secondary diffusion
                        ) 

	rays.update_latlon(lon = ra, lat = dec)
	rtx.trace(epoch)
	hit_rays = rtx.index_ray_container
	Area =  len(hit_rays[0])/rays.norm_factor
	return Area



### ------------------------------- User-Defined functions ------------------------------ ###

from pyRTX.core.parallel_utils import parallel

@parallel
def get_sun_exposed_area(sc, rtx, epoch):
	"""
    Computes the sun-exposed area of a spacecraft.

    Parameters
    ----------
    sc : pyRTX.Spacecraft
        The spacecraft object.
    rtx : pyRTX.RayTracer
        The ray tracer object.
    epoch : float
        The epoch for the computation.

    Returns
    -------
    float
        The sun-exposed area of the spacecraft.
	"""

	# Get ra, dec of the solar direction
	sundir  = sp.spkezr( 'Sun', epoch, sc.base_frame, 'LT+S', sc.name )[0][0:3]
	sundir  = sundir / np.linalg.norm(sundir)
	_, ra, dec = sp.recrad(sundir)

	# Update the ray tracer
	rtx.rays.update_latlon(lon = ra, lat = dec)

	# Get area
	rtx.trace(epoch)
	hit_rays = rtx.index_ray_container
	Area =  len(hit_rays[0])/rtx.rays.norm_factor
 
	return Area

def compute_body_positions(target, epochs, frame, obs, abcorr = 'LT + S'):
	"""
    Computes the relative positions of a target body with respect to an
    observing body.

    Parameters
    ----------
    target : str
        The name of the target body.
    epochs : list of float
        A list of epochs.
    frame : str
        The reference frame of the output position vector.
    obs : str
        The name of the observing body.
    abcorr : str, default='LT + S'
        The aberration correction flag.

    Returns
    -------
    list
        A list of position vectors.
	"""
	state = sp.spkezr(target, np.array(epochs), frame, abcorr, obs)

	return [state[0][i][0:3] for i in range(len(epochs))]

def compute_body_states(target, epochs, frame, obs, abcorr = 'LT + S'):
	"""
    Computes the relative position and velocity of a target body with respect
    to an observing body.

    Parameters
    ----------
    target : str
        The name of the target body.
    epochs : list of float
        A list of epochs.
    frame : str
        The reference frame of the output state vector.
    obs : str
        The name of the observing body.
    abcorr : str, default='LT + S'
        The aberration correction flag.

    Returns
    -------
    list
        A list of state vectors (position and velocity).
	"""
	state = sp.spkezr(target, np.array(epochs), frame, abcorr, obs)

	return [state[0][i][:] for i in range(len(epochs))]

### ------------------------------------------------------------------------------------- ###
