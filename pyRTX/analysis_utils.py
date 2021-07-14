# Utilities for data/output analysis
import numpy as np
from scipy import interpolate
import spiceypy as sp
from PIL import Image
from utils_rt import block_normalize
from numba import jit









class LookupTable():
	def __init__(self, linspace_x, linspace_y, values):
		self.linspace_x = linspace_x
		self.linspace_y = linspace_y
		self.values = values

		self._set_defaults()


	def _set_defaults(self):
		self.interpType = 'cubic'

	def interpolator(self, x, y):
		x, y = np.meshgrid(x,y)
		meshgrid_x, meshgrid_y = np.meshgrid(self.linspace_x, self.linspace_y)
		#return  interpolate.griddata((meshgrid_x.ravel(), meshgrid_y.ravel()), self.values.T.ravel(),np.array([x,y]).T, method = self.interpType)
		return  interpolate.griddata((meshgrid_x.ravel(), meshgrid_y.ravel()), self.values.T.ravel(),(x,y), method = self.interpType)

	def set_interpType(self, interpType):
		self.interpType = interpType


	def get_idx(self, ind, search_list):
		return np.where(search_list == ind)[0][0]

	
	def interp_point(self, x, y):
		return self.interpolator(x,y)


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
		
		return interpolate.interpn(self.axes,self.values, vals, method = 'linear') 


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
			



	def quickPlot(self, xlabel = None, ylabel = None, title = None, conversion = 1, clabel = None, cmap = 'viridis', saveto = None):
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





def convertTIFtoMesh(tifFile = '', latSampling = '', lonSampling = '', planet = '', lat0 = -np.pi/2, lat1 = np.pi/2, lon0 = 0, lon1 = 2*np.pi):
	"""
	Convert a TIF map of emissivities to the format needed for assigning values to each face of a planetary 3D mesh

	Parameters:
	tifFile : [str] the path to the TIFF file
	latSampling: [float, in rad] sampling step of latitude
	lonSampling: [float, in rad] sampling step of longitude
	planet: [class.Planet] Planet object containing the mesh and the planetary frames

	Returns:
	emissivityInterpolator: [class.LookupTableND] an interpolator for mapping the temperatures on the mesh faces. This is intended to be passed
				to the Planet class via the setter method: Planet.emissivity = emissivityInterpolator

	"""


	im = Image.open(tifFile)
	
	#arr = np.expand_dims(np.array(im), axis = 2)
	arr = np.array(im)
	print(arr.shape)

	#import matplotlib.pyplot as plt
	#plt.figure()
	#print(np.array(im).shape)
	#plt.contourf(np.array(im))


	lats = np.linspace(lat0, lat1, int((lat1 - lat0) / latSampling ))
	lons = np.linspace(lon0, lon1, int((lon1 - lon0) / lonSampling ))


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
