import sys
import numpy as np
import spiceypy as sp
import xarray as xr
from scipy import interpolate

from copy import deepcopy


class LookUpTable():
	"""
	This class is used to store results in the shape aof a lookup table.
	This is mainly used to store the results of a set of raytracing results
	example: the solar pressure for a body is computed for a grid of RA/DEC values.
	these values can be stored in the LookupTable object and later retrieved.
	This class offers the possibility of not oly retrieving pre-computed values, but
	also interpolating between grid points.

	NOTE: the grid of the lookup table does not need to be regular
	the interpolation is based on numpy griddata method which is able to cope
	with unstructured grids
	"""
	def __init__(self, filename):

		if not isinstance(filename, str) or not '.nc' in filename:
			print('\n *** ERROR: a .nc filename is expected!')
			sys.exit(0)
		self._init_from_xarray(filename)


	def _init_from_xarray(self, filename):
		"""Init object from an xarray"""

		# Load object
		LUT = xr.open_dataset(filename)
		self._data = deepcopy(LUT)
  
		# Store properties
		self._moving_frames = self._data.moving_frames.split(',') if self._data.moving_frames != '' else []
		self._base_frame    = self._data.base_frame
		self._eul_set       = tuple([int(eul) for eul in self._data.eul_set.split(',')])
		self._eul_idxs      = {ax: idx for idx, ax in enumerate(self._eul_set)}
		self._dims          = self._data.look_up_table.dims
		self._axes          = [self._data.coords[key].data for key in list(self._dims[:-1])]
		self._units         = self._data.units
  
		# Close
		LUT.close()
  

	@property
	def moving_frames(self):
		"""Returns the list of frames wich are NOT fixed wrt spacecraft body-frame."""
		return self._moving_frames

	@property
	def data(self):
		"""Returns the xarray data."""
		return self._data.look_up_table.data

	@property
	def attrs(self):
		"""Returns the xarray attributes."""
		return self._data.attrs

	@property
	def axes(self):
		"""Returns the xarray axes values."""
		return self._axes

	@property
	def base_frame(self):
		"""Returns the spacecraft body-frame."""
		return self._base_frame

	@property
	def ref_epc(self):
		"""Returns the reference epoch of the LUT."""
		return self._ref_epc

	@property
	def eul_set(self):
		"""Returns the euler set used for the computations."""
		return self._eul_set

	@property
	def eul_idxs(self):
		"""Returns the euler set used for the computations."""
		return self._eul_idxs

	@property
	def dims(self):
		"""Returns the dimension axes of the xarray."""
		return self._dims[:-1]

	@property
	def units(self):
		"""Returns the units of the xarray values."""
		return self._units
  

	def query(self, epoch, ra, dec,):
		"""
		Query the look up table for a given epoch, ra, dec (in degrees).
		"""
  
		ra = ra * np.pi / 180.; 
		dec = dec * np.pi / 180.

		query = []

		for frame in self.moving_frames:
      
			if self.sp_data != None:
				rot = np.array(self.sp_data.getRotation(epoch, frame, self.base_frame)[:3,:3])
			else:
				rot = sp.pxform(frame, self.base_frame, epoch)
    
			eul_angles = sp.m2eul(rot, *self.eul_set)
   
			query += [eul_angles[self._eul_idxs[ax]] for ax in self.eul_set if f'{frame}{ax}' in self.dims]
  
		query += [ra, dec]
  
		query  = tuple(query)
	

  
		output = np.squeeze(interpolate.interpn(self.axes, self.data, query, method = 'linear'))

		return output


