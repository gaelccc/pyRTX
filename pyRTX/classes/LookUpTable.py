import sys
import numpy as np
import spiceypy as sp
import xarray as xr
from scipy import interpolate

from copy import deepcopy


class LookUpTable():
	"""
    A class for storing and interpolating lookup tables from NetCDF files.

    This class is used to store results in the shape of a lookup table.
    This is mainly used to store the results of a set of raytracing results
    example: the solar pressure for a body is computed for a grid of RA/DEC values.
    these values can be stored in the LookupTable object and later retrieved.
    This class offers the possibility of not only retrieving pre-computed values, but
    also interpolating between grid points.

    NOTE: the grid of the lookup table does not need to be regular
    the interpolation is based on numpy griddata method which is able to cope
    with unstructured grids.
	"""
	def __init__(self, filename):
		"""
        Initializes the LookUpTable object.

        Parameters
        ----------
        filename : str
            The path to the NetCDF file containing the lookup table.
		"""

		if not isinstance(filename, str) or not '.nc' in filename:
			print('\n *** ERROR: a .nc filename is expected!')
			sys.exit(0)
		self._init_from_xarray(filename)


	def _init_from_xarray(self, filename):
		"""
        Initializes the lookup table from an xarray Dataset.

        Parameters
        ----------
        filename : str
            The path to the NetCDF file.
		"""

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
		"""
        Returns the list of moving frames.

        Returns
        -------
        list
            A list of the moving frames.
		"""
		return self._moving_frames

	@property
	def data(self):
		"""
        Returns the lookup table data.

        Returns
        -------
        numpy.ndarray
            The lookup table data.
		"""
		return self._data.look_up_table.data

	@property
	def attrs(self):
		"""
        Returns the attributes of the xarray Dataset.

        Returns
        -------
        dict
            The attributes of the xarray Dataset.
		"""
		return self._data.attrs

	@property
	def axes(self):
		"""
        Returns the coordinate axes of the lookup table.

        Returns
        -------
        list
            A list of the lookup table's coordinate axes.
		"""
		return self._axes

	@property
	def base_frame(self):
		"""
        Returns the base frame of the lookup table.

        Returns
        -------
        str
            The name of the base frame.
		"""
		return self._base_frame

	@property
	def ref_epc(self):
		"""
        Returns the reference epoch of the lookup table.

        Returns
        -------
        float
            The reference epoch.
		"""
		return self._ref_epc

	@property
	def eul_set(self):
		"""
        Returns the Euler angle set used for the lookup table.

        Returns
        -------
        tuple
            A tuple of the Euler angle set.
		"""
		return self._eul_set

	@property
	def eul_idxs(self):
		"""
        Returns the Euler angle indices.

        Returns
        -------
        dict
            A dictionary mapping Euler angle axes to their indices.
		"""
		return self._eul_idxs

	@property
	def dims(self):
		"""
        Returns the dimensions of the lookup table.

        Returns
        -------
        tuple
            A tuple of the lookup table's dimensions.
		"""
		return self._dims[:-1]

	@property
	def units(self):
		"""
        Returns the units of the lookup table values.

        Returns
        -------
        str
            The units of the lookup table values.
		"""
		return self._units
  

	def query(self, epoch, ra, dec,):
		"""
        Queries the lookup table for a given epoch, right ascension, and
        declination.

        Parameters
        ----------
        epoch : float
            The epoch in TDB seconds past J2000.
        ra : float
            The right ascension in degrees.
        dec : float
            The declination in degrees.

        Returns
        -------
        numpy.ndarray
            The interpolated value from the lookup table.
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
