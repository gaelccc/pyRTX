import os, sys, itertools

import numpy as np
import spiceypy as sp
import xarray as xr
import pickle as pkl
from scipy import interpolate

from pyRTX.classes.SRP import SolarPressure
from pyRTX.classes.RayTracer import RayTracer
from pyRTX.core.physical_utils import preprocess_RTX_geometry
from pyRTX.core.parallel_utils import parallel

from copy import deepcopy

class LookUpTable():
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
	def __init__(self, **kwargs):

		conv = np.pi / 180.

		if 'rtx' in kwargs.keys() and isinstance(kwargs['rtx'],RayTracer):	
			self._rtx = deepcopy(kwargs['rtx'])
			self._srp = SolarPressure( self._rtx.spacecraft, self._rtx, baseflux = None, )
		else: self._rtx = None; self._srp = None

		if 'moving_frames' in kwargs.keys() and isinstance(kwargs['moving_frames'],(list, np.ndarray)):	
			self._moving_frames = kwargs['moving_frames'] 
		else: self._moving_frames = []

		if 'limits' in kwargs.keys() and isinstance(kwargs['limits'],(list, np.ndarray)):	
			self._limits = kwargs['limits'] 
		else: self._limits = []
  
		if 'ra' in kwargs.keys() and isinstance(kwargs['ra'],(list, np.ndarray)):	
			self._ra = kwargs['ra'] * conv
		else: self._ra = np.linspace(0,360,20) * conv

		if 'dec' in kwargs.keys() and isinstance(kwargs['dec'],(list, np.ndarray)):	
			self._dec = kwargs['dec'] * conv
		else: self._dec = np.linspace(-90,90,10) * conv
  
		if 'res' in kwargs.keys() and isinstance(kwargs['res'],(int,float)):	
			self._res = kwargs['res'] 
		else: self._res = 1.0
  
		if 'method' in kwargs.keys() and isinstance(kwargs['method'],str):	
			self._method = kwargs['method']
		else: self._method = 'full'

		if 'eul_set' in kwargs.keys() and isinstance(kwargs['eul_set'],tuple): 
			self._eul_set = kwargs['eul_set']
		else: self._eul_set = None

		if 'eul_set' in kwargs.keys() and isinstance(kwargs['eul_set'],tuple): 
			self._eul_idxs = {ax: idx for idx, ax in enumerate(self._eul_set)}
		else: self._eul_idxs = None
  
		if 'ref_epc' in kwargs.keys() and isinstance(kwargs['ref_epc'],str): 
			self._ref_epc = sp.str2et(kwargs['ref_epc'])
		else: self._ref_epc = None

		if 'xarray' in kwargs.keys() and isinstance(kwargs['xarray'],str): 
			self._init_from_xarray(kwargs['xarray'])
		else: self._data = None
  
		if 'precomputation' in kwargs.keys(): 
			self.sp_data = kwargs['precomputation']
		else: self.sp_data = None


	def _init_from_xarray(self, filename):
		"""Init object from an xarray"""

		# Load object
		LUT = xr.open_dataset(filename)
		self._data = deepcopy(LUT)
  
		# Store properties
		self._elements      = self._data.elements.split(',')
		self._moving_frames = self._data.moving_frames.split(',') if self._data.moving_frames != '' else []
		self._base_frame    = self._data.base_frame
		self._eul_set       = tuple([int(eul) for eul in self._data.eul_set.split(',')])
		self._eul_idxs      = {ax: idx for idx, ax in enumerate(self._eul_set)}
		self._dims          = self._data.look_up_table.dims
		self._axes          = [self._data.coords[key].data for key in list(self._dims[:-1])]

		# Close
		LUT.close()
  

	def store_attrs(self, epochs):
		"""Compute dimension, shape and attributes for the xarray"""
  
		# Attributes
		self._sc_model    = deepcopy(self._rtx.spacecraft.spacecraft_model)
		self._base_frame  = self._rtx.spacecraft.base_frame
		materials         = self._rtx.spacecraft.materials()	
		self._elements    = list(materials['props'].keys())
		self._face_idxs   = materials['idxs']
		self._norm_factor = self._rtx.norm_factor
		if self._ref_epc == None: self._ref_epc = epochs[0]
   
		# Build dimensions, values, moving_frames
		self._dims = []; self._axes = []; self._moving_frames = []
  
		# Loop for every element
		for element in self._elements:

			frame = self._sc_model[element]['frame_name']
			
			# Check if the frame is fixed wrt sc base frame
			if frame != self._base_frame: 
				
				eul_angles  = np.zeros((len(epochs),3))
				
				# Compute euler angles
				for e, epc in enumerate(epochs):
					
					rot  = sp.pxform(frame, self._base_frame, epc)
     
					eul_angles[e,:] = sp.m2eul(rot, *self._eul_set)
				
				# Check lower and upper bound for every axis
				for i, ax in enumerate(self._eul_set):
					
					dim = f'{frame}{ax}'
					
					lb = np.min(eul_angles[:,i])
					ub = np.max(eul_angles[:,i])
					
					if abs(ub - lb) > self._tol:
					
						span = np.arange(lb - 2e-3, ub + 2e-3 + self._res, self._res)
						
						self._dims.append(dim)
						self._axes.append(span) 
						if frame not in self._moving_frames: self._moving_frames.append(frame)

						print(f"\n * RANGE for {dim}")
						print(f" * Max: {ub*180/np.pi}  * Min: {lb*180/np.pi}")
  
		# Update spacecraft model
		for frame in self._moving_frames:
			frame_elms = [elem for elem in self._elements if self._sc_model[elem]['frame_name'] == frame ]
			for element in frame_elms:
				self._rtx.spacecraft.spacecraft_model[element]['frame_type']  = 'UD'

		# Build attribute dictionary for xarray
		units = 'km/s**2' if self._method == 'full' else 'm**2'
		self._attrs = {
			'moving_frames': ",".join(self._moving_frames),
			'eul_set': ",".join([str(eul) for eul in self._eul_set]),
			'tolerance': str(self._tol),
			'base_frame': self._base_frame,
			'elements': ",".join(self._elements),
			'method': self._method,
			'units': units,
			}

		# Append ra and dec
		self._dims.append('ra')
		self._dims.append('dec')
		self._axes.append(self._ra) 
		self._axes.append(self._dec) 

		# Compute shape for xarray
		shape = tuple([len(r) for r in self._axes])

		# Check mode
		if self._method == 'full': shape += (3,)
		elif self._method == 'elements': 
			shape += (len(self._elements),1)
			self._dims.append('element')
			self._axes.append(range(len(self._elements)))           
		self._dims.append('value')
  
		# Build coordinates
		self._coords = {self._dims[i]: vals for i, vals in enumerate(self._axes)}
  
		return shape


	def compute(self, epochs, n_cores = None):
		"""Compute look up table."""

		# Store inputs
		if not isinstance(epochs, (list, np.ndarray)): epochs = [epochs]
		if n_cores == None: n_cores = os.cpu_count() - 1
		else: n_cores = min(n_cores, os.cpu_count() - 1)

		print('\n *** Calculating dimension ...')
  
		# Build configuration dictionary
		shape = self.store_attrs(epochs)

		print(f'\n *** LUT size: {shape} ...')
  
		# Init data array
		self._data  = np.zeros(shape)

		# Find all permutations
		SEQ    = list(itertools.product(*self._axes))
		IDX    = list(itertools.product(*[range(l) for l in shape[:-1]]))
		steps  = [int(i) for i in np.linspace(0,len(SEQ),n_cores+1)]
		SEQ    = [SEQ[i:j] for (i,j) in zip(steps[:-1],steps[1:])]
		IDX    = [IDX[i:j] for (i,j) in zip(steps[:-1],steps[1:])]
		INPUTS = [(I,S) for I,S in zip(IDX,SEQ)]
  
		print('\n *** Filling the arrays ...')
  
		# Fill LUT values
		OUTPUTS = self.fill(INPUTS, n_cores = n_cores)
		for r, result in enumerate(OUTPUTS):
			for i, idxs in enumerate(INPUTS[r][0]):
				self._data[tuple(idxs)] = result[i]
    
		print('\n *** Generating the x-array ...')
  
		# Define X-array LUT
		self._data = xr.Dataset( data_vars = {'look_up_table': (self._dims, self._data)}, 
                            	 coords    = self._coords,
                               	 attrs     = self._attrs,)
  
		print(f'\n *** LUT in {self.mode} completed!\n')
  
  
	@parallel
	def fill(self, INPUT):
		"""Fill look up table values."""
  
		IDX, SEQ = INPUT
		shape    = (len(SEQ),3) if self._method == 'full' else (len(SEQ),)
		OUTPUT   = np.zeros(shape)

		# Loop for every sequence
		for n, tup in enumerate(zip(IDX,SEQ)):
			
			idxs, seq = tup

			# Mapping dictionary
			map            = { frame: np.zeros((3,)) for frame in self._moving_frames }
			map['ra']      = -1.
			map['dec']     = -1.
			map['element'] = ''
    
			# Extract sequence
			for v, val in enumerate(seq):
				
				# Find axis label
				dim = self._dims[v]

				# Store value in the dictionary
				if dim[:-1] in self._moving_frames: 
					eul_idx = self._eul_idxs[int(dim[-1])]
					map[dim[:-1]][eul_idx] = val
				else: map[dim] = val

			# Update spacecraft model
			for frame in self._moving_frames:
				
				# Find rotation matrix
				angles  = map[frame]
				rot     = sp.eul2m(*angles,*self._eul_set)
				tmatrix = np.zeros((4,4))
				tmatrix[:3,:3] = rot
		
				# Update model
				frame_elms = [elem for elem in self._elements if self._sc_model[elem]['frame_name'] == frame ]
    
				# Dump UD rotation for every element
				for element in frame_elms:
					self._rtx.spacecraft.spacecraft_model[element]['UD_rotation'] = tmatrix
			
			# Extract ra, dec
			ra, dec = (map['ra'], map['dec'])

			# Compute full mode
			if self._method == 'full':
       
       			# Update rayTracer
				self._srp.rayTracer.rays.update_latlon(lon = ra, lat = dec)
	
				# Compute normalized accel
				value = self._srp.compute(self._ref_epc)
    
			# Compute elements mode
			elif self._method == 'elements':

				# Retrieve spacecraft mesh
				mesh       = self._rtx.spacecraft.dump(self._ref_epc)
				_, _, N, _ = preprocess_RTX_geometry(mesh)
    
				# Update longitude and latitude
				# NOTE: to update the pixel plane ra and dec, you must furnish the 
				# direction from the spacecraft to the pixel plane in spacecraft coordinates!
				self._rtx.rays.update_latlon(lon = ra, lat = dec)

				# Launch the ray-tracer
				self._rtx.trace( self._ref_epc )

				# Direction from the spacecraft to the pixel plane
				planedir = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])
    
				# Face indexes
				elem_idxs = np.array( range(self._face_idxs[idxs[-1]][0], self._face_idxs[idxs[-1]][1] + 1) )

				# Query the ray tracer
				tri_idxs  = self._rtx.index_tri_container[0]  # only first bounce considered for the moment
				
				# First method (dA is already the effective area)
				elem_idxs = [i for i, id in enumerate( np.in1d(tri_idxs, elem_idxs) ) if id]
				elem_idxs = [tri_idxs[i] for i in elem_idxs]
				normals   = N[elem_idxs]
				cosines   = np.sum(np.multiply(normals, planedir), axis = 1)
				cosines   = cosines[cosines>0]
				dA        = np.ones(len(cosines))/(self._norm_factor) 
				value     = np.sum(dA, axis = 0) * 1e6
    
			OUTPUT[n] = value
   
		return OUTPUT


	def save(self, filename: str, complev: int = 1):
		"""
		Method to save precomputed array.
		"""
		if os.path.exists(filename): os.remove(filename)
		self._data.to_netcdf(filename, encoding = self._data.encoding.update({'zlib': True, 'complevel': complev}))

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
	def mode(self):
		"""Returns the computational mode (full/elements)."""
		return self._method

	@property
	def dims(self):
		"""Returns the dimension axes of the xarray."""
		return self._dims[:-1]

	@property
	def elements(self):
		"""Returns the list of sc elements."""
		return self._elements


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
  

	def query(self, epoch, ra, dec, element = None):
		"""
		Query the look up table for a given epoch, ra, dec and 
  		(eventually) the single element of the shape.
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
  
		if element != None: query.append(self.elements.index(element))
  
		query  = tuple(query)
  
		output = interpolate.interpn(self.axes, self.data, query, method = 'linear')

		if element != None: output = np.reshape(output, (output.shape[0],))

		return output


