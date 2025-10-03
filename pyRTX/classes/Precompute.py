import os, sys
import pickle as pkl
import spiceypy as sp
import numpy as np
import xarray as xr

from pyRTX.core.analysis_utils import compute_body_positions, compute_body_states
from pyRTX.constants import au 


class Precompute():
	"""
	A Class to perform calls to spiceypy in advance.
	"""	
 
	def __init__(self, epochs: list,):
		"""
		Initialization method for the class.
		Params:
		- epochs (epochs of the precomputations)
		"""

		# Store inputs
		self._epochs    = epochs
		self._config 	= {}


	def addPosition(self, observer: str, target: str, frame: str, correction: str = 'CN'):
		"""
		Method to precompute position vectors.
  
		Params:
  		- observer body (str)
		- target body (str)
		- frame (str)
		- aberration correction (str)
		"""
    
		if 'position' not in self._config.keys():
			self._config['position'] = {
				'dimensions': ("time", "pos_param", "pos"),
				'coordinates': {
					'time'	    : self._epochs,
					'pos_param' : [],
					},
				}

		param = observer + ' / ' + target + ' / ' + frame + ' / ' + correction
		if param not in self._config['position']['coordinates']['pos_param']:
			self._config['position']['coordinates']['pos_param'].append(param)


	def addState(self, observer: str, target: str, frame: str, correction: str = 'CN'):
		"""
		Method to precompute state vectors.
  
		Params:
  		- observer body (str)
		- target body (str)
		- frame (str)
		- aberration correction (str)
		"""
    
		if 'state' not in self._config.keys():
			self._config['state'] = {
				'dimensions': ("time", "state_param", "posvel"),
				'coordinates': {
					'time'	    : self._epochs,
					'state_param' : [],
					},
				}

		param = observer + ' / ' + target + ' / ' + frame + ' / ' + correction
		if param not in self._config['state']['coordinates']['state_param']:
			self._config['state']['coordinates']['state_param'].append(param)
   
   
	def addRotation(self, base_frame: str, target_frame: str):
		"""
		Method to precompute position vectors.
  
		Params:
  		- base_frame (str)
		- target_frame (str)
		"""
      
		if 'rotation' not in self._config.keys():
			self._config['rotation'] = {
				'dimensions': ("time", "rot_param", "dim1", "dim2"),
				'coordinates': {
					'time'	     : self._epochs,
					'rot_param'  : [],
					},
				}

		param = base_frame + ' / ' + target_frame 
		if param not in self._config['rotation']['coordinates']['rot_param']:
			self._config['rotation']['coordinates']['rot_param'].append(param)
   

	def precomputeSolarPressure(self, sc, planet, correction = 'LT+S'):
		"""
		Method to perform precalculation for solar radiation pressure.
  
		Params:
		- sc: object of the class Spacecraft
		- planet: object of the class Planet
		"""

		# Get data
		sc_name       = sc.name
		sc_frame      = sc.base_frame
		planet_name   = planet.name
		planet_frame  = planet.bodyFrame
  
		# Add positions
		observers = [sc_name] * 2
		targets   = [planet_name, 'Sun']
		frames    = [sc_frame] * 2 
		abcorr    = [correction] * 2
		for i, obs in enumerate(observers):
			self.addPosition(observer=obs, target=targets[i], frame=frames[i], correction=abcorr[i])
     
     	# Add rotations
		base_frames   = [sc_frame, sc_frame]
		target_frames = [planet_frame, sc_frame]
		for elem in sc.spacecraft_model.keys():
			if sc.spacecraft_model[elem]['frame_name'] not in base_frames:
				base_frames.append(sc.spacecraft_model[elem]['frame_name'])
				target_frames.append(sc_frame)
		for i, base in enumerate(base_frames):
			self.addRotation(base_frame=base, target_frame=target_frames[i])


	def precomputePlanetaryRadiation(self, sc, planet, moving_frames = [], correction = 'CN'):
		"""
		Method to perform precalculation for albedo and 
  		thermal infrared acceleration.
    
		Params:
		- sc: object of the class Spacecraft
		- planet: object of the class Planet
		"""
		
		# Get data
		sc_name		  = sc.name
		sc_frame	  = sc.base_frame
		planet_name   = planet.name
		planet_frame  = planet.bodyFrame
		sunfix_frame  = planet.sunFixedFrame
  
		# Add positions
		observers = [planet_name, planet_name, planet_name] 
		targets   = [sc_name, sc_name, 'Sun'] 
		frames    = [sunfix_frame, planet_frame, planet_frame]	
		abcorr    = [correction] * 3
		for i, obs in enumerate(observers):
			self.addPosition(observer=obs, target=targets[i], frame=frames[i], correction=abcorr[i])

		# Add rotations	
		base_frames   = [sunfix_frame, planet_frame] + moving_frames
		target_frames = [sc_frame, sunfix_frame] + [sc_frame]*len(moving_frames)

		for i, base in enumerate(base_frames):
			self.addRotation(base_frame=base, target_frame=target_frames[i])


	def precomputeDrag(self, sc, planet_name, moving_frames = [], accel_frame = '', correction = 'LT+S'):
		"""
		Method to perform precalculation for drag acceleration.
    
		Params:
		- sc: object of the class Spacecraft
		- planet_name: name of the body
		- accel_frame: frame of the acceleration
		"""
		
		# Get data
		sc_name		  = sc.name
		sc_frame	  = sc.base_frame
		if accel_frame == '': accel_frame = 'IAU_%s'%planet_name.upper()
  
		# Add state
		observers = [planet_name,] 
		targets   = [sc_name,] 
		frames    = [accel_frame,] 
		abcorr    = [correction,]
		for i, obs in enumerate(observers):
			self.addState(observer=obs, target=targets[i], frame=frames[i], correction=abcorr[i])

		# Add rotations	
		base_frames   = [accel_frame,] + moving_frames
		target_frames = [sc_frame,] + [sc_frame]*len(moving_frames)

		for i, base in enumerate(base_frames):
			self.addRotation(base_frame=base, target_frame=target_frames[i])
  
  
	def dump(self):
		"""
		Method to perform precalculation.
  
		Params:
		- filename: path to xarray
		"""

		datavars    = {}
		coordinates = {}

		for datavar in self._config.keys():
			
			config  = self._config[datavar]
			coords  = config['coordinates']
			dims    = config['dimensions']
			times   = coords['time']

			if datavar == 'position':
		
				params  = coords[f'pos_param']
				data  	= np.zeros( (len(times), len(params), 3) )
		
				for i, param in enumerate(params):
					splitted = param.split('/')
					observer = splitted[0].lstrip().rstrip()
					target   = splitted[1].lstrip().rstrip()
					frame    = splitted[2].lstrip().rstrip()
					corr     = splitted[3].lstrip().rstrip()
					data[:,i,:] = compute_body_positions(target, times, frame, observer, abcorr = corr)

			if datavar == 'state':
		
				params  = coords[f'state_param']
				data  	= np.zeros( (len(times), len(params), 6) )
		
				for i, param in enumerate(params):
					splitted = param.split('/')
					observer = splitted[0].lstrip().rstrip()
					target   = splitted[1].lstrip().rstrip()
					frame    = splitted[2].lstrip().rstrip()
					corr     = splitted[3].lstrip().rstrip()
					data[:,i,:] = compute_body_states(target, times, frame, observer, abcorr = corr)
     
			elif datavar == 'rotation':

				params  = coords[f'rot_param']
				data    = np.zeros( (len(times), len(params), 4, 4) )
		
				for i, param in enumerate(params):
					splitted      = param.split('/')
					base_frame    = splitted[0].lstrip().rstrip()
					target_frame  = splitted[1].lstrip().rstrip()
					data[:,i,:,:] = [self.pxform_convert(sp.pxform(base_frame, target_frame, epoch)) for epoch in times] 
		
			datavars[datavar] = (dims, data)
		
			for coord in coords.keys():
				if coord not in coordinates.keys(): coordinates[coord] = coords[coord]

		self._dataset = xr.Dataset( datavars, coordinates )
  

	def getPosition(self, epoch, observer: str, target: str, frame: str, correction: str):
		"""
		Method to get position vector.
		"""
		param = f'{observer} / {target} / {frame} / {correction}'
		return self._dataset.position.sel(time = epoch, pos_param = param).data


	def getState(self, epoch, observer: str, target: str, frame: str, correction: str):
		"""
		Method to get state vector.
		"""
		param = f'{observer} / {target} / {frame} / {correction}'
		return self._dataset.state.sel(time = epoch, state_param = param).data


	def getRotation(self, epoch, base_frame: str, target_frame: str):
		"""
		Method to get rotation matrix.
		"""
		param = f'{base_frame} / {target_frame}'
		return self._dataset.rotation.sel(time = epoch, rot_param = param).data


	def getArray(self):
		"""
		Method to get precomputed array.
		"""
		return self._dataset


	def save(self, filename: str, complev: int = 1):
		"""
		Method to save precomputed array.
		"""
		if os.path.exists(filename): os.remove(filename)
		self._dataset.to_netcdf(filename, encoding = self._dataset.encoding.update({'zlib': True, 'complevel': complev}))


	def pxform_convert(self, pxform):
     
		pxform = np.array([pxform[0],pxform[1],pxform[2]])
		p = np.append(pxform,[[0,0,0]],0)
		mv = np.random.random()
		p = np.append(p,[[0],[0],[0],[0]], 1)
  
		return p