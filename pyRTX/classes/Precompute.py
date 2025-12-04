import os, sys
import pickle as pkl
import spiceypy as sp
import numpy as np
import xarray as xr

from pyRTX.core.analysis_utils import compute_body_positions, compute_body_states
from pyRTX.constants import au 


class Precompute():
	"""
    A class to perform and store SPICE computations in advance.

    This class allows users to precompute and store various SPICE data, such
    as position and state vectors, and rotation matrices, for a given set of
    epochs. This can significantly speed up computations that require repeated
    calls to SPICE.
	"""	
 
	def __init__(self, epochs: list,):
		"""
        Initializes the Precompute object.

        Parameters
        ----------
        epochs : list
            A list of epochs in TDB seconds past J2000 for which to perform
            the precomputations.
		"""

		# Store inputs
		self._epochs    = epochs
		self._config 	= {}


	def addPosition(self, observer: str, target: str, frame: str, correction: str = 'CN'):
		"""
        Adds a position vector computation to the precomputation list.

        Parameters
        ----------
        observer : str
            The name of the observing body.
        target : str
            The name of the target body.
        frame : str
            The reference frame for the computation.
        correction : str, default='CN'
            The aberration correction to use.
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
        Adds a state vector computation to the precomputation list.

        Parameters
        ----------
        observer : str
            The name of the observing body.
        target : str
            The name of the target body.
        frame : str
            The reference frame for the computation.
        correction : str, default='CN'
            The aberration correction to use.
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
        Adds a rotation matrix computation to the precomputation list.

        Parameters
        ----------
        base_frame : str
            The base reference frame.
        target_frame : str
            The target reference frame.
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
        Adds all necessary computations for solar radiation pressure to the
        precomputation list.

        Parameters
        ----------
        sc : pyRTX.Spacecraft
            The spacecraft object.
        planet : pyRTX.Planet
            The planet object.
        correction : str, default='LT+S'
            The aberration correction to use.
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
        Adds all necessary computations for planetary radiation to the
        precomputation list.

        Parameters
        ----------
        sc : pyRTX.Spacecraft
            The spacecraft object.
        planet : pyRTX.Planet
            The planet object.
        moving_frames : list, optional
            A list of any additional moving frames to precompute.
        correction : str, default='CN'
            The aberration correction to use.
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
        Adds all necessary computations for drag to the precomputation list.

        Parameters
        ----------
        sc : pyRTX.Spacecraft
            The spacecraft object.
        planet_name : str
            The name of the planet.
        moving_frames : list, optional
            A list of any additional moving frames to precompute.
        accel_frame : str, optional
            The reference frame for the acceleration.
        correction : str, default='LT+S'
            The aberration correction to use.
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
        Performs all the precomputations and stores the results in an xarray
        Dataset.
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
        Retrieves a precomputed position vector.

        Parameters
        ----------
        epoch : float
            The epoch for which to retrieve the position.
        observer : str
            The name of the observing body.
        target : str
            The name of the target body.
        frame : str
            The reference frame of the position vector.
        correction : str
            The aberration correction used.

        Returns
        -------
        numpy.ndarray
            The position vector.
		"""
		param = f'{observer} / {target} / {frame} / {correction}'
		return self._dataset.position.sel(time = epoch, pos_param = param).data


	def getState(self, epoch, observer: str, target: str, frame: str, correction: str):
		"""
        Retrieves a precomputed state vector.

        Parameters
        ----------
        epoch : float
            The epoch for which to retrieve the state.
        observer : str
            The name of the observing body.
        target : str
            The name of the target body.
        frame : str
            The reference frame of the state vector.
        correction : str
            The aberration correction used.

        Returns
        -------
        numpy.ndarray
            The state vector.
		"""
		param = f'{observer} / {target} / {frame} / {correction}'
		return self._dataset.state.sel(time = epoch, state_param = param).data


	def getRotation(self, epoch, base_frame: str, target_frame: str):
		"""
        Retrieves a precomputed rotation matrix.

        Parameters
        ----------
        epoch : float
            The epoch for which to retrieve the rotation matrix.
        base_frame : str
            The base reference frame.
        target_frame : str
            The target reference frame.

        Returns
        -------
        numpy.ndarray
            The rotation matrix.
		"""
		param = f'{base_frame} / {target_frame}'
		return self._dataset.rotation.sel(time = epoch, rot_param = param).data


	def getArray(self):
		"""
        Returns the entire xarray Dataset of precomputed values.

        Returns
        -------
        xarray.Dataset
            The Dataset of precomputed values.
		"""
		return self._dataset


	def save(self, filename: str, complev: int = 1):
		"""
        Saves the precomputed data to a NetCDF file.

        Parameters
        ----------
        filename : str
            The path to the output file.
        complev : int, default=1
            The compression level for the output file.
		"""
		if os.path.exists(filename): os.remove(filename)
		self._dataset.to_netcdf(filename, encoding = self._dataset.encoding.update({'zlib': True, 'complevel': complev}))


	def pxform_convert(self, pxform):
		"""
        Converts a 3x3 SPICE rotation matrix to a 4x4 transformation matrix.

        Parameters
        ----------
        pxform : numpy.ndarray
            The 3x3 rotation matrix.

        Returns
        -------
        numpy.ndarray
            The 4x4 transformation matrix.
		"""
     
		pxform = np.array([pxform[0],pxform[1],pxform[2]])
		p = np.append(pxform,[[0,0,0]],0)
		mv = np.random.random()
		p = np.append(p,[[0],[0],[0],[0]], 1)
  
		return p
