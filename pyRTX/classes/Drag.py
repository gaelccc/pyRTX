import xarray
import numpy as np
import spiceypy as sp

from pyRTX.classes.LookUpTable import LookUpTable
from pyRTX.core.parallel_utils import parallel
from scipy import interpolate

class Drag():
	"""
    A class for computing the drag acceleration on a spacecraft.
	"""
    
	def __init__(self, spacecraft, crossectionLUT, density, CD, body, precomputation = None,):
		"""
        Initializes the Drag object.

        Parameters
        ----------
        spacecraft : pyRTX.Spacecraft
            The spacecraft object.
        crossectionLUT : pyRTX.classes.LookUpTable
            A lookup table of the spacecraft's cross-sectional area.
        density : callable
            A function that returns the atmospheric density in kg/km^3 as a
            function of altitude.
        CD : float
            The drag coefficient of the spacecraft.
        body : str
            The name of the celestial body causing the drag.
        precomputation : pyRTX.classes.Precompute, optional
            A Precompute object with precomputed SPICE data.
		"""

		self.scName = spacecraft.name
		self.scFrame = spacecraft.base_frame
  
		self.sp_data  = precomputation
                  
		if not isinstance(crossectionLUT, LookUpTable):
			raise TypeError('Error: the input lookup table must be a classes.LookUpTable object')
		self.LUT = crossectionLUT

		if not callable(density):
			raise TypeError('Error: the input density must be a callable function')

		self.density = density
		self.body = body
		self.CD = CD

		if isinstance(spacecraft.mass, (float,int)): 
				self.scMass = spacecraft.mass
		elif isinstance(spacecraft.mass, xarray.core.dataset.Dataset): 
				mass_times = spacecraft.mass.time.data
				mass_data = spacecraft.mass.mass.data
				self.scMass = interpolate.interp1d(mass_times, mass_data, kind='previous', assume_sorted=True)
		else:
				print('\n *** WARNING: SC mass should be float, int or xarray!')
				self.scMass = None


	def _store_precomputations(self):
		"""
        Stores precomputed SPICE data in the lookup table object.
		"""

		self.LUT.sp_data = self.sp_data
			

	def compute(self, epochs, frame = '', n_cores = None):
		"""
        Computes the drag acceleration for a series of epochs.

        Parameters
        ----------
        epochs : list of float
            A list of epochs in TDB seconds past J2000.
        frame : str, optional
            The reference frame for the output acceleration. If not specified,
            the body-fixed frame of the celestial body is used.
        n_cores : int, optional
            The number of CPU cores to use for parallel computation.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays: the drag acceleration vectors
            and the relative velocity vectors.
		"""
  
		# if isinstance(epochs, float): epochs = [epochs]
		
		if frame == '':
			self.frame = 'IAU_%s' %self.body.upper()
		else:
			self.frame = frame
  
		self._store_precomputations()

		drag   = np.zeros((len(epochs), 3))
		vel_r  = np.zeros((len(epochs), 3))
						
		results = self.run(epochs, n_cores = n_cores)

		for r, result in enumerate(results): 
				drag[r,:]  = result[0]
				vel_r[r,:] = result[1]

		return drag, vel_r


	@parallel
	def run(self, epoch):
		"""
        Computes the drag acceleration at a single epoch.

        Parameters
        ----------
        epoch : float
            The epoch in TDB seconds past J2000.

        Returns
        -------
        tuple
            A tuple containing the drag acceleration vector and the relative
            velocity vector.
		"""

		if self.sp_data != None:
			rot = self.sp_data.getRotation(epoch, self.frame, self.scFrame)
			rot = rot[:3,:3]
			st = self.sp_data.getState(epoch, self.body, self.scName, self.frame, 'LT+S')
		else:
			rot = sp.pxform(self.frame, self.scFrame, epoch)        
			st = sp.spkezr(self.scName, epoch, self.frame, 'LT+S', self.body)[0]

		st_r = rot@st[0:3]
		vel_r = rot@st[3:6]

		_, dir1, dir2 = sp.recrad(-vel_r)
		h = np.linalg.norm(st[0:3])
		unitv = vel_r/np.linalg.norm(vel_r)
		rho = self.density(h) * 1e9		# kg/km**3

		A = self.LUT.query(epoch, dir1*180/np.pi, dir2*180/np.pi)   # km**2
  
		# Compute acceleration	
		mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
		drag = - 0.5*rho*self.CD*np.linalg.norm(vel_r)**2*unitv*A/mass	# km/s**2

		if self.frame != '' and self.frame.lower() != self.scFrame:
			rot = rot.T
			drag = rot@drag

		return drag, vel_r
