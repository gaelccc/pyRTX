import numpy as np
import xarray
import spiceypy as sp
from pyRTX.core.utils_rt import get_surface_normals_and_face_areas, block_dot
from timeit import default_timer as dT
from scipy import interpolate
from pyRTX.core.parallel_utils import parallel
from pyRTX.constants import stefan_boltzmann as sb
from pyRTX.constants import au 
from pyRTX.classes.LookUpTable import LookUpTable
from pyRTX.core.analysis_utils import compute_body_positions

"""
Main class for Albedo computations.
"""

class Albedo():
	"""
    A class for computing the albedo acceleration on a spacecraft.
	"""

	def __init__(self, spacecraft, lookup, Planet, precomputation = None, baseflux = 1361.5,):
		"""
        Initializes the Albedo object.

        Parameters
        ----------
        spacecraft : pyRTX.Spacecraft
            The spacecraft object.
        lookup : pyRTX.classes.LookUpTable
            A lookup table for the spacecraft's albedo properties.
        Planet : pyRTX.Planet
            The planet object.
        precomputation : pyRTX.classes.Precompute, optional
            A Precompute object with precomputed SPICE data.
        baseflux : float, default=1361.5
            The base solar flux at 1 AU in W/m^2.
		"""
		self.scname   = spacecraft.name
		self.scFrame  = spacecraft.base_frame
		self.Planet   = Planet
		self.sp_data  = precomputation
		self.baseflux = baseflux

		if isinstance(spacecraft.mass, (float,int)):
			self.scMass = spacecraft.mass
		elif isinstance(spacecraft.mass, xarray.core.dataset.Dataset):
			mass_times = spacecraft.mass.time.data
			mass_data = spacecraft.mass.mass.data
			self.scMass = interpolate.interp1d(mass_times, mass_data, kind='previous', assume_sorted=True)
		else:
			print('\n *** WARNING: SC mass should be float, int or xarray!')
			self.scMass = None

		if not isinstance(lookup, LookUpTable):
			raise TypeError('Error: the input lookup table must be a classes.LookUpTable object')
		self.lookup = lookup

	def _store_precomputations(self):
		"""
        Stores precomputed SPICE data in the Planet and lookup table objects.
		"""

		self.Planet.sp_data = self.sp_data
		self.lookup.sp_data = self.sp_data


	@parallel
	def run(self, epoch):
		"""
        Computes the albedo acceleration at a single epoch.

        Parameters
        ----------
        epoch : float
            The epoch in TDB seconds past J2000.

        Returns
        -------
        tuple
            A tuple containing the albedo acceleration vector, the normalized
            fluxes, the directions of the rays, and the albedo values.
		"""

		normFluxes, scRelative, albedoIdxs, albedoVals = self._core_compute(epoch)

		if self.sp_data != None:
			rotMat = self.sp_data.getRotation(epoch, self.Planet.sunFixedFrame, self.scFrame)
			rotMat = rotMat[:3,:3]
			sundir  = self.sp_data.getPosition(epoch, self.Planet.name, 'Sun', self.Planet.bodyFrame, 'CN')
		else:
			rotMat = self.Planet.rot_toSCframe(epoch, scFrame = self.scFrame)
			sundir = sp.spkezr( 'Sun', epoch, self.Planet.bodyFrame, 'CN', self.Planet.name)[0][0:3]

		dirs_to_sc = np.dot(scRelative, rotMat.T)
		dirs = np.zeros((len(normFluxes), 2))

		for i, ddir in enumerate(dirs_to_sc):
			[_, dirs[i,0], dirs[i, 1]] = sp.recrad(ddir)

		# Compute normalized fluxes, directions and albedo values
		lll         = self.lookup.query(epoch, dirs[:,0]*180/np.pi, dirs[:,1]*180/np.pi)
		norm_fluxes = np.expand_dims(normFluxes, axis = 1)
		alb_values  = np.expand_dims(albedoVals, axis = 1)

		# Scale flux
		sunflux = self.baseflux * (1.0/(np.linalg.norm(sundir)/au))**2

		# Compute acceleration
		mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
		alb_accel = np.sum(alb_values * sunflux/mass * norm_fluxes * lll, axis = 0)

		return alb_accel, normFluxes, dirs, albedoVals


	def compute(self, epochs, n_cores = None):
		"""
        Computes the albedo acceleration for a series of epochs.

        Parameters
        ----------
        epochs : list of float
            A list of epochs in TDB seconds past J2000.
        n_cores : int, optional
            The number of CPU cores to use for parallel computation.

        Returns
        -------
        tuple
            A tuple containing the albedo acceleration vectors, the
            normalized fluxes, the directions of the rays, and the albedo
            values for each epoch.
		"""

		self._store_precomputations()

		alb_accel   = np.zeros((len(epochs), 3))
		norm_fluxes = [0.] * len(epochs)
		dirs        = [0.] * len(epochs)
		albedoVals  = [0.] * len(epochs)

		results = self.run(epochs, n_cores = n_cores)

		for r, result in enumerate(results):
			alb_accel[r,:] = result[0]
			norm_fluxes[r] = result[1]
			dirs[r]        = result[2]
			albedoVals[r]  = result[3]

		return alb_accel, norm_fluxes, dirs, albedoVals


	def _core_compute(self, epoch):
		"""
        Core computation for the albedo acceleration.

        Parameters
        ----------
        epoch : float
            The epoch in TDB seconds past J2000.

        Returns
        -------
        tuple
            A tuple containing the normalized fluxes, the spacecraft-relative
            positions, the albedo indices, and the albedo values.
		"""
		" Get the rays to be used in the computation "

		V, F, N, C = self.Planet.VFNC(epoch)

		albedoIdxs, albedoVals = self.Planet.albedoFaces(epoch, self.scname)

		if self.sp_data != None:
			scPos = self.sp_data.getPosition(epoch, self.Planet.name, self.scname, self.Planet.sunFixedFrame, 'CN')
		else:
			scPos = self.Planet.getScPosSunFixed(epoch, self.scname)

		# Get the direction of the rays in the SC frame
		centers = C[albedoIdxs]
		scRelative = - centers + scPos
		dirs = scRelative / np.linalg.norm(scRelative, axis = 1).reshape(len(scRelative), 1)

		# Get normal-to-spacecraft angles
		normals = N[albedoIdxs]
		cos_theta = block_dot(normals, dirs)

		# Get sun-to-normal angles
		cos_alpha = np.where(normals[:,0]>0, normals[:,0], 0)  # The sun is in the x direction. This is equivalent to dot(normals, [1,0,0])

		# Distance between sc and each element mesh
		scRelativeMag = np.sum(np.array(scRelative)**2, axis = 1)

		# Compute the geometric contribution to the flux
		_, dA = get_surface_normals_and_face_areas(V, F)
		dA = dA[albedoIdxs]

		norm_fluxes = cos_alpha * cos_theta * dA / np.pi / scRelativeMag


		return norm_fluxes, -scRelative, albedoIdxs, albedoVals


class Emissivity():
	"""
    A class for computing the thermal emissivity acceleration on a spacecraft.
	"""

	def __init__(self, spacecraft, lookup, Planet, precomputation = None, baseflux = 1361.5):
		"""
        Initializes the Emissivity object.

        Parameters
        ----------
        spacecraft : pyRTX.Spacecraft
            The spacecraft object.
        lookup : pyRTX.classes.LookUpTable
            A lookup table for the spacecraft's emissivity properties.
        Planet : pyRTX.Planet
            The planet object.
        precomputation : pyRTX.classes.Precompute, optional
            A Precompute object with precomputed SPICE data.
        baseflux : float, default=1361.5
            The base solar flux at 1 AU in W/m^2.
		"""
		self.scname   = spacecraft.name
		self.scFrame  = spacecraft.base_frame
		self.Planet   = Planet
		self.sp_data  = precomputation
		self.baseflux = baseflux

		if isinstance(spacecraft.mass, (float,int)):
			self.scMass = spacecraft.mass
		elif isinstance(spacecraft.mass, xarray.core.dataset.Dataset):
			mass_times = spacecraft.mass.time.data
			mass_data = spacecraft.mass.mass.data
			self.scMass = interpolate.interp1d(mass_times, mass_data, kind='previous', assume_sorted=True)
		else:
			print('\n *** WARNING: SC mass should be float, int or xarray!')
			self.scMass = None

		if not isinstance(lookup, LookUpTable):
			raise TypeError('Error: the input lookup table must be a classes.LookUpTable object')
		self.lookup = lookup


	def _store_precomputations(self):
		"""
        Stores precomputed SPICE data in the Planet and lookup table objects.
		"""

		self.Planet.sp_data = self.sp_data
		self.lookup.sp_data = self.sp_data


	@parallel
	def run(self, epoch):
		"""
        Computes the emissivity acceleration at a single epoch.

        Parameters
        ----------
        epoch : float
            The epoch in TDB seconds past J2000.

        Returns
        -------
        tuple
            A tuple containing the emissivity acceleration vector, the
            normalized fluxes, the directions of the rays, and the face
            emissivities.
		"""

		normFluxes, scRelative, emiIdxs, faceEmi = self._core_compute(epoch)

		if self.sp_data != None:
			rotMat = self.sp_data.getRotation(epoch, self.Planet.sunFixedFrame, self.scFrame)
			rotMat = rotMat[:3,:3]
		else:
			rotMat = self.Planet.rot_toSCframe(epoch, scFrame = self.scFrame)

		dirs_to_sc = np.dot(scRelative, rotMat.T)

		dirs = np.zeros((len(normFluxes), 2))

		for i, ddir in enumerate(dirs_to_sc):
			[_, dirs[i,0], dirs[i, 1]] = sp.recrad(ddir)

		lll         = self.lookup.query(epoch, dirs[:,0]*180/np.pi, dirs[:,1]*180/np.pi)
		norm_fluxes = np.expand_dims(normFluxes, axis = 1)
		emi_values  = np.expand_dims(faceEmi, axis = 1)

		# Compute acceleration
		mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
		ir_accel = np.sum(emi_values * 1/mass * norm_fluxes * lll, axis = 0)

		return ir_accel, normFluxes, dirs, faceEmi


	def compute(self, epochs, n_cores = None):
		"""
        Computes the emissivity acceleration for a series of epochs.

        Parameters
        ----------
        epochs : list of float
            A list of epochs in TDB seconds past J2000.
        n_cores : int, optional
            The number of CPU cores to use for parallel computation.

        Returns
        -------
        tuple
            A tuple containing the emissivity acceleration vectors, the
            normalized fluxes, the directions of the rays, and the face
            emissivities for each epoch.
		"""
		self._store_precomputations()

		ir_accel    = np.zeros((len(epochs), 3))
		norm_fluxes = [0.] * len(epochs)
		dirs        = [0.] * len(epochs)
		faceEmi     = [0.] * len(epochs)

		results     = self.run(epochs, n_cores = n_cores)

		for r, result in enumerate(results):
			ir_accel[r,:]  = result[0]
			norm_fluxes[r] = result[1]
			dirs[r]        = result[2]
			faceEmi[r]     = result[3]

		return ir_accel, norm_fluxes, dirs, faceEmi


	def _core_compute(self, epoch):
		"""
        Core computation for the emissivity acceleration.

        Parameters
        ----------
        epoch : float
            The epoch in TDB seconds past J2000.

        Returns
        -------
        tuple
            A tuple containing the normalized fluxes, the spacecraft-relative
            positions, the emissivity indices, and the face emissivities.
		"""
		" Get the rays to be used in the computation "

		V, F, N, C = self.Planet.VFNC(epoch)

		emiIdxs, faceTemps, faceEmi = self.Planet.emissivityFaces(epoch, self.scname)

		if self.sp_data != None:
			scPos = self.sp_data.getPosition(epoch, self.Planet.name, self.scname, self.Planet.sunFixedFrame, 'CN')
		else:
			scPos = self.Planet.getScPosSunFixed(epoch, self.scname)

		# Get the direction of the rays in the SC frame
		centers = C[emiIdxs]
		scRelative = - centers + scPos

		dirs = scRelative / np.linalg.norm(scRelative, axis = 1).reshape(len(scRelative), 1)

		# Get normal-to-spacecraft angles
		normals = N[emiIdxs]
		#cos_theta = np.dot(normals, scPos/np.linalg.norm(scPos))
		cos_theta = block_dot(normals, dirs)

		# Distance between sc and each element mesh
		scRelativeMag = np.sum(np.array(scRelative)**2, axis=1)

		# Compute the geometric contribution to the flux
		_, dA = get_surface_normals_and_face_areas(V, F)
		dA = dA[emiIdxs]

		norm_fluxes =  sb * faceTemps**4 * cos_theta * dA / np.pi / scRelativeMag

		return norm_fluxes, -scRelative, emiIdxs, faceEmi
