import numpy as np
import spiceypy as sp
import xarray

from pyRTX import constants

from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.classes.Planet import Planet

from pyRTX.core import utils_rt
from pyRTX.core.shadow_utils import circular_mask, circular_rim, compute_directions, compute_beta, compute_pixel_intensities
from pyRTX.core.parallel_utils import parallel
from pyRTX.core.physical_utils import preprocess_RTX_geometry, preprocess_materials
from pyRTX.core.utils_rt import reflected

from scipy import interpolate



class SunShadow():
	"""
	A Class to compute the Solar flux ratio that impacts the spacecraft
	For the moment, limited to airless bodies

	spacecraft [scClass.Spacecraft object]
	body

	"""

	def __init__(self, spacecraft = None, body = None, bodyRadius = None, numrays = 100, sunRadius = 600e3, bodyShape = None, bodyFrame = None, limbDarkening = 'Standard', precomputation = None):
		
		self.sunRadius = sunRadius
		self.spacecraft = spacecraft
		self.body = body
		self.limbDarkening = limbDarkening
		self.pxPlane = PixelPlane(spacecraft = spacecraft,
								  source = 'Sun',
								  mode = 'Dynamic',
								  width = 2*sunRadius,
								  height = 2*sunRadius,
								  ray_spacing = int(2*sunRadius/numrays),
								  units = 'km'
								  )

		if isinstance(bodyShape, Planet):
			self.shape = bodyShape

		elif bodyShape is None:
			self.shape = Planet(radius = bodyRadius, name = body)
   
		else: 
			self.shape = Planet(name = body, fromFile = bodyShape, bodyFrame = bodyFrame)

		self.sp_data = precomputation


	def _store_precomputations(self):
		"""
		Method to store precomputed data.

		Parameters:
		-	sp_data: object of the class Precompute
		"""

		self.shape.sp_data    = self.sp_data
		self.pxPlane.sp_data  = self.sp_data


	@parallel
	def run(self, epoch):
		"""
		Method to compute eclipse ratio at a single epoch.

		Parameters:
		-	epoch: spiceypy epoch
		"""
  
		if self.sp_data != None:
			bodyPos = self.sp_data.getPosition(epoch, self.spacecraft.name, self.body, self.spacecraft.base_frame, 'LT+S')
			sunPos  = self.sp_data.getPosition(epoch, self.spacecraft.name, 'Sun', self.spacecraft.base_frame, 'LT+S')
		else:
			bodyPos = sp.spkezr(self.body, epoch, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name)[0][0:3]
			sunPos  = sp.spkezr('Sun', epoch, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name)[0][0:3]

		dist = np.sqrt(np.sum(np.array(sunPos)**2))
		self.pxPlane.d0 = dist

		coords, _ = self.pxPlane.dump(epoch)
		origin = self.pxPlane.x0

		shape = self.shape.mesh(translate = bodyPos, epoch = epoch, rotate = self.spacecraft.base_frame)

		# Check the circular rim first
		rimIds = circular_rim(self.sunRadius, -coords, origin)
		rimCoords = coords[rimIds]
		rimdirs = compute_directions(rimCoords)
		rim_origins = np.zeros_like(rimdirs)
		_, index_rim, _, _, _, _ = utils_rt.RTXkernel(shape, rim_origins, rimdirs, kernel = 'Embree', bounces = 1, errorMsg = False)

		if len(index_rim[0]) == 0: return 1.0

		maskIds = circular_mask(self.sunRadius, -coords, origin)
		newCoord = coords[maskIds]

		if self.limbDarkening is not None:
			betas= compute_beta(-newCoord, origin, self.sunRadius)
			pixelIntensities = compute_pixel_intensities(betas)
			sum_of_weights= np.sum(pixelIntensities)

		dirs = compute_directions(newCoord)
		ray_origins = np.zeros_like(dirs)

		_, index_ray, _, _, _, _ = utils_rt.RTXkernel(shape, ray_origins, dirs, kernel = 'Embree', bounces = 1, errorMsg = False)

		if np.shape(index_ray)[0] == 1:
			index_ray = index_ray[0]

		numerator = len(index_ray)
		denominator = len(ray_origins)

		# Repeated block!
		#if self.limbDarkening is not None:
		#	betas= compute_beta(-newCoord, origin, self.sunRadius)
		#	pixelIntensities = compute_pixel_intensities(betas)
		#	sum_of_weights= np.sum(pixelIntensities)

		#	numerator = np.sum(pixelIntensities[index_ray])/sum_of_weights
		#	denominator = 1

		if self.limbDarkening is not None:
			numerator = np.sum(pixelIntensities[index_ray])/sum_of_weights
			denominator = 1
			
		return (1-numerator/denominator)


	def compute(self, epochs, n_cores = None):
		"""
		Parameters:
		-	epochs: list of epochs
		-   ncores: number of cores to use for parallel computations
		"""
  
		if not isinstance(epochs, (list, np.ndarray)): epochs = [epochs]
  
		self._store_precomputations()
  
		return self.run(epochs, n_cores = n_cores)




class SolarPressure():
	
	def __init__(self, spacecraft, rayTracer = None, baseflux = 1361.5, grouped = True, shadowObj = None, lookup = None, precomputation = None):

		self.spacecraft = spacecraft
		self.rayTracer  = rayTracer
		self.baseflux   = baseflux
		self.grouped    = grouped
		self.shadowObj  = shadowObj
		self.lookup     = lookup
		self.sp_data    = precomputation
  
		if isinstance(spacecraft.mass, (float,int)): 
				self.scMass = spacecraft.mass
		elif isinstance(spacecraft.mass, xarray.core.dataset.Dataset): 
				mass_times = spacecraft.mass.time.data
				mass_data = spacecraft.mass.mass.data
				self.scMass = interpolate.interp1d(mass_times, mass_data, kind='previous', assume_sorted=True)
		else:
				print('\n *** WARNING: SC mass should be float, int or xarray!')
				self.scMass = None
   
		if self.baseflux is None and (not isinstance(self.scMass, (float,int)) or int(self.scMass) != 1):
			print('\n *** WARNING: For LUT computation SC mass should be set to 1.0!')


	def _store_precomputations(self):
		"""
		Method to store precomputed data.
		"""

		if self.shadowObj is not None and self.shadowObj.sp_data == None:
			self.shadowObj.sp_data = self.sp_data
		if self.rayTracer is not None: 
			self.rayTracer.rays.sp_data = self.sp_data
		if self.lookup is not None: 
			self.lookup.sp_data = self.sp_data
		self.spacecraft.sp_data = self.sp_data


	@parallel
	def run(self, epoch):
		"""
		Method to compute solar pressure acceleration at a single epoch.

		Parameters:
		-	epoch: spiceypy epoch
		"""
  
		# Launch rayTracer
		self.rayTracer.trace(epoch)

		# Retrieve RTX properties
		mesh = self.spacecraft.dump(epoch)
		index_tri = self.rayTracer.index_tri_container
		index_ray = self.rayTracer.index_ray_container
		location = self.rayTracer.locations_container
		ray_origins = self.rayTracer.ray_origins_container
		ray_directions = self.rayTracer.ray_directions_container
		diffusion_pack = self.rayTracer.diffusion_pack
		norm_factor = self.rayTracer.norm_factor
		diffusion = self.rayTracer.diffusion
		num_diffuse = self.rayTracer.num_diffuse
		material_dict = self.spacecraft.materials()	

		# Get the flux
		if self.baseflux is None:
			flux = 1.0
		else:
			flux = self.get_flux(epoch)
		if self.shadowObj is not None:
			shadow = self.shadow_ratios[self._epochs_idxs[epoch]]
			flux = flux*shadow
		
		# Compute solar pressure accel
		mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
		f_srp = np.array(self.get_force(flux, mesh, index_tri, index_ray, location, ray_origins, ray_directions, norm_factor, grouped = self.grouped, materials = material_dict, diffusion = diffusion, num_diffuse = num_diffuse, diffusion_pack = diffusion_pack))

		return f_srp / mass
  
  
	def compute(self, epochs, n_cores = None):
		"""
		Method to compute the solar pressure acceleration.

		Parameters:
		-	epochs: list of epochs
		-   ncores: number of cores to use for parallel computations
		"""
  
		if isinstance(epochs, float): epochs = [epochs]
  
		self._store_precomputations()

		if self.shadowObj is not None:
			self.shadow_ratios = self.shadowObj.compute(epochs, n_cores = n_cores)
			self._epochs_idxs  = {epoch: idx for idx, epoch in enumerate(epochs)}

		return np.array(self.run(epochs, n_cores = n_cores))


	def lookupCompute(self, epochs):
		"""
		Method to compute the solar pressure force with Look Up Table.

		Parameters:
		-	epochs: list of epochs
		"""

		if isinstance(epochs, float): epochs = [epochs]
  
		self._store_precomputations()
  
		acc = np.zeros((len(epochs),3))
  
		for i, epoch in enumerate(epochs):

			# Get solar flux
			eclipse = self.shadowObj.compute(epoch)[0]
			flux    = self.get_flux(epoch) * eclipse 
		
			# Query the look up table
			if self.sp_data != None:
				sundir  = self.sp_data.getPosition(epoch, self.spacecraft.name, 'Sun', self.spacecraft.base_frame, 'LT+S')
			else:
				sundir = sp.spkezr( 'Sun', epoch, self.spacecraft.base_frame,'LT+S', self.spacecraft.name)[0][0:3]
			sundir  = sundir / np.linalg.norm(sundir)
			_, ra, dec = sp.recrad(sundir)
			lll  = self.lookup.query(epoch, ra*180/np.pi, dec*180/np.pi)
			
			# Compute acceleration	
			mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
			acc[i,:] = (flux * lll / mass) 
   
		return acc
 

	def get_flux(self, epoch):
		"""
		Method to get the scaled solar flux.

		Parameters:
		-	epoch: requested epoch
		"""
  
		au = constants.au
  
		if self.sp_data != None:
			sunpos  = self.sp_data.getPosition(epoch, self.spacecraft.name, 'Sun', self.spacecraft.base_frame, 'LT+S')
		else:
			sunpos = sp.spkezr( 'Sun', epoch, self.spacecraft.base_frame,'LT+S', self.spacecraft.name)[0][0:3]
		
		dist = np.sqrt(np.sum(np.array(sunpos)**2))/au

		flux = self.baseflux * (1.0/dist)**2

		return flux


	def get_force(self, flux, mesh_obj, index_tri, index_ray, location, ray_origins, ray_directions, pixel_spacing, materials = 'None', grouped = True,
		diffusion = False, num_diffuse = None, diffusion_pack = None): 
		"""
		Compute the SRP force

		Parameters:
		flux: Solar input flux [W/m^2]
		A: areas of the mesh faces
		s: incident ray directions
		r: reflcted ray directions
		n: normal unit vector to the faces


		"""
		# Compute geometric quantities	
		V, F, N, A = preprocess_RTX_geometry(mesh_obj)

		# Retrieve material properties
		if materials != 'None':
			properties = preprocess_materials(materials)
		else:
			properties = 'None'

		# Automatically get the number of bounces
		n_bounce = len(index_tri)

		if grouped:
			force = np.array([0,0,0], dtype = 'float64')
		else:
			force = []

		for i in range(n_bounce):
			if i == 0:
				flux = np.full(len(ray_directions[i]), flux)
				if diffusion:
					diffusion_pack.append(index_tri[i]) #  Append the emitting triangle indexes
					diffusion_pack.append(flux)  # Append the original flux
					diffusion_pack.append(index_ray[i])   # not needed??

			idx_tri = index_tri[i]
			idx_ray = index_ray[i]
			S = ray_directions[i]

			if i == 1 and diffusion:
				force_temp, flux = self.srp_core(flux, idx_tri, idx_ray, N, S, pixel_spacing, mesh_obj, materials = properties, diffusion = diffusion, num_diffuse = num_diffuse, diffusion_pack = diffusion_pack)
			else:
				force_temp, flux = self.srp_core(flux, idx_tri, idx_ray, N, S, pixel_spacing, mesh_obj, materials = properties)

			if grouped:
				force += force_temp
			else:
				force.append(force_temp)
			
		return force


	def srp_core(self, flux, indexes_tri, indexes_ray, N, S, norm_factor, mesh_obj, materials = 'None', diffusion = False, num_diffuse = None, diffusion_pack = None):
		"""
		Core of SRP computation.
		Highly vectorized version. For explicit algorithm implementation refer to the old version

		Parameters:
		flux: solar flux (float, W/m^2)
		indexes_tri: indexes of intersected triangles
		indexes_ray: indexes of intersecting rays
		N: normals
		S: incident direction vectors
		norm_factor: normalization factor computed from ray spacing (float)
		mesh_obj: trimesh.Trimesh object [Not used for now, will be used when interrogating mesh
					for surface properties]

		Returns:
		force: np.array of SRP force
		"""

		c = constants.c
  
		if isinstance(materials, str) and materials == 'None':
			rho_s = 0.1  #Hardcoded and used just for the dummy case in which the materials are not provided
			rho_d = 0.1  
		else:
			rho_s = materials[:,0][indexes_tri]
			rho_d = materials[:,1][indexes_tri]

		force = np.array([0,0,0], dtype = 'float64')

		counter = 0

		dA = np.ones(len(indexes_ray))/norm_factor
		s = S[indexes_ray]
		n = N[indexes_tri]
		r = reflected(s,n)

		aa = flux[indexes_ray]*dA/c 

		# When using vectorization, this operation must be done through np.multiply operator
		# bb = (s  - rho_s * r - 2.0/3 * rho_d * n)
		#term_2 = np.multiply(r.T, ni*mi).T
		#term_3 = np.multiply(n.T, ni*(1-mi)).T
		term_2 = np.multiply(r.T, rho_s).T
		term_3 = np.multiply(n.T, rho_d).T

		bb = s - term_2 - 2.0/3*term_3 

		forc = np.multiply(bb.T, aa).T

		force = np.sum(forc, axis = 0)
		newFlux = flux[indexes_ray]*rho_s

		# Handle the secondary diffusions
		if diffusion:
			idx_tri_previous = diffusion_pack[4]
			idx_tri_actual = diffusion_pack[0]
			idx_ray = diffusion_pack[1]
			idx_ray_previous = diffusion_pack[6]
			ray_directions = diffusion_pack[2]
			original_flux = diffusion_pack[5]
			rho_d = np.repeat(materials[:,1][idx_tri_previous], num_diffuse, axis = 0)
			
			original_flux = np.repeat(original_flux[idx_ray_previous], num_diffuse, axis = 0)
			flux = original_flux * rho_d / num_diffuse

			new_rho_d = materials[:,1][idx_tri_actual]
			new_rho_s = materials[:,0][idx_tri_actual]
			dA = np.ones(len(idx_ray))/norm_factor
			aa = flux[idx_ray]*dA/c
			s = ray_directions[idx_ray]
			n = N[idx_tri_actual]
			r = reflected(s,n)
			term_2 = np.multiply(r.T, new_rho_s).T
			term_3 = np.multiply(n.T, new_rho_d).T
		
			bb = s - term_2 - 2.0/3*term_3 
		
			forc2 = np.multiply(bb.T, aa).T

			force2 = np.sum(forc2, axis = 0)
			
			force = force + force2

		return force, newFlux


