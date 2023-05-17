import numpy as np
import spiceypy as sp

from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.core import utils_rt
from pyRTX.core.shadow_utils import circular_mask, circular_rim, compute_directions, compute_beta, compute_pixel_intensities
from pyRTX.classes.Planet import Planet
from pyRTX import constants
from pyRTX.core.physical_utils import compute_srp






class SunShadow():
	"""
	A Class to compute the Solar flux ratio that impacts the spacecraft
	For the moment, limited to airless bodies

	spacecraft [scClass.Spacecraft object]
	body

	"""

	def __init__(self, spacecraft = None, body = None, bodyRadius = None, numrays = 100, sunRadius = 600e3, bodyShape = None, bodyFrame = None, limbDarkening = 'Standard'):
		
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



	def compute(self, epochs):

		

		if not isinstance(epochs, (list, np.ndarray)):
			epochs = [epochs]


		ratios = []
		bodyPos = sp.spkezr(self.body, epochs, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name)[0]
		for i,epoch in enumerate(epochs):
			

			dist = sp.spkezr('Sun', epoch, 'J2000', 'LT+S', self.spacecraft.name)[0][0:3]
			dist = np.sqrt(np.sum(np.array(dist)**2))
			self.pxPlane.d0 = dist

			coords, _ = self.pxPlane.dump(epoch)
			origin = self.pxPlane.x0


			shape = self.shape.mesh(translate = bodyPos[i][0:3], epoch = epoch, rotate = self.spacecraft.base_frame)


			# Check the circular rim first
			rimIds = circular_rim(self.sunRadius, -coords, origin)
			rimCoords = coords[rimIds]
			rimdirs = compute_directions(rimCoords)
			rim_origins = np.zeros_like(rimdirs)
			_, index_rim, _, _, _, _ = utils_rt.RTXkernel(shape, rim_origins, rimdirs, kernel = 'Embree', bounces = 1, errorMsg = False)

			
			if len(index_rim[0]) == 0:
				ratios.append(1.0)
				continue



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
				


			ratios.append(1-numerator/denominator)


		return ratios







class SolarPressure():
	
	def __init__(self, spacecraft, rayTracer, baseflux = 1380.0, grouped = True, shadowObj = None):

		self.spacecraft = spacecraft
		self.rayTracer = rayTracer
		self.baseflux = baseflux
		self.grouped = grouped
		self.shadowObj = shadowObj
		



	def compute(self, epoch = None):

		# Launch rayTracer
		rtx = self.rayTracer
		rtx.trace(epoch)


		# Retrieve RTX  properties
		mesh = self.spacecraft.dump(epoch)


		index_tri = rtx.index_tri_container
		index_ray = rtx.index_ray_container
		location = rtx.locations_container
		ray_origins = rtx.ray_origins_container
		ray_directions = rtx.ray_directions_container
		diffusion_pack = rtx.diffusion_pack
		norm_factor = rtx.norm_factor
		diffusion = rtx.diffusion
		num_diffuse = rtx.num_diffuse



		

		material_dict = self.spacecraft.materials()

		if self.baseflux is None:
			flux = 1.0
		else:
                        flux = self.get_flux( epoch)

		if self.shadowObj is not None:
			shadow = self.shadowObj.compute(epoch)[0]
			flux = flux*shadow
		



		force = compute_srp(flux, mesh, index_tri, index_ray, location, ray_origins, ray_directions, norm_factor, grouped = self.grouped, materials = material_dict, diffusion = diffusion, num_diffuse = num_diffuse, diffusion_pack = diffusion_pack)




		return force



	def get_flux(self, epoch):
		
		au = constants.au
		sunpos = sp.spkezr( 'Sun', epoch, 'J2000','LT+S', self.spacecraft.name)
		pos = sunpos[0][0:3]
		dist = np.sqrt(np.sum(np.array(pos)**2))/au

		flux = self.baseflux * (1.0/dist)**2

		return flux


