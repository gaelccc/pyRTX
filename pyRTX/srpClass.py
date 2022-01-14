from pyRTX.physical_utils import compute_srp
import spiceypy as sp
import numpy as np
import timeit
from pyRTX import constants

class solarPressure():
	
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


