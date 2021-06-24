from utils_rt import RTXkernel
import timeit

class rayTracer():
	
	def __init__(self, spacecraft, rays, kernel = 'Embree', bounces = 1, diffusion = False, num_diffuse = None):

		self.kernel = kernel
		self.bounces = bounces
		self.diffusion = diffusion
		self.num_diffuse = num_diffuse
		self.rays = rays
		self.spacecraft = spacecraft
		self.norm_factor = rays.norm_factor

		if self.diffusion and num_diffuse is None:
			raise ValueError('The diffusion computation is activated but the number of diffused rays was not specified')
	



	def trace(self, epoch = None):

		mesh_obj = self.spacecraft.dump(epoch)
		ray_origins, ray_directions = self.rays.dump(epoch)


		a,b,c,d,e,f = RTXkernel(mesh_obj, ray_origins, ray_directions, bounces = self.bounces, kernel = self.kernel, diffusion = self.diffusion, num_diffuse = self.num_diffuse)




		self.index_tri_container = a
		self.index_ray_container = b
		self.locations_container = c
		self.ray_origins_container = d
		self.ray_directions_container = e
		self.diffusion_pack = f


	



		


