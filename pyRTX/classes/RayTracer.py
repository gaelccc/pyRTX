from pyRTX.core.utils_rt import RTXkernel
import timeit

class RayTracer():
	"""
    A class for performing ray-tracing simulations.

    This class orchestrates the ray-tracing process by taking a spacecraft
    model and a set of rays (defined by a PixelPlane object) and using a
    specified ray-tracing kernel to compute the intersections.
	"""
	
	def __init__(self, spacecraft, rays, kernel = 'Embree', bounces = 1, diffusion = False, num_diffuse = None):
		"""
        Initializes the RayTracer object.

        Parameters
        ----------
        spacecraft : pyRTX.Spacecraft
            The spacecraft object to be ray-traced.
        rays : pyRTX.PixelPlane
            The PixelPlane object defining the rays.
        kernel : str, default='Embree'
            The ray-tracing kernel to use (e.g., 'Embree').
        bounces : int, default=1
            The number of bounces to simulate for each ray.
        diffusion : bool, default=False
            Whether to simulate diffuse reflections.
        num_diffuse : int, optional
            The number of diffuse rays to cast from each intersection point.
            Required if `diffusion` is True.
		"""

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
		"""
        Performs the ray-tracing simulation.

        This method computes the intersections of the rays with the spacecraft
        mesh at a given epoch and stores the results in the object's
        attributes.

        Parameters
        ----------
        epoch : float, optional
            The epoch in TDB seconds past J2000 for which to perform the
            ray-tracing. This is necessary if the spacecraft's geometry or
            orientation is time-dependent.
		"""

		mesh_obj = self.spacecraft.dump(epoch)
		ray_origins, ray_directions = self.rays.dump(epoch)
		
		#if self.rays.packets != 1:

		#	for i in range(self.rays.packets):
		#		
		#		a,b,c,d,e,f = RTXkernel(mesh_obj, ray_origins[i], ray_directions[i], bounces = self.bounces, kernel = self.kernel, diffusion = self.diffusion, num_diffuse = self.num_diffuse)


		#		if i == 0:
		#			atemp = a
		#			btemp = b
		#			ctemp = c
		#			dtemp = d
		#			etemp = e
		#			ftemp = f

		#		np.hstack((a, atemp))
		#		np.hstack((b, btemp))
		#		np.hstack((c, ctemp))
		#		np.hstack((d, dtemp))
		#		np.hstack((e, etemp))
		#		if not f is None:
		#			ftemp.extend(f)
		#		else: 
		#			ftemp = None
		#	
		#	a = atemp
		#	b = btemp
		#	c = ctemp
		#	d = dtemp
		#	e = etemp
		#	f = ftemp



		#else:
		a,b,c,d,e,f = RTXkernel(mesh_obj, ray_origins, ray_directions, bounces = self.bounces, kernel = self.kernel, diffusion = self.diffusion, num_diffuse = self.num_diffuse)

		self.index_tri_container = a
		self.index_ray_container = b
		self.locations_container = c
		self.ray_origins_container = d
		self.ray_directions_container = e
		self.diffusion_pack = f
