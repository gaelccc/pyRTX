import numpy as np
from pyRTX.pixelPlaneClass import pixelPlane
import spiceypy as sp
from pyRTX import utils_rt	
from pyRTX.genericClasses import Planet
import timeit
from numba import jit




def circular_mask(R, coords, origin):
	maskIds = []
	maskIds = np.where(np.linalg.norm(coords-origin, axis = 1) <= R)
	

	return np.array(maskIds[0], dtype = 'int32')


def circular_rim(R, coords, origin):
	maskIds = np.where(np.isclose(np.linalg.norm(coords-origin, axis = 1),R, rtol = 0.001))
	mask_coords = coords[maskIds]

	return np.array(maskIds[0], dtype = 'int32')



@jit(nopython=True)
def compute_directions(pixelCoords):
	dirs = np.zeros_like(pixelCoords)
	for i,elem in enumerate(pixelCoords):
		dirs[i] = elem/np.linalg.norm(elem)

	return dirs



@jit(nopython = True)
def compute_beta(coords, origin, R):
	d = np.linalg.norm(origin)
	o = origin / np.linalg.norm(origin)
	betas = np.zeros((len(coords)))
	
	for i,c in enumerate(coords):
		cos = np.dot(c/np.linalg.norm(c), o)
		if cos > 1.0:
			cos = 1.0
		elif cos < -1.0:
			cos = -1.0
		ang = np.arccos(cos)

		
		sinb = d/R*np.sin(ang)
		beta = np.arcsin(sinb)
		betas[i] = beta
		#if np.isnan(beta):
			#print(np.dot(c/np.linalg.norm(c),o))
			#print(sinb)

	return betas

@jit(nopython = True)
def compute_pixel_intensities(beta, Type = 'Standard'):
	if Type == 'Standard':
		a0 = 0.3
		a1 = 0.93
		a2 = -0.23
		intensities = a0 + a1*np.cos(beta) + a2*(np.cos(beta))**2
	elif Type == 'Eddington':
		m = np.cos(beta)
		intensities = 3.0/4* ( 7/12 * 0.5*m * m*(1/3 + 0.5*m)*np.log((1+m)/m))
	

	return intensities

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
		self.pxPlane = pixelPlane(spacecraft = spacecraft,
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

