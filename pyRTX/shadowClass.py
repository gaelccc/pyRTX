import numpy as np
from pixelPlaneClass import pixelPlane
import spiceypy as sp
import utils_rt	
from genericClasses import Planet


def circular_mask(R, coords, origin):
	maskIds = []
	for i,elem in enumerate(coords):
		if np.linalg.norm(elem - origin) <= R:
			maskIds.append(i)
	

	return np.array(maskIds, dtype = 'int32')


def compute_directions(pixelCoords):
	dirs = np.zeros_like(pixelCoords)
	for i,elem in enumerate(pixelCoords):
		dirs[i] = elem/np.linalg.norm(elem)

	return dirs

def compute_beta(coords, origin, R):
	d = np.linalg.norm(origin)
	betas = np.zeros((len(coords)))
	for i,c in enumerate(coords):
		ang = sp.vsep(c, origin)
		sinb = d/R*np.sin(ang)
		beta = np.arcsin(sinb)
		betas[i] = beta

	return betas


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
				     )

		if isinstance(bodyShape, Planet):
			self.shape = bodyShape

		elif bodyShape is None:
		
			self.shape = Planet(radius = bodyRadius, name = body)

		else: 
			self.shape = Planet(name = body, fromFile = bodyShape, bodyFrame = bodyFrame)



	def compute(self, epochs):

		if not isinstance(epochs, list):
			epochs = [epochs]


		ratios = []
		for epoch in epochs:
			dist = sp.spkezr('Sun', epoch, 'J2000', 'LT+S', self.spacecraft.name)[0][0:3]
			dist = np.sqrt(np.sum(np.array(dist)**2))
			self.pxPlane.d0 = dist

			coords, _ = self.pxPlane.dump(epoch)	
			origin = self.pxPlane.x0

			maskIds = circular_mask(self.sunRadius, -coords, origin)
			newCoord = coords[maskIds]


			if self.limbDarkening is not None:
				betas= compute_beta(-newCoord, origin, self.sunRadius)
				pixelIntensities = compute_pixel_intensities(betas)
				sum_of_weights= np.sum(pixelIntensities)


				self.newCoord = newCoord
				self.betas = betas
				self.pixelIntensities = pixelIntensities


			dirs = compute_directions(newCoord)
			ray_origins = np.zeros_like(dirs)


			bodyPos = sp.spkezr(self.body, epoch, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name)[0][0:3]
			sunPos = sp.spkezr('Sun', epoch, self.spacecraft.base_frame, 'LT+S', self.spacecraft.name)[0][0:3]

	

			#print(sp.vsep(sunPos, [0,0,1])*180.0/np.pi)
			#print(bodyPos/np.linalg.norm(bodyPos), sunPos/np.linalg.norm(sunPos), dirs[0])
			
			shape = self.shape.mesh(translate = bodyPos, epoch = epoch, rotate = self.spacecraft.base_frame)

			_, index_ray, _, _, _, _ = utils_rt.RTXkernel(shape, ray_origins, dirs, kernel = 'Embree', bounces = 1, errorMsg = False)

			if np.shape(index_ray)[0] == 1:
				index_ray = index_ray[0]

			numerator = len(index_ray)
			denominator = len(ray_origins)
			if self.limbDarkening is not None:
				betas= compute_beta(-newCoord, origin, self.sunRadius)
				pixelIntensities = compute_pixel_intensities(betas)
				sum_of_weights= np.sum(pixelIntensities)

				numerator = np.sum(pixelIntensities[index_ray])/sum_of_weights
				denominator = 1

			ratios.append(1-numerator/denominator)

		return ratios

