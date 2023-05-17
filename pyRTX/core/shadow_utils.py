import numpy as np 
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