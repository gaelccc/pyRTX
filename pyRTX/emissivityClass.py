import numpy as np
import spiceypy as sp
from pyRTX.utils_rt import get_surface_normals_and_face_areas, block_dot
from pyRTX.constants import stefan_boltzmann as sb

#mod test

class Emissivity():

	@classmethod
	def __init__(self, Planet = None, spacecraftName = None, spacecraftFrame = None):

		self.Planet = Planet
		self.scname = spacecraftName
		self.scFrame = spacecraftFrame

	def compute(self, epoch):

		"""
		Compute the fundamental quantities for the emissivity force computation
		returns 
		1) normalized fluxes (i.e. for each face that is responsible for albedo contribution
		cos(theta)*dA/pi/r**2
		2) direction of each ray relative to the SC frame

		"""

		norm_fluxes, scRelative, emiIdxs, faceEmi = self._core_compute(epoch)
		rotMat = self.Planet.rot_toSCframe(epoch, scFrame = self.scFrame)

		dirs_to_sc = np.dot(scRelative, rotMat.T)
		dirs = np.zeros((len(norm_fluxes), 2))
		for i, ddir in enumerate(dirs_to_sc):
			[_, dirs[i,0], dirs[i, 1]] = sp.recrad(ddir)
		
		return norm_fluxes, dirs, faceEmi

	@classmethod
	def _core_compute(self, epoch):
		" Get the rays to be used in the computation "

		V, F, N, C = self.Planet.VFNC(epoch)
		emiIdxs, faceTemps, faceEmi = self.Planet.emissivityFaces(epoch, self.scname)
		scPos = self.Planet.getScPosSunFixed(epoch, self.scname)


		# Get the direction of the rays in the SC frame
		centers = C[emiIdxs]
		scRelative = - centers + scPos
		
		dirs = scRelative / np.linalg.norm(scRelative, axis = 1).reshape(len(scRelative), 1)
		rot = sp.pxform(self.Planet.sunFixedFrame, self.scFrame, epoch)
		sc_dirs = np.dot(dirs, rot.T)

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




		#print(np.linalg.norm(scPos))	
		return norm_fluxes, -scRelative, emiIdxs, faceEmi


		


		



		


