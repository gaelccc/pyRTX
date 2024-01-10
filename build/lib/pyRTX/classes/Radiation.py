import numpy as np
import spiceypy as sp
from pyRTX.core.utils_rt import get_surface_normals_and_face_areas, block_dot
from timeit import default_timer as dT
from pyRTX.constants import stefan_boltzmann as sb
"""
Main class for Albedo computations.
"""
class Albedo():

        def __init__(self, Planet = None, spacecraftName = None, spacecraftFrame = None):
                """
                Parameters
                ----------
                Planet : pyRTX.classes.Planet
                        The planet object the Albedo is for
                spacecraftName : str
                        The name of the spacecraft
                spacecraftFrame : str
                        The name of the spacecraft body fixed frame
                """
                self.Planet = Planet
                self.scname = spacecraftName
                self.scFrame = spacecraftFrame



        def compute(self, epoch):

                """
                Compute the fundamental quantities for the albedo force computation
                Parameters
                ----------
                epoch : str or float
                        the epoch at which to compute the albedo (it can be either a string or a float)
                Returns
                -------
                normalized_fluxes : np.ndarray
                        (i.e. for each face that is responsible for albedo contribution  cos(alpha)*cos(theta)*dA/pi/r**2
                dirs: np.ndarray
                        direction of each ray relative to the SC frame
                vals : np.array
                        values of the albedo for each face of the planet

                """

                norm_fluxes, scRelative, albedoIdxs, albedoVals = self._core_compute(epoch)
                rotMat = self.Planet.rot_toSCframe(epoch, scFrame = self.scFrame)

                dirs_to_sc = np.dot(scRelative, rotMat.T)
                dirs = np.zeros((len(norm_fluxes), 2))
                

                for i, ddir in enumerate(dirs_to_sc):
                        [_, dirs[i,0], dirs[i, 1]] = sp.recrad(ddir)

                return norm_fluxes, dirs, albedoVals #self.Planet.albedo[albedoIdxs]




        def _core_compute(self, epoch):
                " Get the rays to be used in the computation "

                V, F, N, C = self.Planet.VFNC(epoch)

                albedoIdxs, albedoVals = self.Planet.albedoFaces(epoch, self.scname)

                scPos = self.Planet.getScPosSunFixed(epoch, self.scname)



                # Get the direction of the rays in the SC frame
                centers = C[albedoIdxs]
                scRelative = - centers + scPos
                dirs = scRelative / np.linalg.norm(scRelative, axis = 1).reshape(len(scRelative), 1)
                rot = sp.pxform(self.Planet.sunFixedFrame, self.scFrame, epoch)
                sc_dirs = np.dot(dirs, rot.T)


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


                

                



                


