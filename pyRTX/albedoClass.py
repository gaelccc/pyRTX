import numpy as np
import spiceypy as sp
from pyRTX.utils_rt import get_surface_normals_and_face_areas, block_dot
from timeit import default_timer as dT

class Albedo():

        def __init__(self, Planet = None, spacecraftName = None, spacecraftFrame = None):
                self.Planet = Planet
                self.scname = spacecraftName
                self.scFrame = spacecraftFrame



        def compute(self, epoch):

                """
                Compute the fundamental quantities for the albedo force computation
                returns 
                1) normalized fluxes (i.e. for each face that is responsible for albedo contribution
                cos(alpha)*cos(theta)*dA/pi/r**2
                2) direction of each ray relative to the SC frame

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

                


                



                


