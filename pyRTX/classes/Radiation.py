import numpy as np
import xarray
import spiceypy as sp
from pyRTX.core.utils_rt import get_surface_normals_and_face_areas, block_dot
from timeit import default_timer as dT
from scipy import interpolate
from pyRTX.core.parallel_utils import parallel
from pyRTX.constants import stefan_boltzmann as sb
from pyRTX.constants import au 
from pyRTX.classes.LookUpTable import LookUpTable
from pyRTX.core.analysis_utils import compute_body_positions

"""
Main class for Albedo computations.
"""

class Albedo():

        def __init__(self, spacecraft, lookup, Planet, precomputation = None, baseflux = 1361.5,):
                """
                Parameters
                ----------
                Planet : pyRTX.classes.Planet
                        The planet object the Albedo is for
                spacecraft : pyRTX.classes.Spacecraft
                        The spacecraft object the Albedo is for
                """
                self.scname   = spacecraft.name
                self.scFrame  = spacecraft.base_frame
                self.Planet   = Planet
                self.sp_data  = precomputation
                self.baseflux = baseflux
                
                if isinstance(spacecraft.mass, (float,int)): 
                        self.scMass = spacecraft.mass
                elif isinstance(spacecraft.mass, xarray.core.dataset.Dataset): 
                        mass_times = spacecraft.mass.time.data
                        mass_data = spacecraft.mass.mass.data
                        self.scMass = interpolate.interp1d(mass_times, mass_data, kind='previous', assume_sorted=True)
                else:
                        print('\n *** WARNING: SC mass should be float, int or xarray!')
                        self.scMass = None
                
                if not isinstance(lookup, LookUpTable):
                        raise TypeError('Error: the input lookup table must be a classes.LookUpTable object')
                self.lookup = lookup

        def _store_precomputations(self):
                """
                Method to store precomputed data.

                Parameters:
                -	sp_data: object of the class Precompute
                """

                self.Planet.sp_data = self.sp_data
                self.lookup.sp_data = self.sp_data
                

        @parallel
        def run(self, epoch):

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

                normFluxes, scRelative, albedoIdxs, albedoVals = self._core_compute(epoch)

                if self.sp_data != None:
                        rotMat = self.sp_data.getRotation(epoch, self.Planet.sunFixedFrame, self.scFrame)
                        rotMat = rotMat[:3,:3]
                        sundir  = self.sp_data.getPosition(epoch, self.Planet.name, 'Sun', self.Planet.bodyFrame, 'CN')
                else:
                        rotMat = self.Planet.rot_toSCframe(epoch, scFrame = self.scFrame)
                        sundir = sp.spkezr( 'Sun', epoch, self.Planet.bodyFrame, 'CN', self.Planet.name)[0][0:3]

                dirs_to_sc = np.dot(scRelative, rotMat.T)
                dirs = np.zeros((len(normFluxes), 2))

                for i, ddir in enumerate(dirs_to_sc):
                        [_, dirs[i,0], dirs[i, 1]] = sp.recrad(ddir)

                # Compute normalized fluxes, directions and albedo values
                lll         = self.lookup.query(epoch, dirs[:,0]*180/np.pi, dirs[:,1]*180/np.pi)
                norm_fluxes = np.expand_dims(normFluxes, axis = 1)
                alb_values  = np.expand_dims(albedoVals, axis = 1)
        
                # Scale flux
                sunflux = self.baseflux * (1.0/(np.linalg.norm(sundir)/au))**2
                
                # Compute acceleration	
                mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
                alb_accel = np.sum(alb_values * sunflux/mass * norm_fluxes * lll, axis = 0) 
 
                return alb_accel, normFluxes, dirs, albedoVals
        

        def compute(self, epochs, n_cores = None):
                """
                Method to perform the computations for albedo acceleration.
                """
                
                self._store_precomputations()
                
                alb_accel   = np.zeros((len(epochs), 3))
                norm_fluxes = [0.] * len(epochs)
                dirs        = [0.] * len(epochs)
                albedoVals  = [0.] * len(epochs)
                
                results = self.run(epochs, n_cores = n_cores)

                for r, result in enumerate(results): 
                        alb_accel[r,:] = result[0]
                        norm_fluxes[r] = result[1]
                        dirs[r]        = result[2]
                        albedoVals[r]  = result[3]
                        
                return alb_accel, norm_fluxes, dirs, albedoVals


        def _core_compute(self, epoch):
                " Get the rays to be used in the computation "

                V, F, N, C = self.Planet.VFNC(epoch)

                albedoIdxs, albedoVals = self.Planet.albedoFaces(epoch, self.scname)

                if self.sp_data != None:
                        scPos = self.sp_data.getPosition(epoch, self.Planet.name, self.scname, self.Planet.sunFixedFrame, 'CN')
                else:
                        scPos = self.Planet.getScPosSunFixed(epoch, self.scname)

                # Get the direction of the rays in the SC frame
                centers = C[albedoIdxs]
                scRelative = - centers + scPos
                dirs = scRelative / np.linalg.norm(scRelative, axis = 1).reshape(len(scRelative), 1)

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

        def __init__(self, spacecraft, lookup, Planet, precomputation = None, baseflux = 1361.5):
                """
                Parameters
                ----------
                Planet : pyRTX.classes.Planet
                        The planet object the Albedo is for
                spacecraft : pyRTX.classes.Spacecraft
                        The spacecraft object the Albedo is for
                """
                self.scname   = spacecraft.name
                self.scFrame  = spacecraft.base_frame
                self.Planet   = Planet
                self.sp_data  = precomputation
                self.baseflux = baseflux
                
                if isinstance(spacecraft.mass, (float,int)): 
                        self.scMass = spacecraft.mass
                elif isinstance(spacecraft.mass, xarray.core.dataset.Dataset): 
                        mass_times = spacecraft.mass.time.data
                        mass_data = spacecraft.mass.mass.data
                        self.scMass = interpolate.interp1d(mass_times, mass_data, kind='previous', assume_sorted=True)
                else:
                        print('\n *** WARNING: SC mass should be float, int or xarray!')
                        self.scMass = None
                
                if not isinstance(lookup, LookUpTable):
                        raise TypeError('Error: the input lookup table must be a classes.LookUpTable object')
                self.lookup = lookup


        def _store_precomputations(self):
                """
                Method to store precomputed data.

                Parameters:
                -	sp_data: object of the class Precompute
                """

                self.Planet.sp_data = self.sp_data
                self.lookup.sp_data = self.sp_data

                
        @parallel   
        def run(self, epoch):

                """
                Compute the fundamental quantities for the emissivity force computation
                returns 
                1) normalized fluxes (i.e. for each face that is responsible for albedo contribution
                cos(theta)*dA/pi/r**2
                2) direction of each ray relative to the SC frame

                """

                normFluxes, scRelative, emiIdxs, faceEmi = self._core_compute(epoch)
                
                if self.sp_data != None:
                        rotMat = self.sp_data.getRotation(epoch, self.Planet.sunFixedFrame, self.scFrame)
                        rotMat = rotMat[:3,:3]
                else:
                        rotMat = self.Planet.rot_toSCframe(epoch, scFrame = self.scFrame)

                dirs_to_sc = np.dot(scRelative, rotMat.T)
                
                dirs = np.zeros((len(normFluxes), 2))
                
                for i, ddir in enumerate(dirs_to_sc):
                        [_, dirs[i,0], dirs[i, 1]] = sp.recrad(ddir)
                        
                lll         = self.lookup.query(epoch, dirs[:,0]*180/np.pi, dirs[:,1]*180/np.pi)
                norm_fluxes = np.expand_dims(normFluxes, axis = 1)
                emi_values  = np.expand_dims(faceEmi, axis = 1)
 
                # Compute acceleration	
                mass = self.scMass(epoch) if not isinstance(self.scMass, (float,int)) else self.scMass
                ir_accel = np.sum(emi_values * 1/mass * norm_fluxes * lll, axis = 0) 
                
                return ir_accel, normFluxes, dirs, faceEmi


        def compute(self, epochs, n_cores = None):
                """
                Method to perform the computations for albedo acceleration.
                """
                self._store_precomputations()

                ir_accel    = np.zeros((len(epochs), 3))
                norm_fluxes = [0.] * len(epochs)
                dirs        = [0.] * len(epochs)
                faceEmi     = [0.] * len(epochs)
                               
                results     = self.run(epochs, n_cores = n_cores)

                for r, result in enumerate(results): 
                        ir_accel[r,:]  = result[0]
                        norm_fluxes[r] = result[1]
                        dirs[r]        = result[2]
                        faceEmi[r]     = result[3]
                
                return ir_accel, norm_fluxes, dirs, faceEmi
        
        
        def _core_compute(self, epoch):
                " Get the rays to be used in the computation "

                V, F, N, C = self.Planet.VFNC(epoch)
                
                emiIdxs, faceTemps, faceEmi = self.Planet.emissivityFaces(epoch, self.scname)

                if self.sp_data != None:
                        scPos = self.sp_data.getPosition(epoch, self.Planet.name, self.scname, self.Planet.sunFixedFrame, 'CN')
                else:
                        scPos = self.Planet.getScPosSunFixed(epoch, self.scname)
                
                # Get the direction of the rays in the SC frame
                centers = C[emiIdxs]
                scRelative = - centers + scPos
                
                dirs = scRelative / np.linalg.norm(scRelative, axis = 1).reshape(len(scRelative), 1)

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
                 
                return norm_fluxes, -scRelative, emiIdxs, faceEmi

