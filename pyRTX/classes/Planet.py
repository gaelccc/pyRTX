import copy, sys
import trimesh as tm
import trimesh.transformations as tmt
import spiceypy as sp
import numpy as np
import xarray as xr

from mpl_toolkits.basemap import Basemap

from scipy import interpolate
from copy import deepcopy

from pyRTX.core.utils_rt import get_centroids, block_normalize
from pyRTX.core.analysis_utils import computeRADEC
from pyRTX import constants
from pyRTX.core.analysis_utils import TiffInterpolator
from pyRTX.constants import au
from pyRTX.constants import stefan_boltzmann as sb

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cmx


class PlanetGrid():
	"""
    A base class for representing planetary grids, such as albedo, emissivity,
    and temperature maps.
	"""
    
    
	def __init__(self,):
		"""
		Initializes the PlanetGrid object.
		"""
		self._albedo          =  None
		self._base_flux       =  None
		self._radius          =  None
		self._frame           =  None
		self._planet_name     =  None
		self._nightside_temp  =  None
		self._emissivity      =  None
		self._property        =  None
		self._axes            =  None
		self._dims   		  =  None
  
	@property
	def attrs(self):
		"""
        Returns the attributes of the underlying xarray Dataset.

        Returns
        -------
        dict
            The attributes of the xarray Dataset.
		"""
		return self._data.attrs

	@property
	def axes(self):
		"""
        Returns the coordinate axes of the grid.

        Returns
        -------
        list
            A list of the grid's coordinate axes.
		"""
		return self._axes

	@property
	def frame(self):
		"""
        Returns the SPICE reference frame of the grid.

        Returns
        -------
        str
            The name of the SPICE reference frame.
		"""
		return self._frame

	@property
	def dims(self):
		"""
        Returns the dimensions of the grid's data.

        Returns
        -------
        tuple
            A tuple of the grid's dimensions.
		"""
		return self._dims

	@property
	def periodicity(self):
		"""
        Returns the longitudinal periodicity of the grid (180 or 360 degrees).

        Returns
        -------
        int
            The periodicity of the grid in degrees.
		"""
		periodicity = 180 if self._data.coords['lon'][0] == -180 else 360
		return periodicity


	def _init_from_array(self, filename):
		"""
        Initializes the grid from a file (.nc or .npy).

        Parameters
        ----------
        filename : str
            The path to the input file.
		"""

		# Case 1: Initialization from an xarray
		
		if '.nc' in filename:
      
			# Load object
			ds 	       = xr.open_dataset(filename)
			self._data = deepcopy(ds)
	
			# Store properties
			self._frame = self._data.frame
			self._dims  = self._data.data_vars[self._property].dims
			self._axes  = [self._data.coords[key].data for key in list(self._dims)]
	
			# Close object
			ds.close()

		# Case 2: Initialization from a numpy array
  
		elif '.npy' in filename:

			# Load object
			self._data = np.load(filename)
   
			# Check dims
			if self._property == 'temperature' and len(self._axes)==3:
				dims   = ["time", "lon", "lat"]
				coords = dict( time = self._axes[0], lon = self._axes[1], lat = self._axes[2])
			else:
				dims   = ["lon", "lat"]
				coords = dict( lon = self._axes[0], lat = self._axes[1])    
    
			# Build the x-array dataset
			self._data = xr.Dataset(
							data_vars={f"{self._property}" : (dims, self._data),},
							coords=coords,
							attrs=dict(frame=self._frame,),
						)

			# Store dimensions 
			self._dims  = self._data.data_vars[self._property].dims

  
	def get_data(self, epoch = None):
		"""
        Returns the grid's data.

        Parameters
        ----------
        epoch : float, optional
            The epoch for which to retrieve the data, if the grid is
            time-dependent.

        Returns
        -------
        numpy.ndarray
            The grid's data.
		"""

		if self._property == 'temperature' and epoch != None:
			return self._data.temperature.interp(coords={'time': epoch}, method='linear', assume_sorted=False).data
		return self._data.data_vars[self._property].data


	def __getitem__(self, idxs):
		"""
        Allows indexing into the grid to retrieve interpolated values.

        Parameters
        ----------
        idxs : tuple
            A tuple of indices for each dimension of the grid.

        Returns
        -------
        float or numpy.ndarray
            The interpolated value(s) at the given indices.
		"""

		if np.any([isinstance(x, slice) for x in idxs]):
			return self.values[idxs]
		else:
			outval =  interpolate.interpn(self._axes, self._data.data_vars[self._property].data, idxs, method = 'linear') 
			return np.squeeze(outval)


	def save(self, filename: str, complev: int = 1):
		"""
        Saves the grid to a NetCDF file.

        Parameters
        ----------
        filename : str
            The path to the output file.
        complev : int, default=1
            The compression level for the output file.
		"""
		if os.path.exists(filename): os.remove(filename)
		self._dataset.to_netcdf(filename, encoding = self._dataset.encoding.update({'zlib': True, 'complevel': complev}))


	def plot(self, epoch = None):
		"""
        Plots the grid data on a 2D map and a 3D sphere.

        Parameters
        ----------
        epoch : float, optional
            The epoch for which to plot the data, if the grid is
            time-dependent.
		"""
  
		# Build Lon, Lat array
		Lon      = self._axes[0]
		Lat      = self._axes[1]
		LON, LAT = np.meshgrid(Lon, Lat)
		urlon    = int(Lon[-1])
		lllon    = int(Lon[0])
		LAT      = LAT.T
		LON      = LON.T

		# Get data
		DATA = self.get_data(epoch)
  
		print("")
		print(f"Max: {DATA.max()}")
		print(f"Min: {DATA.min()}")
		print(f"Mean: {DATA.mean()}")
		print("")
  
		if Lon.shape[0] > 360: 
			N     = int(Lon.shape[0]/360)
			idxs1 = list(range(0,Lon.shape[0],N))
			idxs2 = list(range(0,Lat.shape[0],N))
			LON   = LON[np.ix_(idxs1,idxs2)]
			LAT   = LAT[np.ix_(idxs1,idxs2)]
			DATA  = DATA[np.ix_(idxs1,idxs2)]
  
		# Number of rows and columns
		nrows = LON.shape[0]
		ncols = LON.shape[1]
  
		# Init arrays
		X      = np.zeros((nrows,ncols))
		Y      = np.zeros((nrows,ncols))
		Z      = np.zeros((nrows,ncols))

		# Cartesian coordinates
		for r in range(nrows):
			
			for c in range(ncols):
				
				lat = LAT[r, c]*np.pi/180.
				lon = LON[r, c]*np.pi/180.

				X[r, c] = self._radius * np.cos(lat) * np.cos(lon)
				Y[r, c] = self._radius * np.cos(lat) * np.sin(lon)
				Z[r, c] = self._radius * np.sin(lat)

		# - 2D PLOT
		fig, ax = plt.subplots(1, 1, figsize=(12,6))
		norm = plt.Normalize(vmin=DATA.min(), vmax=DATA.max()) 
		map  = Basemap(projection='cyl', llcrnrlon = lllon, urcrnrlon=urlon, llcrnrlat = -90, urcrnrlat = 90,)
		map.drawmeridians(np.arange(lllon, urlon, 45), linewidth=1.0, dashes=[1,3], labels=[0,0,0,1],fontsize = 22, zorder=11, color = 'black', textcolor = 'black')
		map.drawmeridians(np.arange(lllon, urlon, 22.5), linewidth=0.5, dashes=[1,3], labels=[0,0,0,0],fontsize = 22, zorder=11, color = 'black', textcolor = 'black')
		map.drawparallels(np.arange(-90, 90, 20), linewidth=1.0, dashes=[1,3], labels=[1,0,0,0],fontsize = 22, zorder=12, color = 'black', textcolor = 'black')
		map.drawparallels(np.arange(-90, 90, 10), linewidth=0.5, dashes=[1,3], labels=[0,0,0,0],fontsize = 22, zorder=12, color = 'black', textcolor = 'black')
		cbbnd = np.linspace(DATA.min(), DATA.max(), num=5000, endpoint=True)
		if self._property == 'albedo':
			map.contourf(LON, LAT, DATA, cmap = cmx.binary_r, levels = cbbnd )
			m = cmx.ScalarMappable(cmap=cmx.binary_r, norm = norm)
		else:
			map.contourf(LON, LAT, DATA, cmap = cmx.hot, levels = cbbnd )
			m = cmx.ScalarMappable(cmap=cmx.hot, norm = norm) 
		m.set_array(DATA)
		plt.colorbar(m)

		# - 3D PLOT
		fig  = plt.figure(figsize = (8,6))
		norm = plt.Normalize(vmin=DATA.min(), vmax=DATA.max()) 
		ax   = fig.add_subplot(1, 1, 1, projection='3d')
		ax.quiver(0,0,0,2*self._radius,0.,0., zorder=5, color='b')
		if self._property == 'albedo':
			ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmx.binary_r(norm(DATA)), linewidth=0.2, antialiased=False, shade=False, alpha = 1.0)
			m    = cmx.ScalarMappable(cmap=cmx.binary_r, norm = norm)
		else:
			ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cmx.hot(norm(DATA)), linewidth=0.2, antialiased=False, shade=False, alpha = 1.0)
			m    = cmx.ScalarMappable(cmap=cmx.hot, norm = norm)
		ax.set_xlabel('X') 
		ax.set_ylabel('Y')  
		ax.set_zlabel('Z') 
		m.set_array(DATA)
		plt.colorbar(m)
		fig.suptitle(f'{self._property.upper()} MAP IN {self._frame}', fontsize=16)

		plt.tight_layout()
		plt.show()
  

class EmissivityGrid(PlanetGrid):
	"""
    Represents a planetary emissivity grid.
	"""
    
    
	def __init__(self,**kwargs):
		"""
        Initializes the EmissivityGrid object.

        Parameters
        ----------
        radius : float, optional
            The radius of the planet.
        frame : str, optional
            The SPICE reference frame of the grid.
        planet_name : str, optional
            The name of the planet.
        axes : list, optional
            A list of the grid's coordinate axes.
        from_array : str, optional
            The path to a file (.nc or .npy) from which to load the grid data.
		"""
		super().__init__()
  
		# Set property
		self._property = 'emissivity'

		# Store inputs
		if 'radius' in kwargs.keys():
			self._radius          =  kwargs['radius']
		if 'frame' in kwargs.keys():
			self._frame           = kwargs['frame']
		if 'planet_name' in kwargs.keys():
			self._planet_name     = kwargs['planet_name']
		if 'axes' in kwargs.keys():
			self._axes 			  = kwargs['axes']
		if 'from_array' in kwargs.keys():
			self._init_from_array(kwargs['from_array'])
   
    
class AlbedoGrid(PlanetGrid):
	"""
    Represents a planetary albedo grid.
	"""
    
    
	def __init__(self,**kwargs):
		"""
        Initializes the AlbedoGrid object.

        Parameters
        ----------
        radius : float, optional
            The radius of the planet.
        frame : str, optional
            The SPICE reference frame of the grid.
        planet_name : str, optional
            The name of the planet.
        axes : list, optional
            A list of the grid's coordinate axes.
        from_array : str, optional
            The path to a file (.nc or .npy) from which to load the grid data.
		"""
		super().__init__()
  
		# Set property
		self._property = 'albedo'

		# Store inputs
		if 'radius' in kwargs.keys():
			self._radius          =  kwargs['radius']
		if 'frame' in kwargs.keys():
			self._frame           = kwargs['frame']
		if 'planet_name' in kwargs.keys():
			self._planet_name     = kwargs['planet_name']
		if 'axes' in kwargs.keys():
			self._axes 			  = kwargs['axes']
		if 'from_array' in kwargs.keys():
			self._init_from_array(kwargs['from_array'])
  

class TemperatureGrid(PlanetGrid):
	"""
    Represents a planetary temperature grid.
	"""
    
    
	def __init__(self,**kwargs):
		"""
        Initializes the TemperatureGrid object.

        Parameters
        ----------
        albedo : AlbedoGrid or float, optional
            The albedo of the planet.
        base_flux : float, optional
            The base solar flux at 1 AU.
        radius : float, optional
            The radius of the planet.
        frame : str, optional
            The SPICE reference frame of the grid.
        planet_name : str, optional
            The name of the planet.
        nightside_temp : float, optional
            The nightside temperature of the planet.
        emissivity : EmissivityGrid or float, optional
            The emissivity of the planet.
        step : int, optional
            The step size for the grid computation.
        axes : list, optional
            A list of the grid's coordinate axes.
        epochs : list of float, optional
            A list of epochs for which to compute the temperature grid.
        from_array : str, optional
            The path to a file (.nc or .npy) from which to load the grid data.
		"""
		super().__init__()

		# Set property
		self._property = 'temperature'
  
		# Store inputs
		if 'albedo' in kwargs.keys():	
			self._albedo          =  kwargs['albedo']
		if 'base_flux' in kwargs.keys():
			self._base_flux       =  kwargs['base_flux']
		if 'radius' in kwargs.keys():
			self._radius          =  kwargs['radius']
		if 'frame' in kwargs.keys():
			self._frame           = kwargs['frame']
		if 'planet_name' in kwargs.keys():
			self._planet_name     = kwargs['planet_name']
		if 'nightside_temp' in kwargs.keys():
			self._nightside_temp  = kwargs['nightside_temp']
		if 'emissivity' in kwargs.keys():
			self._emissivity      = kwargs['emissivity']
		if 'step' in kwargs.keys():
			self._step    		  = kwargs['step']
		if 'axes' in kwargs.keys():
			self._axes 			  = kwargs['axes']
		if 'epochs' in kwargs.keys():
			step = kwargs['step'] if 'step' in kwargs.keys() else 1
			self.compute(kwargs['epochs'], step)
		elif 'from_array' in kwargs.keys():
			self._init_from_array(kwargs['from_array'])


	def get_albedo(self, epoch, dir):
		"""
        Retrieves the albedo value for a given direction and epoch.

        Parameters
        ----------
        epoch : float
            The epoch for which to retrieve the albedo.
        dir : numpy.ndarray
            The direction vector for which to retrieve the albedo.

        Returns
        -------
        float
            The albedo value.
		"""

		if self._albedo.frame != self._frame:
      
			rot = sp.pxform(self._frame, self._albedo.frame, epoch)
			dir = np.dot(rot, dir)

		dir = np.reshape(dir, (1,3))
  
		lon, lat = computeRADEC(dir, periodicity=self._albedo.periodicity)

		return self._albedo[lon*180/np.pi, lat*180/np.pi]


	def get_emissivity(self, epoch, dir):
		"""
        Retrieves the emissivity value for a given direction and epoch.

        Parameters
        ----------
        epoch : float
            The epoch for which to retrieve the emissivity.
        dir : numpy.ndarray
            The direction vector for which to retrieve the emissivity.

        Returns
        -------
        float
            The emissivity value.
		"""

		if self._emissivity.frame != self._frame:
      
			rot = sp.pxform(self._frame, self._emissivity.frame, epoch)
			dir = np.dot(rot, dir)

		dir = np.reshape(dir, (1,3))
  
		lon, lat = computeRADEC(dir, periodicity=self._emissivity.periodicity)

		return self._emissivity[lon*180/np.pi, lat*180/np.pi]

     
	def compute(self, epochs, step):
		"""
        Computes the temperature grid in the sun-fixed frame.

        Parameters
        ----------
        epochs : list of float
            A list of epochs for which to compute the temperature grid.
        step : int
            The latitude and longitude step in degrees.
		"""
  
		# -------------------------------------------------------------------
		# COMPUTE SUN DIRECTION

		sunVec = sp.spkezr('Sun', np.array(epochs), self._frame, 'None', self._planet_name)
		sunVec = [sunVec[0][i][0:3] for i in range(len(epochs))]

		# -------------------------------------------------------------------
		# DEFINE MESH GRID

		# Compute grid of latitude and Lonitude (DO NOT CHANGE RESOLUTION)
		Lon      = np.arange(-180, 181, step)
		Lat      = np.arange(90, -91, -step)

		# Number of rows and columns
		nrows = len(Lon)
		ncols = len(Lat)
  
		# Init array
		self._data  = np.zeros((len(epochs), nrows, ncols))
   
		for i, sunvec in enumerate(sunVec):
		
			for r in range(nrows):
				
				for c in range(ncols):
        
					lon = Lon[r]*np.pi/180.
					lat = Lat[c]*np.pi/180.

					# Cell normal unit vector
					nij  = np.array( [np.cos(lat)*np.cos(lon),
									  np.cos(lat)*np.sin(lon),
									  np.sin(lat),] )
		
					# Get albedo value
					if isinstance(self._albedo, AlbedoGrid):
						alb = self.get_albedo(epochs[i], nij)
					elif isinstance(self._albedo, float):
						alb = self._albedo

					# Get emissivity value
					if isinstance(self._emissivity, EmissivityGrid):
						emi = self.get_emissivity(epochs[i], nij)
					elif isinstance(self._emissivity, float):
						emi = self._emissivity
     
					# Sun directions
					cellSunDir = [-self._radius*nij[0] + sunvec[0],  
								  -self._radius*nij[1] + sunvec[1],  
								  -self._radius*nij[2] + sunvec[2]]

					# Scaled solar flux
					phi = self._base_flux*((au/np.linalg.norm(cellSunDir))**2)
					
					# Sun-aspect angle
					x          = np.dot(nij, cellSunDir/np.linalg.norm(cellSunDir))
					GammaAngle = np.arccos(x)
					inLight    = -np.pi/2 < GammaAngle < np.pi/2
     
					# Compute temperature				
					if inLight:      
						
						# In-light condition
						self._data[i,r,c]  =  ((1-alb) * phi * np.cos(GammaAngle) / (sb*emi))**0.25
     
					else: 
						
						# Terminator condition
						self._data[i,r,c] = self._nightside_temp
      
					if self._data[i,r,c] < self._nightside_temp: self._data[i,r,c] = self._nightside_temp
     
		# Build the x-array dataset
		self._data = xr.Dataset(
						data_vars=dict(
							temperature=(["time", "lon", "lat"], self._data),
						),
						coords=dict(
							lon = Lon,
							lat = Lat,
							time = epochs,
						),
						attrs=dict(frame=self._frame,),
					)

		# Store dimensions and axes
		self._dims  = self._data.temperature.dims
		self._axes  = [self._data.coords[key].data for key in list(self._dims)]
  

class Planet():
	"""
    Represents a celestial body, such as a planet or moon.
	"""
    
    
	def __init__(self, fromFile = None, radius = 0, name = '', bodyFrame = '', sunFixedFrame = '', units = 'km', subdivs = 4):
		"""
        Initializes the Planet object.

        Parameters
        ----------
        fromFile : str, optional
            Path to an OBJ file to build the model from. If None, a sphere is
            created.
        radius : float, default=0
            The radius of the planet (if creating a sphere).
        name : str, default=''
            The name of the planet.
        bodyFrame : str, default=''
            The SPICE reference frame for the planet's body.
        sunFixedFrame : str, default=''
            The body-centered, sun-fixed reference frame.
        units : str, default='km'
            The units for the planet's dimensions.
        subdivs : int, default=4
            The number of subdivisions for the icosphere (if creating a
            sphere).
		"""

		self.name = name
		self.fromFile = fromFile
		self.sunFixedFrame = sunFixedFrame
		conversion_factor = constants.unit_conversions[units]
		if fromFile is None:
			self.base_shape = tm.creation.icosphere(subdivisions = subdivs, radius = radius)
			self.bodyFrame = bodyFrame
		else:
			self.base_shape = tm.load_mesh(fromFile)
			self.bodyFrame = bodyFrame

		self.base_shape.apply_transform(tmt.scale_matrix(conversion_factor, [0,0,0]))
		self.numFaces = len(self.base_shape.faces)
		self._nightside_temp_temperature = -1
		self._dayside_temperature = -1
		self._gridded_temperature = -1
		self._albedo = 0
  
		self.sp_data = None


	def mesh(self, translate = None, rotate = None, epoch = None, targetFrame = None):
		"""
        Returns the planet's mesh, optionally transformed.

        Parameters
        ----------
        translate : list or numpy.ndarray, optional
            A 3-element vector for the translation.
        rotate : str, optional
            The name of the reference frame to rotate from.
        epoch : float, optional
            The epoch for the rotation.
        targetFrame : str, optional
            The name of the reference frame to rotate to.

        Returns
        -------
        trimesh.Trimesh
            The transformed mesh.
		"""

		if targetFrame is None:
			targetFrame = self.bodyFrame

		newShape = copy.deepcopy(self.base_shape)
  
		if rotate is not None:  
			if self.sp_data != None:
				tmatrix = self.sp_data.getRotation(epoch, rotate, targetFrame)
			else:
				tmatrix = sp.pxform(rotate, targetFrame, epoch)	
				tmatrix = self.pxform_convert(tmatrix)
			newShape.apply_transform(tmatrix)
			
		if translate is not None:
			transl_matrix = tmt.translation_matrix(translate)
			newShape.apply_transform(transl_matrix)
   
		return newShape


	def _is_sunlit(self, epoch):
		"""
        Determines which faces of the planet are sunlit at a given epoch.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.

        Returns
        -------
        numpy.ndarray
            The indices of the sunlit faces.
		"""
     
		rotated_mesh = self.mesh(epoch = epoch, rotate = self.bodyFrame, targetFrame = self.sunFixedFrame, translate = None)
		
		V = rotated_mesh.vertices
		F = rotated_mesh.faces

		centers = get_centroids(V,F)

		idxs = np.where(centers[:,0] > 0)

		return idxs[0]


	def _is_visible(self, spacecraft_name, epoch):
		"""
        Determines which faces of the planet are visible from a spacecraft at a
        given epoch.

        Parameters
        ----------
        spacecraft_name : str
            The name of the spacecraft.
        epoch : float
            The epoch for the calculation.

        Returns
        -------
        numpy.ndarray
            The indices of the visible faces.
		"""

		if self.name == '':
			raise Error('You must provide a name for the planet')

		#mesh = self.mesh(epoch = epoch) # The following line should be perfectly equivalent
		mesh = self.base_shape

		if self.sp_data != None:
			sc_pos = self.sp_data.getPosition(epoch, self.name, spacecraft_name, self.bodyFrame, 'CN')
		else:
			sc_pos = sp.spkezr(spacecraft_name, epoch, self.bodyFrame, 'CN', self.name)[0][0:3]
	
		V = mesh.vertices
		F = mesh.faces
		N = mesh.face_normals
		centers = get_centroids(V,F)

		sc_pos = -centers + sc_pos
		sc_pos = block_normalize(sc_pos)

		angles = np.sum(np.multiply(N, sc_pos), axis = 1)
		idxs = np.where(angles > 0)

		return  idxs[0]


	def albedoFaces(self, epoch, spacecraft_name):
		"""
        Returns the indices and albedo values of the faces that contribute to
        the albedo calculation.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.
        spacecraft_name : str
            The name of the spacecraft.

        Returns
        -------
        tuple
            A tuple containing the indices of the albedo faces and their
            corresponding albedo values.
		"""

		id_visible = self._is_visible(spacecraft_name, epoch)

		id_sunlit = self._is_sunlit(epoch)

		albedoIdxs = np.intersect1d(id_visible, id_sunlit, assume_unique = True)
		
		albedoValues = self.getFaceAlbedo(epoch)[albedoIdxs]

		return albedoIdxs, albedoValues


	def rot_toSCframe(self, epoch, scFrame = None):
		"""
        Returns the rotation matrix from the sun-fixed frame to a spacecraft
        frame.

        Parameters
        ----------
        epoch : float
            The epoch for the rotation.
        scFrame : str, optional
            The name of the spacecraft frame.

        Returns
        -------
        numpy.ndarray
            The 3x3 rotation matrix.
		"""
     
		return sp.pxform(self.sunFixedFrame, scFrame, epoch)


	def emissivityFaces(self, epoch, spacecraft_name):
		"""
        Returns the indices, temperatures, and emissivities of the faces that
        contribute to the emissivity calculation.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.
        spacecraft_name : str
            The name of the spacecraft.

        Returns
        -------
        tuple
            A tuple containing the indices of the emissive faces, their
            temperatures, and their emissivities.
		"""
	
		visible_ids = self._is_visible(spacecraft_name, epoch)
		visibleTemps = self.getFaceTemperatures(epoch)[visible_ids]
		visibleEmi = self.getFaceEmissivity(epoch)[visible_ids]

		return visible_ids, visibleTemps, visibleEmi


	def getFaceAlbedo(self, epoch):
		"""
        Returns the albedo of each face at a given epoch.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.

        Returns
        -------
        numpy.ndarray
            An array containing the albedo of each face.
		"""
		
		if isinstance(self._albedo, AlbedoGrid):
      
			sff = True if self._albedo.frame == self.sunFixedFrame else False
			_,_,_,C = self.VFNC(epoch, sunFixedFrame = sff)
			lon, lat = computeRADEC(C, periodicity = self._albedo.periodicity)
			
			albedoValues = self._albedo[lon*180/np.pi, lat*180/np.pi]
   
		elif isinstance(self._albedo, np.ndarray):
      
			albedoValues = self._albedo

		return albedoValues
	
 
	def getFaceTemperatures(self, epoch):
		"""
        Returns the temperature of each face at a given epoch.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.

        Returns
        -------
        numpy.ndarray
            An array containing the temperature of each face.
		"""
  
		if (self._dayside_temperature == -1 or self._nightside_temp_temperature == -1) and self._gridded_temperature == -1:
			raise ValueError('Error: the planet temperatures have not been set. Set via: Planet.dayside_temperature and Planed.nightside_temperature')
		id_sunlit = self._is_sunlit(epoch)

		if self._gridded_temperature != -1:
      
			sff = True if self.gridded_temperature.frame == self.sunFixedFrame else False
			_,_,_,C = self.VFNC(epoch, sunFixedFrame = sff)
			lon, lat = computeRADEC(C, periodicity = self.gridded_temperature.periodicity)
			if len(self.gridded_temperature.axes)==3:
				faceTemps = self.gridded_temperature[epoch, lon*180/np.pi, lat*180/np.pi] 
			else:
				faceTemps = self.gridded_temperature[lon*180/np.pi, lat*180/np.pi] 
       
		else:

			faceTemps = np.full(self.numFaces, self._nightside_temp_temperature)
			faceTemps[id_sunlit] = self._dayside_temperature
  
		return faceTemps


	def getFaceEmissivity(self, epoch, sunFixedFrame = False):
		"""
        Returns the emissivity of each face at a given epoch.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.
        sunFixedFrame : bool, default=False
            If True, the calculation is performed in the sun-fixed frame.

        Returns
        -------
        numpy.ndarray
            An array containing the emissivity of each face.
		"""
     
		if isinstance(self._emissivity, TiffInterpolator):
			_,_,_,C = self.VFNC(epoch, sunFixedFrame = sunFixedFrame)
			lon, lat = computeRADEC(C, periodicity = 180)
			emissivity = self._emissivity[lon,lat]
		else:
			emissivity = self._emissivity
		
		return emissivity


	def VFNC(self, epoch, sunFixedFrame = True):
		"""
        Returns the vertices, faces, normals, and centroids of the planet's mesh.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.
        sunFixedFrame : bool, default=True
            If True, the mesh is rotated to the sun-fixed frame.

        Returns
        -------
        tuple
            A tuple containing the vertices, faces, normals, and centroids.
		"""
  
		if sunFixedFrame:
			rotated_mesh = self.mesh(epoch = epoch, rotate = self.bodyFrame, targetFrame = self.sunFixedFrame, translate = None)
		else:
			rotated_mesh = self.mesh(epoch = epoch)
			
		V = rotated_mesh.vertices
		F = rotated_mesh.faces 
		N = rotated_mesh.face_normals
		C = get_centroids(V,F)

		return V, F, N, C


	def getScPosSunFixed(self,epoch, spacecraft_name):
		"""
        Returns the position of a spacecraft in the sun-fixed frame.

        Parameters
        ----------
        epoch : float
            The epoch for the calculation.
        spacecraft_name : str
            The name of the spacecraft.

        Returns
        -------
        numpy.ndarray
            The position of the spacecraft.
		"""
		correction = 'CN'
		sc_pos = sp.spkezr(spacecraft_name, epoch, self.sunFixedFrame, correction, self.name)

		return sc_pos[0][0:3]

	
	#def getFaceTemp(self, epoch):
	#	"""
	#	Return the temperature of each face at epoch

	#	"""


	#	temps = np.full(self.numFaces, self._nightside_temp_temperature)
	#	litIdxs = self._is_sunlit(epoch)

	#	temps[litIdxs] = self._dayside_temperature
	#	
	#	return temps


	def pxform_convert(self,pxform):
		"""
        Converts a 3x3 SPICE rotation matrix to a 4x4 transformation matrix.

        Parameters
        ----------
        pxform : numpy.ndarray
            The 3x3 rotation matrix.

        Returns
        -------
        numpy.ndarray
            The 4x4 transformation matrix.
		"""
		pxform = np.array([pxform[0],pxform[1],pxform[2]])

		p = np.append(pxform,[[0,0,0]],0)

		mv = np.random.random()
		p = np.append(p,[[0],[0],[0],[0]], 1)
		return p


	# PROPERTIES
 
	@property
	def dayside_temperature(self):
		"""
        The dayside temperature of the planet.
		"""

		return self._dayside_temperature

	@dayside_temperature.setter
	def dayside_temperature(self, value):
		"""
        Sets the dayside temperature of the planet.
		"""
		self._dayside_temperature = value

	@property
	def nightside_temperature(self):
		"""
        The nightside temperature of the planet.
		"""

		return self._nightside_temp_temperature

	@nightside_temperature.setter
	def nightside_temperature(self, value):
		"""
        Sets the nightside temperature of the planet.
		"""
		self._nightside_temp_temperature = value
		
	@property
	def gridded_temperature(self):
		"""
        The gridded temperature of the planet.
		"""

		return self._gridded_temperature

	@gridded_temperature.setter
	def gridded_temperature(self, value):
		"""
        Sets the gridded temperature of the planet.
		"""
		if not isinstance(value, TemperatureGrid):
			raise ValueError('Error: the input must be a TemperatureGrid object')
		self._gridded_temperature = value

	@property
	def albedo(self):
		"""
        The albedo of the planet.
		"""
		return self._albedo

	@albedo.setter
	def albedo(self, value):
		"""
        Sets the albedo of the planet.
		"""
		nFaces = len(self.base_shape.faces)  
		if isinstance(value, float):
			print('Setting a single value of albedo for all the faces')

			self._albedo = np.full(nFaces, value)
		elif isinstance(value, AlbedoGrid):
			self._albedo = value
		elif len(value) == nFaces:
			self._albedo = value
		else:
			raise IndexError('Error mismatch between input and number of face meshes')

	@property
	def albedo_map(self):
		"""
        The albedo map settings for the planet.
		"""
		try:
			return self._albedo_map
		except AttributeError:
			print(f'Albedo Map Settings not set for body {self.name}. \n Set them by declaring a dictionaty of the form [lon0 : the zero longitude (either 0 or -180), lontype: LST (for local solar time) or True (for true longitude)]')
   
	@albedo_map.setter
	def albedo_map(self, value):
		"""
        Sets the albedo map settings for the planet.
		"""
		self._albedo_map = value

	@property
	def emissivity(self):
		"""
        The emissivity of the planet.
		"""
		try:
			return self._emissivity
		except AttributeError:
			print(f'Error: emissivity not set for body {self.name}')
   
	@emissivity.setter
	def emissivity(self, value):
		"""
        Sets the emissivity of the planet.
		"""
		nFaces = len(self.base_shape.faces)  
		if isinstance(value, float) == 1:
			print('Setting a single value of emissivity for all the faces')

			self._emissivity = np.full(nFaces, value)
		elif isinstance(value, TiffInterpolator):
			self._emissivity = value
		elif len(value) == nFaces:
			self._emissivity = value
		
		else:
			raise IndexError('Error mismatch between input and number of face meshes')
			
