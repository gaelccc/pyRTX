import trimesh as tm
import trimesh.transformations as tmt
import copy
import spiceypy as sp
import numpy as np
from pyRTX.utils_rt import get_centroids, block_normalize
from pyRTX.analysis_utils import computeRADEC
from pyRTX import constants
from pyRTX.analysis_utils import TiffInterpolator



class Planet():
	def __init__(self, fromFile = None, radius = 0, name = '', bodyFrame = '', sunFixedFrame = '', units = 'km', subdivs = 4):
		"""
		A class to represent a planet/moon

		Input:
		fromFile : put here an obj file if requested to build the model. If None (default) a sphere with radius defined in "radius" will be built
		radius : (float) the radius of the planet. Not used if "fromFile" is not None
		name : (str) the name of the planet
		bodyFrame: (str) the planet body fixed frame
		sunFixedFrame: (str) the body centered - sun fixed frame
		units: (str) [Default: km] the measurement units defining the body (can be km or m)
		subdivs: (int) [Default: 4] the number of subdivision for the creation of the spherical planet. Note that the number of faces will grow as function of 4 ** subdivisions, so you probably want to keep this under ~5.


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
		self._nightside_temperature = -1
		self._dayside_temperature = -1
		self._gridded_temperature = -1
		self._albedo = 0
		
	
	def mesh(self, translate = None, rotate = None, epoch = None, targetFrame = None):

		
		if targetFrame is None:
			targetFrame = self.bodyFrame
		newShape = copy.deepcopy(self.base_shape)
		if rotate is not None:  
			tmatrix = sp.pxform(rotate, targetFrame, epoch)	
			tmatrix = self.pxform_convert(tmatrix)
			newShape.apply_transform(tmatrix)
		if translate is not None:
			transl_matrix = tmt.translation_matrix(translate)
			newShape.apply_transform(transl_matrix)
		return newShape




	def _is_sunlit(self, epoch):
		rotated_mesh = self.mesh(epoch = epoch, rotate = self.bodyFrame, targetFrame = self.sunFixedFrame, translate = None)
		
		V = rotated_mesh.vertices
		F = rotated_mesh.faces

		centers = get_centroids(V,F)

		idxs = np.where(centers[:,0] > 0)


		return idxs[0]

	def _is_visible(self, spacecraft_name, epoch):

		if self.name == '':
			raise Error('You must provide a name for the planet')
		#mesh = self.mesh(epoch = epoch) # The following line should be perfectly equivalent
		mesh = self.base_shape
		sc_pos = sp.spkezr(spacecraft_name, epoch, self.bodyFrame, 'LT+S', self.name)
	

		V = mesh.vertices
		F = mesh.faces
		N = mesh.face_normals
		centers = get_centroids(V,F)


		sc_pos = -centers + sc_pos[0][0:3]
		sc_pos = block_normalize(sc_pos)

		angles = np.sum(np.multiply(N, sc_pos), axis = 1)
		idxs = np.where(angles > 0)




		return  idxs[0]



	def albedoFaces(self, epoch, spacecraft_name):

		"""
		Public method:
		Return the idxs of the mesh faces that are needed for albedo computation at time:epoch for the spacecraft:spacectaft name

		"""

		id_visible = self._is_visible(spacecraft_name, epoch)

		id_sunlit = self._is_sunlit(epoch)


		albedoIdxs = np.intersect1d(id_visible, id_sunlit, assume_unique = True)
		
		albedoValues = self.getFaceAlbedo(epoch)[albedoIdxs]

		return albedoIdxs, albedoValues

	def rot_toSCframe(self, epoch, scFrame = None):
		return sp.pxform(self.sunFixedFrame, scFrame, epoch)


	def emissivityFaces(self, epoch, spacecraft_name):
		"""
		Public method:
		Return the idxs of the mesh faces and the temperature of each face that are needed for emissivity computation at time:epoch for the spacecraft:spacectaft name

		"""
	
		visible_ids = self._is_visible(spacecraft_name, epoch)
		visibleTemps = self.getFaceTemperatures(epoch)[visible_ids]
		visibleEmi = self.getFaceEmissivity(epoch)[visible_ids]
		#return self._is_visible(spacecraft_name, epoch), visibleTemps
		return visible_ids, visibleTemps, visibleEmi

	def getFaceAlbedo(self, epoch):
		"""
		Return the albedo of each face at epoch
		"""
		
		if isinstance(self._albedo, TiffInterpolator):
			# Get information on the used map
			options = self._albedo_map
			perio = 180 if options['lon0'] == -180 else 360
			sff = 1 if options['lontype'] == 'LST' else 0


			_,_,_,C = self.VFNC(epoch, sunFixedFrame = sff)
			
			lon, lat = computeRADEC(C, periodicity = perio)
			albedoValues = self._albedo[lon, lat]
		elif isinstance(self._albedo, np.ndarray):
			albedoValues = self._albedo

		return albedoValues
	
	def getFaceTemperatures(self, epoch):
		"""
		Return the temperature of each face at epoch
		"""
		if (self._dayside_temperature == -1 or self._nightside_temperature == -1) and self._gridded_temperature == -1:
			raise ValueError('Error: the planet temperatures have not been set. Set via: Planet.dayside_temperature and Planed.nightside_temperature')
		id_sunlit = self._is_sunlit(epoch)

		if self._gridded_temperature != -1:
			_,_,_,C = self.VFNC(epoch)
			lon, lat = computeRADEC(C, periodicity = 180)
			#print(lon)
			faceTemps = self.gridded_temperature[lon, lat] 

		else:

			faceTemps = np.full(self.numFaces, self._nightside_temperature)
			faceTemps[id_sunlit] = self._dayside_temperature

		return faceTemps

	def getFaceEmissivity(self, epoch):
		if isinstance(self._emissivity, TiffInterpolator):
			_,_,_,C = self.VFNC(epoch)
			lon, lat = computeRADEC(C, periodicity = 180)
			emissivity = self._emissivity[lon,lat]
		else:
			emissivity = self._emissivity
		
		return emissivity


	def VFNC(self, epoch, sunFixedFrame = True):
		"""
		Public method:
		Returns  V F N C rotating the planet in the sunFixedFrame at epoch epoch
		V: Vertices
		F: Faces
		N: Normals
		C: Centroids


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
		correction = 'CN'
		sc_pos = sp.spkezr(spacecraft_name, epoch, self.sunFixedFrame, correction, self.name)

		return sc_pos[0][0:3]

	


	#def getFaceTemp(self, epoch):
	#	"""
	#	Return the temperature of each face at epoch

	#	"""


	#	temps = np.full(self.numFaces, self._nightside_temperature)
	#	litIdxs = self._is_sunlit(epoch)

	#	temps[litIdxs] = self._dayside_temperature
	#	
	#	return temps

		
		
		

	def pxform_convert(self,pxform):
		pxform = np.array([pxform[0],pxform[1],pxform[2]])

		p = np.append(pxform,[[0,0,0]],0)

		mv = np.random.random()
		p = np.append(p,[[0],[0],[0],[0]], 1)
		return p





	# PROPERTIES

 
	@property
	def dayside_temperature(self):

		return self._dayside_temperature

	@dayside_temperature.setter
	def dayside_temperature(self, value):
		self._dayside_temperature = value

	@property
	def nightside_temperature(self):

		return self._nightside_temperature

	@nightside_temperature.setter
	def nightside_temperature(self, value):
		self._nightside_temperature = value
		
	@property
	def gridded_temperature(self):

		return self._gridded_temperature

	@gridded_temperature.setter
	def gridded_temperature(self, value):
		if not isinstance(value, TiffInterpolator):
			raise ValueError('Error: the input must be a TiffInterpolator object')
		self._gridded_temperature = value

	@property
	def albedo(self):
		return self._albedo

	@albedo.setter
	def albedo(self, value):
		nFaces = len(self.base_shape.faces)  
		if isinstance(value, float):
			print('Setting a single value of albedo for all the faces')

			self._albedo = np.full(nFaces, value)
		elif isinstance(value, TiffInterpolator):
			self._albedo = value
		elif len(value) == nFaces:
			self._albedo = value
		else:
			raise IndexError('Error mismatch between input and number of face meshes')

	@property
	def albedo_map(self):
		try:
			return self._albedo_map
		except AttributeError:
			print(f'Albedo Map Settings not set for body {self.name}. \n Set them by declaring a dictionaty of the form [lon0 : the zero longitude (either 0 or -180), lontype: LST (for local solar time) or True (for true longitude)]')
	@albedo_map.setter
	def albedo_map(self, value):
		self._albedo_map = value



	@property
	def emissivity(self):
		try:
			return self._emissivity
		except AttributeError:
			print(f'Error: emissivity not set for body {self.name}')
	@emissivity.setter
	def emissivity(self, value):
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
			


	














