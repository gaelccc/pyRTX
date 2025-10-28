# Spacecraft class
import trimesh as tm 
import trimesh.transformations as tmt 
import numpy as np 
import spiceypy as sp
import matplotlib
import copy
from pyRTX import constants


class Spacecraft():
	"""
	This is the main class for defining spacecraft objects. 
	"""
	
 
	def __init__(self, name = None, base_frame = None, spacecraft_model = None, units = 'm', mass = 0.):
		"""
 		Parameters
		----------
		name : str 
			Spacecraft name

		base_frame : str 
			Spacecraft body (base) frame

		spacecraft_model : dict 
			dict of {file:str, frame_type:str, frame_name:str, center:list, specular:float, diffuse:float, UD_rotation:trimesh.rotation}
        	a dictionary with keys the name of

			file: str 
				the obj file for the part

			frame_type: str 
				'Spice' or 'UD' to choose wether to define a reference to a spice frame or UserDefined one
				in the case of 'UD' a rotation matrix must be specified in the UD_rotation (optional) key

			frame_name: str 
				The name of the Spice (or UD) frame

			center: list  
				position of the origin of the object (in km) with respect to the base frame

			specular: float 
				specular coefficient

			diffuse: float 
				diffuse coefficient
				
			UD_rotation: trimesh.rotation
				optional specify a user defined rotations matrix

		units : str 
			units for transformations
   
		mass : float or .nc file
			Spacecraft mass. Can be a float value or an xarray with times and mass values
   
		Returns
		-------
		bla : pyRTX.classes.Spacecraft
		"""
  
		self.name = name
		self.part_number = len(spacecraft_model.keys())
		self.base_frame = base_frame
		self.mesh_cont = {}
		self.mesh_prop = {}
		self.units = units
		self.conversion_factor = constants.unit_conversions[units]
		

		self.spacecraft_model = {}
		self._initialize(spacecraft_model)
		self.material_dict = {}
		self.dump_materials()

		self.mass    = mass
		self.sp_data = None
		#self._last_epoch = 0


	def _load_obj(self, fname):
		mesh = tm.load_mesh(fname, skip_texture = True)
		#if isinstance(mesh, tm.Scene): mesh = mesh.dump(concatenate = True)
		mesh.apply_transform(tmt.scale_matrix(self.conversion_factor, [0,0,0]))
		return mesh


	def _initialize(self, input_model):

		# Load the meshes
		self.spacecraft_model.update(input_model)

		for elem in input_model.keys():
			
			if isinstance(input_model[elem]['file'], tm.Trimesh):
				self.spacecraft_model[elem]['base_mesh'] = input_model[elem]['file'].apply_transform(tmt.scale_matrix(self.conversion_factor, [0,0,0]))
			else:
				self.spacecraft_model[elem]['base_mesh'] = self._load_obj(input_model[elem]['file'])
			self.spacecraft_model[elem]['translation'] = tmt.translation_matrix(np.array(input_model[elem]['center']) * self.conversion_factor)


			#print(tmt.translation_matrix(input_model[elem]['center']))


	def _precompute_rot_matrices(self, epochs, convert = True):
		"""
		Method to pre-compute rotation matrices.

		Parameters:
		-	epochs: list of epochs
		-	convert: boolean flag to convert the output of pxform into a 4x4 matrix
		"""
		# Index - epochs mapping
		self._epochs_dict = {epoch: idx for idx, epoch in enumerate(epochs)}
  
		# Initialize transformation matrices:
		dim = 4 if convert else 3
		self._rot_matrices = {elem: np.zeros((len(epochs), dim, dim), dtype = np.float64) for elem in self.spacecraft_model.keys()}

		# Compute rotation matrices:
		for i, epoch in enumerate(epochs):
			for elem in self.spacecraft_model.keys():	
				bframe  = self.spacecraft_model[elem]['frame_name']
				tframe  = self.base_frame
				tmatrix = sp.pxform(bframe, tframe, epoch)
				if convert: tmatrix = self.pxform_convert(tmatrix)
				self._rot_matrices[elem][i] = tmatrix
    

	def add_parts(self, spacecraft_model = None):
		"""
		Add parts to the model instance
		Parameters
		----------
		spacecraft_model : dict
			See the main constructor documentation
		"""

		if name in  self.spacecraft_model.keys():
			raise Exception(f'{name} is already defined')

		self._initialize(spacecraft_model)


	def subset(self, elem_names):
		'''
		Return an instance of Spacecraft with only the elements contained
		in the list elem_names.
		Suppose the Spacecraft (self) is composed of elements A,B,C
		Spacecraft.subset(['A','B']) would return a new instance
		of Spacecraft with only the elements A and B
		'''
		
		cself  = copy.deepcopy(self)
		orig_elems = copy.deepcopy(list(cself.spacecraft_model.keys()))
		for k in orig_elems:
			if k not in elem_names:
				cself.remove_part(k)
		return cself


	def remove_part(self, name):
		del self.spacecraft_model[name]


	def pxform_convert(self,pxform):
		pxform = np.array([pxform[0],pxform[1],pxform[2]])

		p = np.append(pxform,[[0,0,0]],0)

		mv = np.random.random()
		p = np.append(p,[[0],[0],[0],[0]], 1)
		return p


	def apply_transforms(self, epoch):
		"""
		Method to rotate and translate the components.

		Parameters:
		-	epoch: epoch of the transformation
		"""
		for elem in self.spacecraft_model.keys():

			self.spacecraft_model[elem]['mesh'] = copy.deepcopy(self.spacecraft_model[elem]['base_mesh'])
			
			if self.spacecraft_model[elem]['frame_type'] == 'Spice':

				bframe  = self.spacecraft_model[elem]['frame_name']
				tframe  = self.base_frame
    
				if self.sp_data != None:

					tmatrix = self.sp_data.getRotation(epoch, bframe, tframe)
     
				else:

					tmatrix = sp.pxform(bframe, tframe, epoch)
					tmatrix = self.pxform_convert(tmatrix)	
     
				self.spacecraft_model[elem]['mesh'].apply_transform(tmatrix)
    
			else:
				
				self.spacecraft_model[elem]['mesh'].apply_transform(self.spacecraft_model[elem]['UD_rotation'])

			self.spacecraft_model[elem]['mesh'].apply_transform(self.spacecraft_model[elem]['translation'])


	def materials(self):
		return self.material_dict


	def dump(self,epoch = None, split = False):

		mesh_todump =[]
  
		if not epoch == None:
			if isinstance(epoch, str):
				et = sp.str2et( epoch )
			else:
				et = epoch

		#if et == self._last_epoch:
		#	return self._last_mesh

			self.apply_transforms(et)

		else:
			self.apply_transforms(None)

		for elem in self.spacecraft_model:

			try:
				mesh_todump.append(np.sum(self.spacecraft_model[elem]['mesh'].dump()))
			except AttributeError:
				mesh_todump.append(self.spacecraft_model[elem]['mesh'])
		if not split:
			mesh = np.array(mesh_todump).sum()
		else:
			mesh = mesh_todump
		#self._last_mesh = np.array(mesh_todump).sum()
		#self._last_epoch = et

		return mesh


	def dump_materials(self):
		counter = 0
		stored_idxs = []
		props = {}
		for elem in self.spacecraft_model:
			try:
				elemMesh = np.sum(self.spacecraft_model[elem]['base_mesh'].dump())
			except AttributeError:
				elemMesh = self.spacecraft_model[elem]['base_mesh']
			stored_idxs.append([counter, counter + len(elemMesh.faces)-1])
			counter += len(elemMesh.faces)
			props[elem] = {'specular' : self.spacecraft_model[elem]['specular'] , 'diffuse' :self.spacecraft_model[elem]['diffuse'] }
			
		self.material_dict = {'idxs' : stored_idxs, 'props': props}


	def _elem_info(self, elem):
		# print(self.spacecraft_model[elem]['frame_name'])
		return f"{elem}: Proper Frame: {self.spacecraft_model[elem]['frame_name']} | Frame Type: {self.spacecraft_model[elem]['frame_type']}"


	def info(self):
		elems = self.spacecraft_model.keys()
		n_parts = len(elems)
		printstr = f"Spacecraft {self.name} composed of {n_parts} elements: \n"
		for i,elem in enumerate(elems):
			printstr += f'{i+1}) ' + self._elem_info(elem) + ' \n'
		print(printstr)


	def __str__(self):
		return self.info()

