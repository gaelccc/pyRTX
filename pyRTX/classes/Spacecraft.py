# Spacecraft class
import trimesh as tm
import trimesh.transformations as tmt
import numpy as np
import spiceypy as sp
import matplotlib
import copy
from pyRTX import constants
import pyRTX.core.utils_rt as utils_rt


class Spacecraft():
	"""
	Represents a spacecraft, including its geometry, materials, and orientation.

    This class manages the different components of a spacecraft, their transformations,
    and material properties. It can load geometry from OBJ files and uses SPICE
    kernels to determine the orientation of each component.
	"""


	def __init__(self, name=None, base_frame=None, spacecraft_model=None, units='m', mass=0.):
		"""
		Initializes the Spacecraft object.

        Parameters
        ----------
        name : str, optional
            The name of the spacecraft.
        base_frame : str, optional
            The SPICE reference frame for the spacecraft's body.
        spacecraft_model : dict, optional
            A dictionary defining the spacecraft's components. Each key is the
            component name, and the value is another dictionary with the following
            keys:
            - 'file' (str): Path to the OBJ file for the component.
            - 'frame_type' (str): 'Spice' or 'UD' (User Defined).
            - 'frame_name' (str): The name of the SPICE or UD frame.
            - 'center' (list): The position of the component's origin in the
              base frame.
            - 'specular' (float): The specular reflection coefficient.
            - 'diffuse' (float): The diffuse reflection coefficient.
            - 'UD_rotation' (trimesh.transformations.Transform, optional): A
              user-defined rotation matrix.
        units : str, default='m'
            The units for the spacecraft model's dimensions ('m' for meters,
            'km' for kilometers, etc.).
        mass : float or xarray.Dataset, default=0.
            The spacecraft's mass. Can be a constant float or an xarray Dataset
            with time-varying mass.
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
		"""
		Loads a mesh from an OBJ file and applies the specified unit conversion.

        Parameters
        ----------
        fname : str
            The path to the OBJ file.

        Returns
        -------
        trimesh.Trimesh
            The loaded mesh, scaled according to the spacecraft's units.
		"""
		mesh = tm.load_mesh(fname, skip_texture = True)
		#if isinstance(mesh, tm.Scene): mesh = mesh.dump(concatenate = True)
		mesh.apply_transform(tmt.scale_matrix(self.conversion_factor, [0,0,0]))
		return mesh


	def _initialize(self, input_model):
		"""
        Initializes the spacecraft model by loading meshes and setting up
        transformations.

        Parameters
        ----------
        input_model : dict
            A dictionary describing the spacecraft's components, as defined
            in the `__init__` method.
		"""

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
		Pre-computes the rotation matrices for each component at a given set of
        epochs.

        Parameters
        ----------
        epochs : list of float
            The list of epochs (in SPICE ephemeris time) for which to
            pre-compute the rotation matrices.
        convert : bool, default=True
            If True, converts the 3x3 SPICE rotation matrices to 4x4
            transformation matrices.
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
				if convert: tmatrix = utils_rt.pxform_convert(tmatrix)
				self._rot_matrices[elem][i] = tmatrix


	def add_parts(self, spacecraft_model=None):
		"""
		Adds new parts to the spacecraft model.

        Parameters
        ----------
        spacecraft_model : dict
            A dictionary describing the new components to add, in the same
            format as the `spacecraft_model` parameter of the `__init__`
            method.

        Raises
        ------
        Exception
            If a part with the same name already exists in the model.
		"""

		if name in  self.spacecraft_model.keys():
			raise Exception(f'{name} is already defined')

		self._initialize(spacecraft_model)


	def subset(self, elem_names):
		"""
		Creates a new Spacecraft instance containing only a subset of the
        components of the current instance.

        Parameters
        ----------
        elem_names : list of str
            A list of the names of the components to include in the new
            instance.

        Returns
        -------
        Spacecraft
            A new Spacecraft instance with only the specified components.
		"""

		cself  = copy.deepcopy(self)
		orig_elems = copy.deepcopy(list(cself.spacecraft_model.keys()))
		for k in orig_elems:
			if k not in elem_names:
				cself.remove_part(k)
		return cself


	def remove_part(self, name):
		"""
		Removes a part from the spacecraft model.

        Parameters
        ----------
        name : str
            The name of the part to remove.
		"""
		del self.spacecraft_model[name]


	def apply_transforms(self, epoch):
		"""
		Applies the rotations and translations to each component for a given
        epoch.

        Parameters
        ----------
        epoch : float
            The epoch (in SPICE ephemeris time) at which to apply the
            transformations.
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
					tmatrix = utils_rt.pxform_convert(tmatrix)

				self.spacecraft_model[elem]['mesh'].apply_transform(tmatrix)

			else:

				self.spacecraft_model[elem]['mesh'].apply_transform(self.spacecraft_model[elem]['UD_rotation'])

			self.spacecraft_model[elem]['mesh'].apply_transform(self.spacecraft_model[elem]['translation'])


	def materials(self):
		"""
		Returns the material properties dictionary for the spacecraft.

        Returns
        -------
        dict
            A dictionary containing the material properties of each component.
		"""
		return self.material_dict


	def dump(self, epoch=None, split=False):
		"""
		Returns the combined or individual meshes of the spacecraft's components
        at a specific epoch.

        Parameters
        ----------
        epoch : float, optional
            The epoch (in SPICE ephemeris time) for which to get the meshes.
            If None, the base meshes are used without transformation.
        split : bool, default=False
            If True, returns a list of individual meshes for each component.
            If False, returns a single combined mesh.

        Returns
        -------
        trimesh.Trimesh or list of trimesh.Trimesh
            The combined mesh or a list of individual component meshes.
		"""

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
		"""
		Generates and stores a dictionary of material properties for all
        components.
		"""
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
		"""
		Returns a formatted string with information about a specific component.

        Parameters
        ----------
        elem : str
            The name of the component.

        Returns
        -------
        str
            A string containing the component's frame information.
		"""
		# print(self.spacecraft_model[elem]['frame_name'])
		return f"{elem}: Proper Frame: {self.spacecraft_model[elem]['frame_name']} | Frame Type: {self.spacecraft_model[elem]['frame_type']}"


	def info(self):
		"""
		Prints a summary of the spacecraft's components.
		"""
		elems = self.spacecraft_model.keys()
		n_parts = len(elems)
		printstr = f"Spacecraft {self.name} composed of {n_parts} elements: \n"
		for i,elem in enumerate(elems):
			printstr += f'{i+1}) ' + self._elem_info(elem) + ' \n'
		print(printstr)


	def __str__(self):
		"""
		Returns a string representation of the spacecraft's information.

        Returns
        -------
        str
            A string containing the spacecraft's summary.
		"""
		return self.info()
