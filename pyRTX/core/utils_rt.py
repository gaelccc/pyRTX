##################################
# Utilities for ray tracing (part of pyRTX module)
# 
# Developed by: Gael Cascioli 2021

import numpy as np
import trimesh
from abc import ABC
import os
import pickle as pkl
from numba import jit
import multiprocessing as mproc
from pyRTX.defaults import dFloat, dInt
from pyRTX import EMBREE_AVAILABLE

if EMBREE_AVAILABLE:
    import embree
    
else:
    # Provide fallback or informative error
    raise ImportError(
            "Embree is not available. This feature requires Embree.\n"
            "Please install it by running: python install_deps.py\n"
            "On Linux, the environment should be configured automatically.\n"
            "On other platforms, you may need to manually source embree-vars.sh"
        )

try:
        from aabb import AABB
except: 
        pass

#####
# Mods for performanc optimization
import hashlib
import threading

# Thread-safe Embree scene cache
_EMBREE_CACHE = {}
_EMBREE_CACHE_LOCK = threading.Lock()

def _hash_mesh(mesh_obj):
    """
    Compute a reproducible hash for mesh geometry (vertices + faces).

    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        The mesh object to hash.

    Returns
    -------
    str
        A hexadecimal hash string.
    """
    V = np.asarray(mesh_obj.vertices, dtype=np.float32)
    F = np.asarray(mesh_obj.faces, dtype=np.int32)
    # Stable hash across sessions
    m = hashlib.sha1()
    m.update(V.tobytes())
    m.update(F.tobytes())
    return m.hexdigest()

def get_cached_embree_scene(mesh_obj):
    """
    Return a cached EmbreeTrimeshShapeModel for this mesh, creating if needed.

    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        The mesh object to get a scene for.

    Returns
    -------
    EmbreeTrimeshShapeModel
        The cached or newly created Embree shape model.
    """
    mesh_hash = _hash_mesh(mesh_obj)
    with _EMBREE_CACHE_LOCK:
        if mesh_hash not in _EMBREE_CACHE:
            
            _EMBREE_CACHE[mesh_hash] = Embree3_init_geometry(mesh_obj)
          
        return _EMBREE_CACHE[mesh_hash]

####

# except ImportError:
#	print("""Could not import Embree3 library. \n Be sure that: 
#		\n 1) Embree3 is installed\n 2) you activated embree variables (source /path/to/lib/embree-vars.sh)""")


# No more imports after here
# ___________________________________________________________

############################################################
# Define general utils
###########################################################


# def parallelize(iterator, function, chunks):
#     """Define a general parallelization framework to speedup computations
#         Parameters
#         ----------
#         iterator : array_like
#             the array-like object over which to parallelize
#         function : function handle
#             the function that must be called by each subprocess. The function should be of the form y = f(iterator)
#         chunks : int
#             number of parallel workers
#         Returns
#         -------
#         result : array_like
#             the result of the parallelized computation
#             """

#     with mproc.Pool(chunks) as p:
#         result = p.map(function, iterator)

#     return result


def chunker(iterator, chunks):
    """
    Divide an iterator or array into approximately equal-sized chunks for
    parallel processing.
    
    Parameters
    ----------
    iterator : array_like
        The array-like object to divide into chunks.
    chunks : int
        Number of chunks to create.
    
    Returns
    -------
    list of arrays
        List containing the chunked arrays.
    """
    return np.array_split(iterator, chunks)


def pxform_convert(pxform):
    """
    Convert a 3x3 SPICE rotation matrix to a 4x4 homogeneous transformation matrix.
    
    Parameters
    ----------
    pxform : array_like, shape (3, 3)
        The 3x3 rotation matrix from SPICE.
    
    Returns
    -------
    ndarray, shape (4, 4)
        The 4x4 homogeneous transformation matrix.
    """
    pxform = np.array([pxform[0], pxform[1], pxform[2]], dtype=dFloat)

    p = np.append(pxform, [[0, 0, 0]], 0)

    p = np.append(p, [[0], [0], [0], [1]], 1)
    return p


def block_normalize(V):
    """
    Compute unit vectors for a block of vectors.
    
    Parameters
    ----------
    V : ndarray, shape (N, 3) or (3,)
        Array of vectors to normalize.
    
    Returns
    -------
    ndarray
        The normalized vectors.
    """

    if V.ndim > 1:
        return V / np.linalg.norm(V, axis=1).reshape(len(V), 1)
    else:
        return V / np.linalg.norm(V)


def block_dot(a, b):
    """
    Perform an element-wise dot product between two arrays of vectors.
    
    Parameters
    ----------
    a : ndarray, shape (N, m)
        First array of vectors.
    b : ndarray, shape (N, m)
        Second array of vectors.
    
    Returns
    -------
    ndarray, shape (N,)
        Array of dot products.
    """


    return np.sum(a * b, axis=1)


def pixel_plane(d0, lon, lat, width=1, height=1, ray_spacing=.1):
    """
    Generate a rectangular grid of rays for ray tracing.
    
    Parameters
    ----------
    d0 : float
        Distance of the pixel plane from the origin.
    lon : float
        Longitude of the pixel plane's center direction in radians.
    lat : float
        Latitude of the pixel plane's center direction in radians.
    width : float, default=1
        Width of the plane.
    height : float, default=1
        Height of the plane.
    ray_spacing : float, default=0.1
        Spacing between adjacent rays.
    
    Returns
    -------
    tuple
        A tuple containing:
        - locs (ndarray): Ray origin positions.
        - dirs (ndarray): Ray direction unit vectors.
    """




    w2 = width / 2
    h2 = height / 2
    # Build the direction vector
    x0 = np.array([-d0 * np.cos(lon) * np.cos(lat), -d0 * np.sin(lon) * np.cos(lat), -d0 * np.sin(lat)])

    # Build the transformation matrix
    R1 = np.array([[np.cos(lon), -np.sin(lon), 0],
                   [np.sin(lon), np.cos(lon), 0],
                   [0, 0, 1]]
                  )
    R2 = np.array([[np.cos(-lat), 0, np.sin(-lat)],
                   [0, 1, 0],
                   [-np.sin(-lat), 0, np.cos(-lat)]])
    R = R1 @ R2

    # Build the pixel matrix
    basic_coords = np.zeros(((int(width / ray_spacing) + 1) * (int(height / ray_spacing) + 1), 3))
    basic_dirs = np.zeros_like(basic_coords)
    counter = 0
    for i, w in enumerate(np.linspace(-w2, w2, num=int(width / ray_spacing) + 1)):
        for j, h in enumerate(np.linspace(-h2, h2, num=int(height / ray_spacing) + 1)):
            basic_coords[counter, :] = R @ np.array([0, w, h]) - x0
            basic_dirs[counter, :] = x0 / np.linalg.norm(x0)
            counter += 1

    # Return the output in the shape required by trimesh
    return basic_coords, basic_dirs


@jit(nopython=True)
def fast_vector_build(linsp1, linsp2, dim1, dim2):
    """
    Efficiently build a pixel array coordinate grid using Numba.
    
    Parameters
    ----------
    linsp1 : ndarray
        Positions along the first dimension.
    linsp2 : ndarray
        Positions along the second dimension.
    dim1 : int
        Number of points in the first dimension.
    dim2 : int
        Number of points in the second dimension.
    
    Returns
    -------
    ndarray
        Array of 3D coordinates.
    """
    basic_coords = np.zeros((dim1 * dim2, 3))
    counter = 0

    for w in linsp1:
        for h in linsp2:
            basic_coords[counter, :] = [0, w, h]
            counter += 1
    return basic_coords


def pixel_plane_opt(d0, lon, lat, width=1, height=1, ray_spacing=.1, packets=1):
    """
    Generate a rectangular pixel array (optimized version).
    
    Parameters
    ----------
    d0 : float
        Distance of the pixel plane from the origin.
    lon : float
        Longitude of the pixel plane's center direction in radians.
    lat : float
        Latitude of the pixel plane's center direction in radians.
    width : float, default=1
        Width of the plane.
    height : float, default=1
        Height of the plane.
    ray_spacing : float, default=0.1
        Spacing between adjacent rays.
    packets : int, default=1
        Number of ray packets to subdivide the rays into.
    
    Returns
    -------
    tuple
        A tuple containing:
        - locs (ndarray): Ray origin positions.
        - dirs (ndarray): Ray direction unit vectors.
    """
    w2 = width / 2
    h2 = height / 2
    # Build the direction vector
    x0 = np.array([-d0 * np.cos(lon) * np.cos(lat), -d0 * np.sin(lon) * np.cos(lat), -d0 * np.sin(lat)])
    x0_unit = x0 / np.linalg.norm(x0)
    # Build the transformation matrix
    R1 = np.array([[np.cos(lon), -np.sin(lon), 0],
                   [np.sin(lon), np.cos(lon), 0],
                   [0, 0, 1]]
                  )
    R2 = np.array([[np.cos(-lat), 0, np.sin(-lat)],
                   [0, 1, 0],
                   [-np.sin(-lat), 0, np.cos(-lat)]])
    R = R1 @ R2

    # Build the pixel matrix

    dim1 = int(width / ray_spacing) + 1
    dim2 = int(height / ray_spacing) + 1
    basic_coords = np.zeros((dim1 * dim2, 3))
    basic_dirs = np.full(basic_coords.shape, x0_unit)

    linsp1 = np.linspace(-w2, w2, num=dim1)
    linsp2 = np.linspace(-h2, h2, num=dim2)

    basic_coords = fast_vector_build(linsp1, linsp2, dim1, dim2)

    basic_coords = np.dot(np.array(basic_coords), R.T)
    basic_coords -= x0

    # Return the output in the shape required by trimesh

    if not packets == 1:
        basic_coords = np.array_split(basic_coords, packets)
        basic_dirs = np.array_split(basic_dirs, packets)

    return basic_coords, basic_dirs


def reflected(incoming, normal):
    """
    Compute reflected ray directions.
    
    Parameters
    ----------
    incoming : ndarray
        Incoming ray direction vectors.
    normal : ndarray
        Surface normal vectors.
    
    Returns
    -------
    ndarray
        Reflected ray direction vectors.
    """

    return incoming - 2 * np.multiply(normal.T, np.einsum('ij,ij->i', incoming, normal)).T


@jit(nopython=True)
def get_orthogonal(v):
    """
    Generate a unit vector orthogonal to the input vector.
    
    Parameters
    ----------
    v : ndarray
        Input vector.
    
    Returns
    -------
    ndarray
        Orthogonal unit vector.
    """

    x = np.random.random(3)
    x -= x.dot(v) * v
    x = x / np.linalg.norm(x)

    return x


@jit(nopython=True)
def sample_lambert_dist(normal, num=100):
    """
    Generate direction vectors following the Lambert cosine distribution.
    
    Parameters
    ----------
    normal : ndarray
        Surface normal vector.
    num : int, default=100
        Number of sample directions to generate.
    
    Returns
    -------
    ndarray
        Array of sampled direction vectors.
    """

    theta = np.arccos(np.sqrt(np.random.random(num)))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    psi = np.random.random(num) * 2 * np.pi

    a = sin_theta * np.cos(psi)
    b = sin_theta * np.sin(psi)
    c = cos_theta

    t1 = get_orthogonal(normal)
    t2 = np.cross(normal, t1)

    v = np.zeros((num, 3))
    for i in range(num):
        v[i] = a[i] * t1 + b[i] * t2 + c[i] * normal
    return v


def _core_diffuse(normals, diffuse_directions, num):
    """
    Core function for computing diffuse reflection directions.

    Parameters
    ----------
    normals : ndarray
        Surface normal vectors.
    diffuse_directions : ndarray
        Array to store the diffuse directions.
    num : int
        Number of diffuse samples to generate for each normal.

    Returns
    -------
    ndarray
        Array of diffuse direction vectors.
    """
    for i, n in enumerate(normals):
        diff_dirs = sample_lambert_dist(n, num=num)

        diffuse_directions[i * num: (i + 1) * num] = diff_dirs
    return diffuse_directions


def diffuse(normals, num=10):
    """
    Compute multiple diffuse reflection directions for an array of surface normals.
    
    Parameters
    ----------
    normals : ndarray
        Array of surface normal unit vectors.
    num : int, default=10
        Number of diffuse samples to generate for each normal.
    
    Returns
    -------
    ndarray
        Array of sampled diffuse direction vectors.
    """


    diffuse_directions = np.repeat(normals, num, axis=0) * 0.0

    for i, n in enumerate(normals):
        diff_dirs = sample_lambert_dist(n, num=num)

        diffuse_directions[i * num: (i + 1) * num] = diff_dirs

    return diffuse_directions


# return _core_diffuse(normals, diffuse_directions, num)

def compute_secondary_bounce(location, index_tri, mesh_obj, ray_directions, index_ray, diffusion=False,
                             num_diffuse=None):
    """
    Prepare ray origins and directions for subsequent bounces.
    
    Parameters
    ----------
    location : ndarray
        3D coordinates of intersection points.
    index_tri : ndarray
        Indices of the intersected mesh faces.
    mesh_obj : trimesh.Trimesh
        The mesh object.
    ray_directions : ndarray
        Direction vectors of the incident rays.
    index_ray : ndarray
        Indices of the rays that intersected the mesh.
    diffusion : bool, default=False
        If True, compute diffuse reflection directions.
    num_diffuse : int, optional
        Number of diffuse samples per intersection point.
    
    Returns
    -------
    tuple
        A tuple containing:
        - location (ndarray): Intersection point coordinates.
        - reflect_dirs (ndarray): Specularly reflected ray directions.
        - diffuse_dirs (ndarray or int): Diffusely reflected directions or -1.
    """
    reflect_dirs = np.zeros_like(location)
    normals = mesh_obj.face_normals

    reflect_dirs = reflected(ray_directions[index_ray], normals[index_tri])

    if diffusion:
        diffuse_dirs = diffuse(normals[index_tri], num=num_diffuse)

    else:
        diffuse_dirs = -1  # Dummy variable for return values management

    return location, reflect_dirs, diffuse_dirs


####################################################################################################
# Saving utilities

def save_for_visualization(outputFilePath, mesh, ray_origins, ray_directions, location, index_tri, diffusion_pack):
    """
    Save ray tracing results to a pickled dictionary for visualization.
    
    Parameters
    ----------
    outputFilePath : str
        Path to the output file.
    mesh : trimesh.Trimesh
        The mesh object.
    ray_origins : list of ndarrays
        List of ray origin arrays for each bounce.
    ray_directions : list of ndarrays
        List of ray direction arrays for each bounce.
    location : list of ndarrays
        List of intersection point coordinates for each bounce.
    index_tri : list of ndarrays
        List of indices of intersected triangles for each bounce.
    diffusion_pack : list or None
        Diffuse ray tracing data.
    """

    outdict = {'mesh': mesh, 'ray_origins': ray_origins, 'ray_directions': ray_directions, 'locations': location,
               'index_tri': index_tri, 'diffusion_pack': diffusion_pack}

    with open(outputFilePath, 'wb') as f:
        pkl.dump(outdict, f, protocol=4)




#####################################################################################################
# Define utils specific to Embree3 implementation
# (Provided by Sam Potter)

def Embree3_init_geometry(mesh_obj):
    """
    Initialize mesh geometry for the Embree 3 ray tracing kernel.
    
    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        Input mesh object.
    
    Returns
    -------
    EmbreeTrimeshShapeModel
        The Embree shape model.
    """
    V = np.array(mesh_obj.vertices, dtype=np.float32, copy = False)
    F = np.array(mesh_obj.faces, dtype=np.int32, copy = False)

    # P = get_centroids(V, F)
    N, A = get_surface_normals_and_face_areas(V, F)

    return EmbreeTrimeshShapeModel(V, F, N=N, A=A)


def Embree3_init_rayhit(ray_origins, ray_directions):
    """
    Initialize Embree 3 RayHit data structure.
    
    Parameters
    ----------
    ray_origins : ndarray
        Starting positions of rays.
    ray_directions : ndarray
        Direction vectors of rays.
    
    Returns
    -------
    embree.RayHit1M
        Initialized RayHit structure.
    """
    nb = np.shape(ray_origins)[0]  # Number of tracked rays
    rayhit = embree.RayHit1M(nb)

    # Initialize the ray structure
    # rayhit.tnear[:] = 0.001 #Avoid numerical problems
    rayhit.tnear[:] = 1e-6  # Avoid numerical problems
    rayhit.tfar[:] = np.inf
    rayhit.prim_id[:] = embree.INVALID_GEOMETRY_ID
    rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
    rayhit.org[:] = ray_origins
    rayhit.dir[:] = ray_directions

    return rayhit


def Embree3_dump_solution(rayhit, V, F):
    """
    Extract and process intersection results from an Embree RayHit structure.
    
    Parameters
    ----------
    rayhit : embree.RayHit1M
        The RayHit structure populated by Embree.
    V : ndarray
        Vertex coordinates of the mesh.
    F : ndarray
        Face indices of the mesh.
    
    Returns
    -------
    tuple
        A tuple containing:
        - hits (ndarray or int): Indices of intersected triangles.
        - nhits (int): Number of intersected rays.
        - idh (ndarray or int): Indices of rays that hit the mesh.
        - Ph (ndarray or int): 3D coordinates of intersection points.
    """

    ishit = rayhit.prim_id != embree.INVALID_GEOMETRY_ID
    idh = np.nonzero(ishit)[0]
    hits = rayhit.prim_id[idh]
    nhits = hits.size

    if nhits > 0:
        # p = V[F[hits]]
        # v1 = p[:, 0]
        # v2 = p[:, 1]
        # v3 = p[:, 2]
        v1 = V[F[hits, 0]]
        v2 = V[F[hits, 1]]
        v3 = V[F[hits, 2]]
        u = rayhit.uv[idh, 0]
        v = rayhit.uv[idh, 1]
        Ph = v1 + (v2 - v1) * u[:, None] + (v3 - v1) * v[:, None]

        return hits, nhits, idh, Ph

    else:
        return -1, -1, -1, -1

# Define utils specific to CGAL implementation
# (from python-flux)

def cgal_init_geometry(mesh_obj):
    """
    Initialize mesh geometry for the CGAL ray tracing kernel.
    
    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        Input mesh object.
    
    Returns
    -------
    CgalTrimeshShapeModel
        The CGAL shape model.
    """
    V = np.array(mesh_obj.vertices, dtype=np.float64)
    F = np.array(mesh_obj.faces, dtype=np.int64)

    N, A = get_surface_normals_and_face_areas(V, F)

    return CgalTrimeshShapeModel(V, F, N=N, A=A)


######################### from python-flux.src.flux.shape.py

def get_centroids(V, F):
    """
    Compute the geometric centroids of all triangular faces in a mesh.
    
    Parameters
    ----------
    V : ndarray
        Vertex coordinates of the mesh.
    F : ndarray
        Face indices.
    
    Returns
    -------
    ndarray
        Centroid coordinates for each face.
    """

    return V[F].mean(axis=1)


def get_cross_products(V, F):
    """
    Compute cross products of edge vectors for all triangular faces in a mesh.
    
    Parameters
    ----------
    V : ndarray
        Vertex coordinates of the mesh.
    F : ndarray
        Face indices.
    
    Returns
    -------
    ndarray
        Cross product vectors for each face.
    """
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C


def get_face_areas(V, F):
    """
    Compute the areas of all triangular faces in a mesh.
    
    Parameters
    ----------
    V : ndarray
        Vertex coordinates of the mesh.
    F : ndarray
        Face indices.
    
    Returns
    -------
    ndarray
        Area of each face.
    """

    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C ** 2, axis=1))
    A = C_norms / 2
    return A


def get_surface_normals(V, F):
    """
    Compute outward-pointing unit normal vectors for all triangular faces.
    
    Parameters
    ----------
    V : ndarray
        Vertex coordinates of the mesh.
    F : ndarray
        Face indices.
    
    Returns
    -------
    ndarray
        Unit normal vectors for each face.
    """

    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C ** 2, axis=1))
    N = C / C_norms.reshape(C.shape[0], 1)
    return N


def get_surface_normals_and_face_areas(V, F):
    """
    Efficiently compute both surface normals and face areas simultaneously.
    
    Parameters
    ----------
    V : ndarray
        Vertex coordinates of the mesh.
    F : ndarray
        Face indices.
    
    Returns
    -------
    tuple
        A tuple containing:
        - N (ndarray): Unit normal vectors for each face.
        - A (ndarray): Area of each face.
    """
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C ** 2, axis=1))
    N = C / C_norms.reshape(C.shape[0], 1)
    A = C_norms / 2
    return N, A


class ShapeModel(ABC):
    """An abstract base class for shape models."""
    pass


class TrimeshShapeModel(ShapeModel):
    """A shape model consisting of a single triangle mesh."""

    def __init__(self, V, F, N=None, P=None, A=None):
        """
        Initialize a triangle mesh shape model.

        Parameters
        ----------
        V : array_like
            An array with shape (num_verts, 3) of vertex coordinates.
        F : array_like
            An array with shape (num_faces, 3) of face indices.
        N : array_like, optional
            An array with shape (num_faces, 3) of face normals.
        P : array_like, optional
            An array with shape (num_faces, 3) of triangle centroids.
        A : array_like, optional
            An array of shape (num_faces,) of triangle areas.
        """
        if type(self) == TrimeshShapeModel:
            raise RuntimeError("tried to instantiate TrimeshShapeModel directly")

        self.dtype = V.dtype

        self.V = V
        self.F = F

        if N is None and A is None:
            N, A = get_surface_normals_and_face_areas(V, F)
        elif A is None:
            if N.shape[0] != F.shape[0]:
                raise Exception(
                    'must pass same number of surface normals as faces (got ' +
                    '%d faces and %d normals' % (F.shape[0], N.shape[0])
                )
            A = get_face_areas(V, F)
        elif N is None:
            N = get_surface_normals(V, F)

        self.P = get_centroids(V, F)
        self.N = N
        self.A = A

        assert self.P.dtype == self.dtype
        assert self.N.dtype == self.dtype
        assert self.A.dtype == self.dtype

        self._make_scene()

    def __reduce__(self):
        """
        Serialization method for pickling.
        """
        return (self.__class__, (self.V, self.F, self.N, self.P, self.A))

    def __repr__(self):
        """
        String representation of the object.
        """
        return 'a TrimeshShapeModel with %d vertices and %d faces' % (
            self.num_verts, self.num_faces)

    @property
    def num_faces(self):
        """The number of faces in the mesh."""
        return self.P.shape[0]

    @property
    def num_verts(self):
        """The number of vertices in the mesh."""
        return self.V.shape[0]

    def intersect1(self, x, d):
        """
        Trace a single ray.

        Parameters
        ----------
        x : array_like
            The origin of the ray.
        d : array_like
            The direction of the ray.

        Returns
        -------
        tuple
            A tuple containing the index of the hit and the distance to the hit.
        """
        return self._intersect1(x, d)

    def intersect1_2d_with_coords(self, X, D):
        """
        Trace multiple rays.

        Parameters
        ----------
        X : array_like
            The origins of the rays.
        D : array_like
            The directions of the rays.

        Returns
        -------
        tuple
            A tuple containing the indices of the hits and the coordinates of the hits.
        """
        return self._intersect1_2d_with_coords(X, D)

    def intersect1_2d(self, X, D):
        """
        Trace multiple rays and return only the indices of the hits.

        Parameters
        ----------
        X : array_like
            The origins of the rays.
        D : array_like
            The directions of the rays.

        Returns
        -------
        array_like
            The indices of the hits.
        """

        fint, xta = self.intersect1_2d_with_coords(X, D)

        return fint

class CgalTrimeshShapeModel(TrimeshShapeModel):
    """A triangle mesh shape model that uses the CGAL AABB tree for ray tracing."""
    def _make_scene(self):
        """
        Set up a CGAL AABB tree.
        """
        self.aabb = AABB.from_trimesh(
            self.V.astype(np.float64), self.F.astype(np.uintp))

    def _intersect1(self, x, d):
        """
        Trace a single ray using CGAL.
        """
        return self.aabb.intersect1(x, d)

    def _intersect1_2d(self, X, D):
        """
        Trace multiple rays using CGAL.
        """
        return self.aabb.intersect1_2d(X, D)

    def _intersect1_2d_with_coords(self, X, D):
        """
        Trace multiple rays with coordinates using CGAL.
        """
        return self.aabb.intersect1_2d_with_coords(X, D)

class EmbreeTrimeshShapeModel(TrimeshShapeModel):
    """A triangle mesh shape model that uses the Embree ray tracing kernel."""
    def _make_scene(self):
        """
        Set up an Embree scene.
        """
        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        geometry.set_build_quality(embree.BuildQuality.High)

        scene = device.make_scene()
        scene.set_build_quality(embree.BuildQuality.High)
        scene.set_flags(embree.SceneFlags.Robust)

        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex,  # buf_type
            0,  # slot
            embree.Format.Float3,  # fmt
            3 * np.dtype('float32').itemsize,  # byte_stride
            self.V.shape[0],  # item_count
        )
        vertex_buffer[:] = self.V[:]
        #
        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index,  # buf_type
            0,  # slot
            embree.Format.Uint3,  # fmt
            3 * np.dtype('uint32').itemsize,  # byte_stride,
            self.F.shape[0]
        )
        index_buffer[:] = self.F[:]

        geometry.commit()

        scene.attach_geometry(geometry)

        geometry.release()

        scene.commit()

        # This is the only variable we need to retain a reference to
        # (I think)
        self.scene = scene
        device.release()


    def _intersect1(self, x, d):
        """
        Trace a single ray using Embree.
        """
        raise RuntimeError('intersect1 no implemented for EmbreeTrimeshShapeModel')


trimesh_shape_models = [
    CgalTrimeshShapeModel,
    EmbreeTrimeshShapeModel
]


# Main definition of the kernel wrapper
# -----------------------------------------------------------------------------------------------------#

# def RTXkernel(mesh_obj, ray_origins, ray_directions, bounces=1, kernel='Embree3', diffusion=False, num_diffuse=None,
#               errorMsg=True):

#     """
#     Main ray tracing kernel wrapper.
#     """



#     ray_origins_container = []
#     ray_directions_container = []
#     locations_container = []
#     index_tri_container = []
#     index_ray_container = []

#     # Set variables for diffusion computation
#     diffusion_directions = 0
#     diffusion_pack = []
#     diffusion_control = False

#     # Select the kernel
#     if kernel in ['Embree', 'Native']:
#         for i in range(bounces):

#             ray_origins_container.append(ray_origins)

#             if kernel == 'Embree':
#                 intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_obj)

#             elif kernel == 'Native':
#                 intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_obj)

#             # Avoid numerical problems
#             ray_origins = ray_origins + 1e-3 * ray_directions

#             # If computing bounce number 1 and the diffusion computation has been requested
#             # do a separate raytracing also for the diffused rays
#             # and pack results in the variable: diffusion pack
#             if i == 1 and diffusion:
#                 ray_origins_diffusion = np.repeat(ray_origins, num_diffuse, axis=0)  # this should be correct
#                 ray_directions_diffusion = diffuse_directions

#                 index_tri_diffusion, index_ray_diffusion, location_diffusion = intersector.intersects_id(
#                     ray_origins=ray_origins_diffusion,
#                     ray_directions=ray_directions_diffusion,
#                     multiple_hits=False,
#                     return_locations=True)
#                 diffusion_pack = [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,
#                                   location_diffusion]

#             # Main Raytracer
#             index_tri, index_ray, location = intersector.intersects_id(
#                 ray_origins=ray_origins,
#                 ray_directions=ray_directions,
#                 multiple_hits=False,
#                 return_locations=True)
#             # Get the number of hits
#             n_hits = len(index_tri)

#             # Manage the possibility of no hits
#             if n_hits == 0 and errorMsg:
#                 print('No intersections found for bounce {}. Results provided up to bounce {}'.format(i + 1, i))
#                 break
#             else:
#                 locations_container.append(location)
#                 index_tri_container.append(index_tri)
#                 index_ray_container.append(index_ray)
#                 ray_directions_container.append(ray_directions)

#                 if i != bounces - 1:
#                     # If at bounce number 1 compute the diffused directions:
#                     if diffusion and i == 0:
#                         diffusion_control = True

#                     ray_origins, ray_directions, diffuse_directions = compute_secondary_bounce(location, index_tri,
#                                                                                                mesh_obj, ray_directions,
#                                                                                                index_ray,
#                                                                                                diffusion=diffusion_control,
#                                                                                                num_diffuse=num_diffuse)

#                     # Set back to false the diffusion computation control flag
#                     diffusion_control = False




#     elif kernel == 'Embree3':

#         # Initialize the geometry
#         shape_model = Embree3_init_geometry(mesh_obj)
#         context = embree.IntersectContext()

#         for i in range(bounces):
#             ray_origins_container.append(ray_origins)

#             # Initialize the rayhit object
#             ray_origins = ray_origins + 1e-3 * ray_directions
#             rayhit = Embree3_init_rayhit(ray_origins, ray_directions)

#             # Run the intersector
#             shape_model.scene.intersect1M(context, rayhit)

#             # Post-process the results
#             index_tri, n_hits, index_ray, location = Embree3_dump_solution(rayhit, shape_model.V, shape_model.F)

#             # embree.Device().release()
#             # Handle: not bounces found
#             if n_hits == -1:
#                 print('No intersections found for bounce {}. Results provided up to bounce {}'.format(i + 1, i))
#                 break


#             # Otherwise append results and proceed with next bounce
#             else:
#                 locations_container.append(location)
#                 index_tri_container.append(index_tri)
#                 index_ray_container.append(index_ray)
#                 ray_directions_container.append(ray_directions)

#                 if i != bounces - 1:
#                     ray_origins, ray_directions, _ = compute_secondary_bounce(location, index_tri, mesh_obj,
#                                                                               ray_directions, index_ray)
#         # # release memory
#         # scene.release()
#         shape_model.scene.release()

#     elif kernel == 'CGAL':

#         # Initialize the geometry
#         shape_model = cgal_init_geometry(mesh_obj)

#         for i in range(bounces):
#             ray_origins_container.append(ray_origins)

#             # Initialize the rayhit object
#             # ray_origins = ray_origins + 1e-3 * ray_directions

#             index_tri, location = shape_model.intersect1_2d_with_coords(ray_origins, ray_directions)

#             index_ray = np.arange(len(ray_origins))
#             n_hits = len(np.where(index_tri > -1))

#             index_ray = index_ray[np.where(index_tri > -1)]
#             location = location[np.where(index_tri > -1)]
#             index_tri = index_tri[np.where(index_tri > -1)]

#             # print(index_tri, n_hits, index_ray, location)

#             # Handle: not bounces found
#             if n_hits == -1:
#                 print('No intersections found for bounce {}. Results provided up to bounce {}'.format(i + 1, i))
#                 break


#             # Otherwise append results and proceed with next bounce
#             else:
#                 locations_container.append(location)
#                 index_tri_container.append(index_tri)
#                 index_ray_container.append(index_ray)
#                 ray_directions_container.append(ray_directions)

#                 if i != bounces - 1:
#                     ray_origins, ray_directions, _ = compute_secondary_bounce(location, index_tri, mesh_obj,
#                                                                               ray_directions, index_ray)

#     else:
#         print('No Recognized kernel')

#     # Manage output variables
#     if diffusion:
#         return index_tri_container, index_ray_container, locations_container, ray_origins_container, ray_directions_container, diffusion_pack
#     else:
#         return index_tri_container, index_ray_container, locations_container, ray_origins_container, ray_directions_container, None


def RTXkernel(mesh_obj, ray_origins, ray_directions, bounces=1, kernel='Embree3', diffusion=False, num_diffuse=None,
              errorMsg=True):
    """
    Main ray tracing kernel wrapper.
    
    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        The mesh geometry to ray trace.
    ray_origins : ndarray
        Starting positions of rays.
    ray_directions : ndarray
        Direction vectors of rays.
    bounces : int, default=1
        Number of reflection bounces to simulate.
    kernel : str, default='Embree3'
        Ray tracing backend to use ('Embree3', 'CGAL', or 'Native').
    diffusion : bool, default=False
        If True, compute diffuse reflections.
    num_diffuse : int, optional
        Number of diffuse samples per intersection.
    errorMsg : bool, default=True
        If True, print warnings when no intersections are found.
    
    Returns
    -------
    tuple
        A tuple containing the results of the ray tracing.
    """

    ray_origins_container = []
    ray_directions_container = []
    locations_container = []
    index_tri_container = []
    index_ray_container = []

    # Set variables for diffusion computation
    diffusion_directions = 0
    diffusion_pack = []
    diffusion_control = False

    # Select the kernel
    if kernel in ['Native']:
        for i in range(bounces):

            ray_origins_container.append(ray_origins)

            if kernel == 'Native':
                intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_obj)

            # Avoid numerical problems
            # ray_origins = ray_origins + 1e-3 * ray_directions
            ray_origins += 1e-3 * ray_directions

            # If computing bounce number 1 and the diffusion computation has been requested
            # do a separate raytracing also for the diffused rays
            # and pack results in the variable: diffusion pack
            if i == 1 and diffusion:
                ray_origins_diffusion = np.repeat(ray_origins, num_diffuse, axis=0)
                ray_directions_diffusion = diffuse_directions

                index_tri_diffusion, index_ray_diffusion, location_diffusion = intersector.intersects_id(
                    ray_origins=ray_origins_diffusion,
                    ray_directions=ray_directions_diffusion,
                    multiple_hits=False,
                    return_locations=True)
                diffusion_pack = [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,
                                  location_diffusion]

            # Main Raytracer
            index_tri, index_ray, location = intersector.intersects_id(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                multiple_hits=False,
                return_locations=True)
            # Get the number of hits
            n_hits = len(index_tri)

            # Manage the possibility of no hits
            if n_hits == 0 and errorMsg:
                print('No intersections found for bounce {}. Results provided up to bounce {}'.format(i + 1, i))
                break
            else:
                locations_container.append(location)
                index_tri_container.append(index_tri)
                index_ray_container.append(index_ray)
                ray_directions_container.append(ray_directions)

                if i != bounces - 1:
                    # If at bounce number 1 compute the diffused directions:
                    if diffusion and i == 0:
                        diffusion_control = True

                    ray_origins, ray_directions, diffuse_directions = compute_secondary_bounce(location, index_tri,
                                                                                               mesh_obj, ray_directions,
                                                                                               index_ray,
                                                                                               diffusion=diffusion_control,
                                                                                               num_diffuse=num_diffuse)

                    # Set back to false the diffusion computation control flag
                    diffusion_control = False

    elif kernel == 'Embree3':

        # Initialize the geometry
        # shape_model = Embree3_init_geometry(mesh_obj) # NOTE changed for performance optimization
        shape_model = get_cached_embree_scene(mesh_obj)
        
        context = embree.IntersectContext()

        for i in range(bounces):
            ray_origins_container.append(ray_origins)

            # Avoid numerical problems
            # NOTE: changed this and used native embree setting (tnear)
            # ray_origins = ray_origins + 1e-3 * ray_directions

            # If computing bounce number 1 and the diffusion computation has been requested
            # do a separate raytracing also for the diffused rays
            if i == 1 and diffusion:
                ray_origins_diffusion = np.repeat(ray_origins, num_diffuse, axis=0)
                ray_directions_diffusion = diffuse_directions

                # Initialize rayhit for diffusion rays
                rayhit_diffusion = Embree3_init_rayhit(ray_origins_diffusion, ray_directions_diffusion)
                
                # Run the intersector for diffusion
                shape_model.scene.intersect1M(context, rayhit_diffusion)
                
                # Post-process diffusion results
                index_tri_diffusion, n_hits_diffusion, index_ray_diffusion, location_diffusion = \
                    Embree3_dump_solution(rayhit_diffusion, shape_model.V, shape_model.F)
                
                # Pack diffusion results
                if n_hits_diffusion != -1:
                    diffusion_pack = [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,
                                      location_diffusion]
                else:
                    # No diffusion hits, create empty diffusion pack
                    diffusion_pack = [np.array([]), np.array([]), np.array([]), np.array([])]

            # Initialize the rayhit object for main rays
            rayhit = Embree3_init_rayhit(ray_origins, ray_directions)

            # Run the intersector
            shape_model.scene.intersect1M(context, rayhit)

            # Post-process the results
            index_tri, n_hits, index_ray, location = Embree3_dump_solution(rayhit, shape_model.V, shape_model.F)

            # Handle: no bounces found
            if n_hits == -1:
                if errorMsg:
                    print('No intersections found for bounce {}. Results provided up to bounce {}'.format(i + 1, i))
                break

            # Otherwise append results and proceed with next bounce
            else:
                locations_container.append(location)
                index_tri_container.append(index_tri)
                index_ray_container.append(index_ray)
                ray_directions_container.append(ray_directions)

                if i != bounces - 1:
                    # If at bounce number 1 compute the diffused directions:
                    if diffusion and i == 0:
                        diffusion_control = True

                    ray_origins, ray_directions, diffuse_directions = compute_secondary_bounce(
                        location, index_tri, mesh_obj, ray_directions, index_ray,
                        diffusion=diffusion_control, num_diffuse=num_diffuse)

                    # Set back to false the diffusion computation control flag
                    diffusion_control = False

        # Release memory
        # shape_model.scene.release() # Note: changed for performance optimization

    elif kernel == 'CGAL':

        # Initialize the geometry
        shape_model = cgal_init_geometry(mesh_obj)

        for i in range(bounces):
            ray_origins_container.append(ray_origins)

            # Avoid numerical problems
            ray_origins = ray_origins + 1e-3 * ray_directions

            # If computing bounce number 1 and the diffusion computation has been requested
            # do a separate raytracing also for the diffused rays
            if i == 1 and diffusion:
                ray_origins_diffusion = np.repeat(ray_origins, num_diffuse, axis=0)
                ray_directions_diffusion = diffuse_directions

                index_tri_diffusion, location_diffusion = shape_model.intersect1_2d_with_coords(
                    ray_origins_diffusion, ray_directions_diffusion)

                # Process diffusion results
                index_ray_diffusion = np.arange(len(ray_origins_diffusion))
                valid_diffusion = index_tri_diffusion > -1
                
                index_ray_diffusion = index_ray_diffusion[valid_diffusion]
                location_diffusion = location_diffusion[valid_diffusion]
                index_tri_diffusion = index_tri_diffusion[valid_diffusion]

                diffusion_pack = [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion[valid_diffusion],
                                  location_diffusion]

            # Main ray tracing
            index_tri, location = shape_model.intersect1_2d_with_coords(ray_origins, ray_directions)

            index_ray = np.arange(len(ray_origins))
            n_hits = len(np.where(index_tri > -1)[0])

            index_ray = index_ray[np.where(index_tri > -1)]
            location = location[np.where(index_tri > -1)]
            index_tri = index_tri[np.where(index_tri > -1)]

            # Handle: no bounces found
            if n_hits == 0:
                if errorMsg:
                    print('No intersections found for bounce {}. Results provided up to bounce {}'.format(i + 1, i))
                break

            # Otherwise append results and proceed with next bounce
            else:
                locations_container.append(location)
                index_tri_container.append(index_tri)
                index_ray_container.append(index_ray)
                ray_directions_container.append(ray_directions)

                if i != bounces - 1:
                    # If at bounce number 1 compute the diffused directions:
                    if diffusion and i == 0:
                        diffusion_control = True

                    ray_origins, ray_directions, diffuse_directions = compute_secondary_bounce(
                        location, index_tri, mesh_obj, ray_directions, index_ray,
                        diffusion=diffusion_control, num_diffuse=num_diffuse)

                    # Set back to false the diffusion computation control flag
                    diffusion_control = False

    else:
        print('No Recognized kernel')

    # Manage output variables
    if diffusion:
        return index_tri_container, index_ray_container, locations_container, ray_origins_container, ray_directions_container, diffusion_pack
    else:
        return index_tri_container, index_ray_container, locations_container, ray_origins_container, ray_directions_container, None
