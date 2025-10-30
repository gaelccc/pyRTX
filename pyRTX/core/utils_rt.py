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
    result : list of arrays
        List containing the chunked arrays. Chunks will be approximately equal
        in size, with the last chunk potentially being smaller.
    
    Notes
    -----
    Uses numpy.array_split which handles uneven divisions automatically.
    """
    return np.array_split(iterator, chunks)


def pxform_convert(pxform):
    """
    Convert a SPICE-generated rotation matrix (pxform) to the 4x4 homogeneous
    transformation matrix format required by trimesh.
    
    Parameters
    ----------
    pxform : array_like, shape (3, 3)
        The 3x3 rotation matrix from SPICE (obtained via spiceypy.pxform).
    
    Returns
    -------
    result : ndarray, shape (4, 4)
        The 4x4 homogeneous transformation matrix with the rotation in the
        upper-left 3x3 block, zeros in the translation column, and [0,0,0,0]
        in the bottom row.
    
    Notes
    -----
    Trimesh uses 4x4 homogeneous transformation matrices for geometric operations.
    This function adds the necessary padding to convert a pure rotation matrix.
    """
    pxform = np.array([pxform[0], pxform[1], pxform[2]], dtype=dFloat)

    p = np.append(pxform, [[0, 0, 0]], 0)

    mv = np.asarray(np.random.random(), dtype=dFloat)
    p = np.append(p, [[0], [0], [0], [0]], 1)
    return p


def block_normalize(V):
    """
    Compute unit vectors for a block of vectors efficiently using vectorized
    operations.
    
    Parameters
    ----------
    V : ndarray, shape (N, 3) or (3,)
        Array of vectors to normalize. Can be a single vector or multiple vectors.
    
    Returns
    -------
    result : ndarray, same shape as V
        The normalized vectors (unit vectors with magnitude 1).
    
    Notes
    -----
    Uses vectorized numpy operations for efficiency with large arrays. Handles
    both single vectors and arrays of vectors automatically.
    """

    if V.ndim > 1:
        return V / np.linalg.norm(V, axis=1).reshape(len(V), 1)
    else:
        return V / np.linalg.norm(V)


def block_dot(a, b):
    """
    Perform element-wise dot product between corresponding vectors in two arrays.
    
    Parameters
    ----------
    a : ndarray, shape (N, m)
        First array of vectors.
    b : ndarray, shape (N, m)
        Second array of vectors (must have same shape as a).
    
    Returns
    -------
    result : ndarray, shape (N,)
        Array containing the dot product of each pair of corresponding vectors.
        result[i] = a[i] · b[i]
    
    Notes
    -----
    This is more efficient than using a loop for computing many dot products.
    Equivalent to np.einsum('ij,ij->i', a, b) but more readable.
    """


    return np.sum(a * b, axis=1)


def pixel_plane(d0, lon, lat, width=1, height=1, ray_spacing=.1):
    """
    Generate a rectangular pixel array (grid of rays) for ray tracing, as defined
    in Li et al., 2018. This is the explicit implementation showing the full
    algorithm.
    
    This function creates a planar grid of ray origins and directions pointing
    toward a specified direction in 3D space, useful for simulating parallel
    light sources like the Sun.
    
    Parameters
    ----------
    d0 : float
        Distance of the pixel plane from the origin (in meters). This defines
        how far the ray origins are from the coordinate system origin.
    lon : float
        Longitude of the pixel plane's center direction (in radians). Defines
        the azimuthal angle in spherical coordinates.
    lat : float
        Latitude of the pixel plane's center direction (in radians). Defines
        the elevation angle in spherical coordinates.
    width : float, default=1
        Width of the plane (in meters). The plane extends ±width/2 in the
        horizontal direction.
    height : float, default=1
        Height of the plane (in meters). The plane extends ±height/2 in the
        vertical direction.
    ray_spacing : float, default=0.1
        Spacing between adjacent rays (in meters). Smaller values create denser
        ray grids but increase computation time.
    
    Returns
    -------
    locs : ndarray, shape (N, 3)
        Ray origin positions in 3D space. N = (width/ray_spacing + 1) × 
        (height/ray_spacing + 1).
    dirs : ndarray, shape (N, 3)
        Ray direction unit vectors. All rays point toward the origin (or away
        from the direction specified by lon/lat).
    
    Notes
    -----
    The pixel plane is oriented perpendicular to the direction vector defined
    by (lon, lat) and positioned at distance d0 from the origin. This creates
    a uniform grid of parallel rays suitable for simulating distant light sources.
    
    For performance-critical applications, use pixel_plane_opt instead.
    
    References
    ----------
    Li et al., 2018 - Solar radiation pressure modeling methodology
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
    Efficiently build a pixel array coordinate grid using Numba's JIT compilation
    for performance.
    
    Parameters
    ----------
    linsp1 : ndarray, shape (dim1,)
        Linear space defining positions along the first dimension (typically width).
    linsp2 : ndarray, shape (dim2,)
        Linear space defining positions along the second dimension (typically height).
    dim1 : int
        Number of points in the first dimension.
    dim2 : int
        Number of points in the second dimension.
    
    Returns
    -------
    result : ndarray, shape (dim1 × dim2, 3)
        Array of 3D coordinates forming a rectangular grid in the y-z plane
        (x=0 for all points). The grid is built by nested iteration over linsp1
        and linsp2.
    
    Notes
    -----
    This function is JIT-compiled with Numba for significant performance improvement
    over pure Python loops. Used internally by pixel_plane_opt.
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
    Generate a rectangular pixel array for ray tracing - optimized version with
    optional ray packet subdivision.
    
    This is a performance-optimized implementation of pixel_plane that uses
    vectorized operations and Numba JIT compilation. It also supports dividing
    the rays into packets to avoid segmentation faults with very large ray counts.
    
    Parameters
    ----------
    d0 : float
        Distance of the pixel plane from the origin (in meters).
    lon : float
        Longitude of the pixel plane's center direction (in radians).
    lat : float
        Latitude of the pixel plane's center direction (in radians).
    width : float, default=1
        Width of the plane (in meters).
    height : float, default=1
        Height of the plane (in meters).
    ray_spacing : float, default=0.1
        Spacing between adjacent rays (in meters). Determines ray grid density.
    packets : int, default=1
        Number of ray packets to subdivide the rays into. Use values > 1 to
        avoid segmentation faults or memory issues with very large numbers of
        rays (typically > 10^6). Each packet is processed separately by the
        ray tracer.
    
    Returns
    -------
    locs : ndarray or list of ndarrays
        Ray origin positions. If packets=1, returns single array of shape (N, 3).
        If packets>1, returns list of arrays, each containing a subset of rays.
    dirs : ndarray or list of ndarrays
        Ray direction unit vectors. Same structure as locs.
    
    Notes
    -----
    This is the recommended function for pixel plane generation due to its
    performance optimizations. Use packets > 1 when dealing with very dense
    ray grids (small ray_spacing values).
    
    The function uses Numba JIT compilation via fast_vector_build for efficient
    coordinate grid generation.
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
    Compute reflected ray directions given incoming rays and surface normals.
    Uses the law of reflection: r = i - 2(i·n)n
    
    This is a vectorized implementation using numpy.einsum for efficient
    computation of many reflections simultaneously.
    
    Parameters
    ----------
    incoming : ndarray, shape (N, 3)
        Incoming ray direction vectors (do not need to be normalized).
    normal : ndarray, shape (N, 3)
        Surface normal vectors at reflection points (should be unit vectors).
    
    Returns
    -------
    reflected : ndarray, shape (N, 3)
        Reflected ray direction vectors. These are NOT normalized - maintain
        the same magnitude as the incoming vectors.
    
    Notes
    -----
    The reflection formula used is: r = i - 2(i·n)n, where:
    - i is the incoming direction
    - n is the surface normal
    - r is the reflected direction
    
    This formula gives the specular (mirror-like) reflection direction.
    Uses np.einsum for efficient vectorized dot product computation.
    """

    return incoming - 2 * np.multiply(normal.T, np.einsum('ij,ij->i', incoming, normal)).T


@jit(nopython=True)
def get_orthogonal(v):
    """
    Generate a unit vector orthogonal to the input vector using randomization.
    
    Parameters
    ----------
    v : ndarray, shape (3,)
        Input vector to which the result should be orthogonal.
    
    Returns
    -------
    x : ndarray, shape (3,)
        Unit vector orthogonal to v (x · v = 0 and ||x|| = 1).
    
    Notes
    -----
    Uses a random vector projection method: generates a random 3D vector,
    projects out the component parallel to v, and normalizes. This is used
    internally for constructing local coordinate systems on surface normals.
    
    JIT-compiled with Numba for performance.
    """

    x = np.random.random(3)
    x -= x.dot(v) * v
    x = x / np.linalg.norm(x)

    return x


@jit(nopython=True)
def sample_lambert_dist(normal, num=100):
    """
    Generate a cloud of direction vectors following the Lambert cosine
    distribution (also known as Lambertian distribution or cosine-weighted
    hemisphere sampling).
    
    The Lambert distribution models ideal diffuse reflection where the
    probability of a ray being reflected in a given direction is proportional
    to the cosine of the angle between that direction and the surface normal.
    This is physically accurate for perfectly diffuse (Lambertian) surfaces.
    
    Parameters
    ----------
    normal : ndarray, shape (3,)
        Surface normal vector defining the hemisphere orientation (should be
        a unit vector).
    num : int, default=100
        Number of sample directions to generate.
    
    Returns
    -------
    v : ndarray, shape (num, 3)
        Array of sampled direction vectors distributed according to Lambert's
        cosine law. All vectors point into the hemisphere defined by the normal.
    
    Notes
    -----
    The sampling uses spherical coordinates with:
    - θ (polar angle): sampled as θ = arccos(√ξ) where ξ ~ U(0,1)
    - ψ (azimuthal angle): sampled uniformly as ψ ~ U(0, 2π)
    
    This ensures that the probability density is proportional to cos(θ), which
    is characteristic of ideal diffuse reflection (Lambert's cosine law).
    
    JIT-compiled with Numba for performance. Used for modeling diffuse
    reflection in ray tracing.
    
    References
    ----------
    Lambert's Cosine Law: I = I₀ cos(θ) where θ is the angle from the normal
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
    for i, n in enumerate(normals):
        diff_dirs = sample_lambert_dist(n, num=num)

        diffuse_directions[i * num: (i + 1) * num] = diff_dirs
    return diffuse_directions


def diffuse(normals, num=10):
    """
    Compute multiple diffuse reflection directions for an array of surface
    normals by sampling the Lambert cosine distribution.
    
    For each input normal, generates num diffusely reflected ray directions
    following Lambert's cosine law. This models realistic diffuse scattering
    from rough surfaces.
    
    Parameters
    ----------
    normals : ndarray, shape (N, 3)
        Array of surface normal unit vectors at reflection points.
    num : int, default=10
        Number of diffuse samples to generate for each normal. Higher values
        give more accurate diffuse reflection modeling but increase computation.
    
    Returns
    -------
    diffuse_directions : ndarray, shape (N × num, 3)
        Array of sampled diffuse direction vectors. For each of the N input
        normals, generates num directions, resulting in N×num total directions.
        Directions are ordered so that directions [i×num : (i+1)×num] correspond
        to normal i.
    
    Notes
    -----
    This function is used to model non-specular (rough) surface reflections.
    Each diffuse direction is randomly sampled from the hemisphere above the
    surface, weighted by the cosine of the angle from the normal (Lambert's law).
    
    The returned array can be fed directly into the ray tracer to simulate
    secondary illumination from diffuse reflections.
    
    Example
    -------
    >>> normals = np.array([[0, 0, 1], [0, 1, 0]])  # Two normals
    >>> dirs = diffuse(normals, num=5)  # 5 samples each
    >>> dirs.shape
    (10, 3)  # 2 normals × 5 samples = 10 directions
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
    Prepare ray origins and directions for subsequent ray tracing bounces by
    computing specular and optionally diffuse reflection directions.
    
    This function takes the results of a ray-surface intersection and computes
    the reflected rays needed for the next bounce iteration in multi-bounce
    ray tracing.
    
    Parameters
    ----------
    location : ndarray, shape (N_hits, 3)
        3D coordinates of ray-surface intersection points.
    index_tri : ndarray, shape (N_hits,)
        Indices of the mesh faces (triangles) that were intersected by rays.
    mesh_obj : trimesh.Trimesh
        The mesh object containing geometry information (vertices, faces, normals).
    ray_directions : ndarray, shape (N_rays, 3)
        Direction vectors of the incident rays (before intersection).
    index_ray : ndarray, shape (N_hits,)
        Indices of the rays that successfully intersected the mesh. Used to
        map from the original ray array to the subset that hit surfaces.
    diffusion : bool, default=False
        If True, compute diffuse reflection directions in addition to specular
        reflections. Used for modeling rough surface scattering.
    num_diffuse : int or None, default=None
        Number of diffuse samples per intersection point. Required if
        diffusion=True, ignored otherwise.
    
    Returns
    -------
    location : ndarray, shape (N_hits, 3)
        Same as input location (pass-through for convenience).
    reflect_dirs : ndarray, shape (N_hits, 3)
        Specularly reflected ray directions for each intersection point.
        Computed using the law of reflection with surface normals.
    diffuse_dirs : ndarray, shape (N_hits × num_diffuse, 3) or int
        If diffusion=True: array of diffusely reflected directions sampled
        from Lambert distribution for each intersection point.
        If diffusion=False: returns -1 (dummy value for consistent return signature).
    
    Notes
    -----
    This function is typically called iteratively in multi-bounce ray tracing:
    1. First bounce: rays from source hit surface
    2. Compute secondary bounces from intersection points
    3. Trace secondary rays
    4. Repeat for N bounces
    
    The specular reflections follow the law of reflection, while diffuse
    reflections sample the Lambert cosine distribution, providing physically
    accurate modeling of both mirror-like and rough surfaces.
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
    Save ray tracing results to a pickled dictionary for post-processing and
    visualization.
    
    Creates a standardized output format that can be loaded by visualization
    scripts (see visual_utils module) for analyzing ray tracing results,
    creating visualizations, and debugging ray path geometries.
    
    Parameters
    ----------
    outputFilePath : str
        Path to output file. Should end with '.pkl' extension. Parent directory
        must exist.
    mesh : trimesh.Trimesh
        The mesh object that was ray-traced. Contains geometry (vertices, faces,
        normals) for visualization.
    ray_origins : list of ndarrays
        List containing ray origin arrays for each bounce. Each element is an
        array of shape (N_rays_i, 3) where i is the bounce number.
    ray_directions : list of ndarrays
        List containing ray direction arrays for each bounce. Same structure
        as ray_origins.
    location : list of ndarrays
        List containing intersection point coordinates for each bounce. Each
        element is shape (N_hits_i, 3).
    index_tri : list of ndarrays
        List containing indices of intersected triangles for each bounce.
        Each element is shape (N_hits_i,).
    diffusion_pack : list or None
        Diffuse ray tracing data if computed, otherwise None. Contains:
        [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,
         location_diffusion] for visualization of diffuse scattering.
    
    Returns
    -------
    None
        Data is written to file at outputFilePath.
    
    Notes
    -----
    The output file contains a dictionary with keys:
    - 'mesh': trimesh.Trimesh object
    - 'ray_origins': list of origin arrays per bounce
    - 'ray_directions': list of direction arrays per bounce  
    - 'locations': list of intersection point arrays per bounce
    - 'index_tri': list of triangle index arrays per bounce
    - 'diffusion_pack': diffuse ray data or None
    
    This standardized format allows visualization scripts to recreate the full
    ray tracing geometry including multiple bounces and diffuse scattering.
    
    The file is saved using pickle protocol 4 for Python 3 compatibility.
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
    Initialize mesh geometry for ray tracing using the Embree 3 ray tracing kernel.
    
    Converts a trimesh mesh object into an EmbreeTrimeshShapeModel that can be
    efficiently ray-traced using Intel's Embree library. Precomputes surface
    normals and face areas for performance.
    
    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        Input mesh object containing vertices and faces. Must be a valid
        triangular mesh.
    
    Returns
    -------
    scene : EmbreeTrimeshShapeModel
        Shape model object containing:
        - V: vertex coordinates array
        - F: face index array
        - N: face normal vectors
        - A: face areas
        - scene: Embree scene object for ray tracing
    
    Notes
    -----
    This function performs initial geometry setup required by Embree:
    1. Extracts vertices (V) and faces (F) from mesh
    2. Computes face normals (N) and areas (A)
    3. Creates Embree device, geometry, and scene objects
    4. Loads vertex and index buffers into Embree
    
    The returned object can be used with Embree's ray intersection functions
    for high-performance ray tracing. Embree uses hardware-accelerated BVH
    (Bounding Volume Hierarchy) structures for fast ray-triangle intersections.
    
    NOTE: The Embree 3 wrapper functions were developed by Sam Potter
    (https://github.com/sampotter/python-embree)
    
    See Also
    --------
    EmbreeTrimeshShapeModel : The shape model class
    RTXkernel : Main ray tracing interface
    """
    V = np.array(mesh_obj.vertices, dtype=np.float64)
    F = np.array(mesh_obj.faces, dtype=np.int64)

    # P = get_centroids(V, F)
    N, A = get_surface_normals_and_face_areas(V, F)

    return EmbreeTrimeshShapeModel(V, F, N=N, A=A)


def Embree3_init_rayhit(ray_origins, ray_directions):
    """
    Initialize Embree 3 RayHit data structure for ray tracing queries.
    
    Creates and configures an Embree RayHit1M object that stores ray information
    (origins, directions, parameters) and will be populated with hit information
    (intersection distances, geometry IDs) by the ray tracer.
    
    Parameters
    ----------
    ray_origins : ndarray, shape (N, 3)
        Starting positions of rays in 3D space.
    ray_directions : ndarray, shape (N, 3)
        Direction vectors of rays (do not need to be normalized; Embree handles
        this internally).
    
    Returns
    -------
    rayhit : embree.RayHit1M
        Initialized RayHit structure containing:
        - org: ray origin coordinates (set from ray_origins)
        - dir: ray direction vectors (set from ray_directions)
        - tnear: minimum ray parameter (set to 0.0 to trace from origin)
        - tfar: maximum ray parameter (set to infinity for unbounded rays)
        - prim_id: primitive (triangle) ID (initialized to INVALID, filled by tracer)
        - geom_id: geometry ID (initialized to INVALID, filled by tracer)
    
    Notes
    -----
    The RayHit1M structure supports tracing multiple rays simultaneously (the "1M"
    indicates "1 Million" rays capability). After calling Embree's intersect
    function, the structure will contain:
    - Updated tnear/tfar values indicating intersection distances
    - prim_id: index of intersected triangle (-1 if no hit)
    - geom_id: geometry identifier (-1 if no hit)
    - uv: barycentric coordinates of intersection point within triangle
    
    The tnear parameter is set to 0.0 rather than a small epsilon to avoid
    missing intersections, but this may cause numerical issues with very close
    surfaces. Adjust if needed for specific applications.
    """
    nb = np.shape(ray_origins)[0]  # Number of tracked rays
    rayhit = embree.RayHit1M(nb)

    # Initialize the ray structure
    # rayhit.tnear[:] = 0.001 #Avoid numerical problems
    rayhit.tnear[:] = 0.00  # Avoid numerical problems
    rayhit.tfar[:] = np.inf
    rayhit.prim_id[:] = embree.INVALID_GEOMETRY_ID
    rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
    rayhit.org[:] = ray_origins
    rayhit.dir[:] = ray_directions

    return rayhit


def Embree3_dump_solution(rayhit, V, F):
    """
    Extract and process ray-surface intersection results from Embree RayHit structure.
    
    After Embree's ray tracing kernel completes, this function extracts the
    intersection data and converts it into standard numpy arrays. It computes
    actual 3D intersection point coordinates from barycentric coordinates.
    
    Parameters
    ----------
    rayhit : embree.RayHit1M
        RayHit structure populated by Embree's intersect function, containing
        primitive IDs, geometry IDs, barycentric coordinates, etc.
    V : ndarray, shape (N_vertices, 3)
        Vertex coordinates of the mesh.
    F : ndarray, shape (N_faces, 3)
        Face indices of the mesh (each row contains indices of 3 vertices
        forming a triangle).
    
    Returns
    -------
    hits : ndarray, shape (N_hits,) or int
        Indices of triangles that were intersected. Returns -1 if no hits.
    nhits : int
        Number of rays that intersected the mesh. Returns -1 if no hits.
    idh : ndarray, shape (N_hits,) or int
        Indices of rays that successfully hit the mesh (mapping from original
        ray array to hit subset). Returns -1 if no hits.
    Ph : ndarray, shape (N_hits, 3) or int
        3D coordinates of intersection points computed from barycentric
        coordinates. Returns -1 if no hits.
    
    Notes
    -----
    The function identifies valid hits by checking if prim_id != INVALID_GEOMETRY_ID.
    
    For valid hits, intersection points are computed using barycentric interpolation:
        Ph = v1 + (v2 - v1) * u + (v3 - v1) * v
    where:
    - v1, v2, v3 are the triangle vertices
    - u, v are barycentric coordinates from rayhit.uv
    - Ph is the 3D intersection point
    
    If no intersections occurred (nhits=0), all return values are -1 to indicate
    no valid data.
    """

    ishit = rayhit.prim_id != embree.INVALID_GEOMETRY_ID
    idh = np.nonzero(ishit)[0]
    hits = rayhit.prim_id[idh]
    nhits = hits.size

    if nhits > 0:
        p = V[F[hits]]
        v1 = p[:, 0]
        v2 = p[:, 1]
        v3 = p[:, 2]
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
    Initialize mesh geometry for ray tracing using the CGAL (Computational Geometry
    Algorithms Library) ray tracing kernel.
    
    Converts a trimesh mesh object into a CgalTrimeshShapeModel that uses CGAL's
    AABB (Axis-Aligned Bounding Box) tree for efficient ray-triangle intersections.
    
    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        Input mesh object containing vertices and faces.
    
    Returns
    -------
    CgalTrimeshShapeModel : shape model object
        Shape model configured for CGAL ray tracing, containing:
        - V: vertex coordinates
        - F: face indices
        - N: face normals
        - A: face areas
        - aabb: CGAL AABB tree structure for fast queries
    
    Notes
    -----
    CGAL is a C++ library providing robust geometric algorithms. The AABB tree
    structure enables efficient ray tracing through hierarchical spatial subdivision.
    
    This function is adapted from python-flux. CGAL may be more robust than
    Embree for certain edge cases (nearly degenerate triangles, numerical precision
    issues) but is typically slower for large numbers of rays.
    
    See Also
    --------
    CgalTrimeshShapeModel : The CGAL-based shape model class
    RTXkernel : Main ray tracing interface that can use CGAL kernel
    """
    V = np.array(mesh_obj.vertices, dtype=np.float64)
    F = np.array(mesh_obj.faces, dtype=np.int64)

    N, A = get_surface_normals_and_face_areas(V, F)

    return CgalTrimeshShapeModel(V, F, N=N, A=A)


######################### from python-flux.src.flux.shape.py

def get_centroids(V, F):
    """
    Compute the geometric centroids of all triangular faces in a mesh.
    
    The centroid of a triangle is the arithmetic mean of its three vertices,
    representing the triangle's center of mass (assuming uniform density).
    
    Parameters
    ----------
    V : ndarray, shape (N_vertices, 3)
        Vertex coordinates of the mesh.
    F : ndarray, shape (N_faces, 3)
        Face indices. Each row contains three vertex indices forming a triangle.
    
    Returns
    -------
    P : ndarray, shape (N_faces, 3)
        Centroid coordinates for each face. P[i] = (V[F[i][0]] + V[F[i][1]] +
        V[F[i][2]]) / 3
    
    Notes
    -----
    Uses vectorized numpy operations: V[F] creates shape (N_faces, 3, 3) array
    where V[F][i] is the 3×3 matrix of vertices for face i. Taking mean along
    axis=1 computes centroids efficiently for all faces simultaneously.

    """

    return V[F].mean(axis=1)


def get_cross_products(V, F):
    """
    Compute cross products of edge vectors for all triangular faces in a mesh.
    
    For each triangle, computes the cross product of two edge vectors. The
    magnitude of this cross product equals twice the triangle's area, and its
    direction is perpendicular to the triangle plane (unnormalized normal).
    
    Parameters
    ----------
    V : ndarray, shape (N_vertices, 3)
        Vertex coordinates of the mesh.
    F : ndarray, shape (N_faces, 3)
        Face indices. Each row contains three vertex indices [v0, v1, v2].
    
    Returns
    -------
    C : ndarray, shape (N_faces, 3)
        Cross product vectors for each face. C[i] = (v1 - v0) × (v2 - v0)
        where v0, v1, v2 are the vertices of triangle i.
    
    Notes
    -----
    Used internally by get_face_areas and get_surface_normals for efficient
    vectorized computation of geometric properties.
    """
    V0 = V[F][:, 0, :]
    C = np.cross(V[F][:, 1, :] - V0, V[F][:, 2, :] - V0)
    return C


def get_face_areas(V, F):
    """
    Compute the areas of all triangular faces in a mesh.
    
    Parameters
    ----------
    V : ndarray, shape (N_vertices, 3)
        Vertex coordinates of the mesh.
    F : ndarray, shape (N_faces, 3)
        Face indices.
    
    Returns
    -------
    A : ndarray, shape (N_faces,)
        Area of each face in the same units as V (e.g., if V is in meters,
        areas are in square meters).
    
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
    V : ndarray, shape (N_vertices, 3)
        Vertex coordinates of the mesh.
    F : ndarray, shape (N_faces, 3)
        Face indices.
    
    Returns
    -------
    N : ndarray, shape (N_faces, 3)
        Unit normal vectors perpendicular to each face. Direction follows
        right-hand rule with respect to vertex ordering in F.
    

 
    """

    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C ** 2, axis=1))
    N = C / C_norms.reshape(C.shape[0], 1)
    return N


def get_surface_normals_and_face_areas(V, F):
    """
    Efficiently compute both surface normals and face areas simultaneously.
    
    This is more efficient than calling get_surface_normals and get_face_areas
    separately because it computes the cross products only once.
    
    Parameters
    ----------
    V : ndarray, shape (N_vertices, 3)
        Vertex coordinates of the mesh.
    F : ndarray, shape (N_faces, 3)
        Face indices.
    
    Returns
    -------
    N : ndarray, shape (N_faces, 3)
        Unit normal vectors for each face.
    A : ndarray, shape (N_faces,)
        Area of each face.
    
    Notes
    -----
    Computation steps:
    1. Compute cross products C = (v1 - v0) × (v2 - v0)
    2. Compute magnitudes ||C||
    3. Normals: N = C / ||C||
    4. Areas: A = ||C|| / 2
    
    This is the recommended function when both quantities are needed, as it
    avoids redundant cross product calculations.
    """
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C ** 2, axis=1))
    N = C / C_norms.reshape(C.shape[0], 1)
    A = C_norms / 2
    return N, A


class ShapeModel(ABC):
    pass


class TrimeshShapeModel(ShapeModel):
    """A shape model consisting of a single triangle mesh."""

    def __init__(self, V, F, N=None, P=None, A=None):
        """Initialize a triangle mesh shape model. No assumption is made about
        the way vertices or faces are stored when building the shape
        model except that V[F] yields the faces of the mesh. Vertices
        may be repeated or not.

        Parameters
        ----------
        V : array_like
            An array with shape (num_verts, 3) whose rows correspond to the
            vertices of the triangle mesh
        F : array_like
            An array with shape (num_faces, 3) whose rows index the faces
            of the triangle mesh (i.e., V[F] returns an array with shape
            (num_faces, 3, 3) such that V[F][i] is a 3x3 matrix whose rows
            are the vertices of the ith face.
        N : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            mesh face normals. Can be passed to specify the face normals.
            Otherwise, the face normals will be computed from the cross products
            of the face edges (i.e. np.cross(vi1 - vi0, vi2 - vi0) normalized).
        P : array_like, optional
            An array with shape (num_faces, 3) consisting of the triangle
            centroids. Can be optionally passed to avoid recomputing.
        A : array_like, optional
            An array of shape (num_faces,) containing the triangle areas. Can
            be optionally passed to avoid recomputing.

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
        return (self.__class__, (self.V, self.F, self.N, self.P, self.A))

    def __repr__(self):
        return 'a TrimeshShapeModel with %d vertices and %d faces' % (
            self.num_verts, self.num_faces)

    @property
    def num_faces(self):
        return self.P.shape[0]

    @property
    def num_verts(self):
        return self.V.shape[0]

    def intersect1(self, x, d):
        '''Trace a single ray starting from `x` and in the direction `d`.  If
        there is a hit, return the index (`i`) of the hit and a
        parameter `t` such that the hit point is given by `x(t) = x +
        t*d`.

        '''
        return self._intersect1(x, d)

    def intersect1_2d_with_coords(self, X, D):
        '''Trace a single ray starting from `X` and in the direction `D`.  If
        there is a hit, return the index (`i`) of the hit and the coordinates of
        the centroid of the hit triangle `X(t) = X + t*D`.

        '''
        return self._intersect1_2d_with_coords(X, D)

    def intersect1_2d(self, X, D):
        '''Trace a single ray starting from `X` and in the direction `D`.  If
        there is a hit, return the index (`i`).
        '''

        fint, xta = self.intersect1_2d_with_coords(X, D)

        return fint

class CgalTrimeshShapeModel(TrimeshShapeModel):
    def _make_scene(self):
        self.aabb = AABB.from_trimesh(
            self.V.astype(np.float64), self.F.astype(np.uintp))

    def _intersect1(self, x, d):
        return self.aabb.intersect1(x, d)

    def _intersect1_2d(self, X, D):
        return self.aabb.intersect1_2d(X, D)

    def _intersect1_2d_with_coords(self, X, D):
        return self.aabb.intersect1_2d_with_coords(X, D)

class EmbreeTrimeshShapeModel(TrimeshShapeModel):
    def _make_scene(self):
        '''Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.

        '''
        device = embree.Device()
        geometry = device.make_geometry(embree.GeometryType.Triangle)
        # geometry.set_build_quality(embree.BuildQuality.High)

        scene = device.make_scene()
        # scene.set_build_quality(embree.BuildQuality.High)
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
#     Main ray tracing kernel wrapper supporting multiple ray tracing backends and
#     multi-bounce simulations.
    
#     This is the primary interface for performing ray tracing operations in pyRTX.
#     It supports various ray tracing kernels (Embree, CGAL, Native), multiple
#     reflection bounces, and optional diffuse scattering for realistic radiation
#     momentum exchange calculatons.
    
#     Parameters
#     ----------
#     mesh_obj : trimesh.Trimesh
#         The mesh geometry to ray trace against. Must be a valid triangular mesh.
#     ray_origins : ndarray, shape (N_rays, 3)
#         Starting positions of rays in 3D space (in same coordinate system as mesh).
#     ray_directions : ndarray, shape (N_rays, 3)
#         Direction vectors of rays. Do not need to be normalized.
#     bounces : int, default=1
#         Number of reflection bounces to simulate. bounces=1 means direct
#         illumination only, bounces=2 includes one reflection, etc.
#     kernel : str, default='Embree'
#         Ray tracing backend to use:
#         - 'Embree3': Intel Embree library (fastest, recommended)
#         - 'Embree' : Intel Embree library (Version 2, slower than 3)
#         - 'CGAL': CGAL AABB tree implementation (robust, slower)
#         - 'Native': Pure Python implementation (very slow, for reference only)
#     diffusion : bool, default=False
#         If True, compute diffuse (Lambertian) reflections in addition to
#         specular reflections. Only applied to the first bounce. Enables
#         realistic modeling of rough surfaces.
#     num_diffuse : int or None, default=None
#         Number of diffuse samples per intersection point. Required if
#         diffusion=True. Typical values: 10-100 depending on accuracy needs.
#     errorMsg : bool, default=True
#         If True, print warning messages when no intersections are found for
#         a bounce. Set to False to suppress warnings in batch processing.
    
#     Returns
#     -------
#     index_tri_container : list of ndarrays
#         List containing triangle indices for each bounce. Each element is an
#         array of shape (N_hits_i,) containing indices of faces hit at bounce i.
#         Length equals number of computed bounces (≤ bounces parameter).
#     index_ray_container : list of ndarrays
#         List containing ray indices for each bounce. index_ray_container[i]
#         maps from the original ray array to rays that hit at bounce i.
#     locations_container : list of ndarrays
#         List of intersection point coordinates for each bounce. Each element
#         has shape (N_hits_i, 3).
#     ray_origins_container : list of ndarrays
#         List of ray origin positions for each bounce. Note that
#         ray_origins_container[0] equals the input ray_origins parameter.
#     ray_directions_container : list of ndarrays
#         List of ray direction vectors for each bounce. Element [0] contains
#         input directions, subsequent elements contain reflected directions.
#     diffusion_pack : list or None
#         If diffusion=True, contains diffuse ray tracing results:
#         [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,
#          location_diffusion]. If diffusion=False, returns None.
    
#     Notes
#     -----
#     Algorithm Overview:
#     1. Initialize ray tracing kernel (Embree, CGAL, or Native)
#     2. For each bounce:
#        a. Add small offset to ray origins (avoids self-intersection)
#        b. Trace rays to find intersections
#        c. If no hits found, terminate and return results up to current bounce
#        d. Compute specular reflection directions for next bounce
#        e. If diffusion enabled and bounce==1, also compute diffuse reflections
#     3. Return accumulated results for all bounces
    
#     Performance Notes:
#     - Embree is typically 10-100× faster than Native for large meshes
#     - CGAL offers good robustness for edge cases but slower than Embree
#     - Diffusion increases computation by factor of num_diffuse
#     - Memory usage scales linearly with bounces and num_diffuse
    
#     Kernel-Specific Details:
#     - Embree/Embree3: Uses BVH acceleration, highly optimized for x86 CPUs
#     - CGAL: Uses AABB tree, more robust numerical handling
#     - Native: Pure Python/Trimesh, no special acceleration
    
#     The 1e-3 offset added to ray origins prevents numerical precision issues
#     where a reflected ray might re-intersect the same surface it just bounced
#     from (self-intersection artifact).
    
#     Examples
#     --------
#     >>> # Simple direct illumination
#     >>> results = RTXkernel(mesh, origins, directions, bounces=1, kernel='Embree')
#     >>> hits, ray_ids, locations, _, _, _ = results
#     >>> 
#     >>> # Multi-bounce with diffuse scattering
#     >>> results = RTXkernel(mesh, origins, directions, bounces=3, 
#     ...                     kernel='Embree', diffusion=True, num_diffuse=50)
#     >>> hits, ray_ids, locs, origins, dirs, diffuse_data = results
    
#     See Also
#     --------
#     pixel_plane_opt : Generate ray grids for illumination sources
#     compute_secondary_bounce : Compute reflection directions
#     diffuse : Generate diffuse reflection samples
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
    Main ray tracing kernel wrapper supporting multiple ray tracing backends and
    multi-bounce simulations.
    
    This is the primary interface for performing ray tracing operations in pyRTX.
    It supports various ray tracing kernels (Embree, CGAL, Native), multiple
    reflection bounces, and optional diffuse scattering for realistic radiation
    momentum exchange calculations.
    
    Parameters
    ----------
    mesh_obj : trimesh.Trimesh
        The mesh geometry to ray trace against. Must be a valid triangular mesh.
    ray_origins : ndarray, shape (N_rays, 3)
        Starting positions of rays in 3D space (in same coordinate system as mesh).
    ray_directions : ndarray, shape (N_rays, 3)
        Direction vectors of rays. Do not need to be normalized.
    bounces : int, default=1
        Number of reflection bounces to simulate. bounces=1 means direct
        illumination only, bounces=2 includes one reflection, etc.
    kernel : str, default='Embree3'
        Ray tracing backend to use:
        - 'Embree3': Intel Embree library (fastest, recommended)
        - 'Embree' : Intel Embree library (Version 2, slower than 3)
        - 'CGAL': CGAL AABB tree implementation (robust, slower)
        - 'Native': Pure Python implementation (very slow, for reference only)
    diffusion : bool, default=False
        If True, compute diffuse (Lambertian) reflections in addition to
        specular reflections. Only applied to the first bounce. Enables
        realistic modeling of rough surfaces.
    num_diffuse : int or None, default=None
        Number of diffuse samples per intersection point. Required if
        diffusion=True. Typical values: 10-100 depending on accuracy needs.
    errorMsg : bool, default=True
        If True, print warning messages when no intersections are found for
        a bounce. Set to False to suppress warnings in batch processing.
    
    Returns
    -------
    index_tri_container : list of ndarrays
        List containing triangle indices for each bounce. Each element is an
        array of shape (N_hits_i,) containing indices of faces hit at bounce i.
        Length equals number of computed bounces (≤ bounces parameter).
    index_ray_container : list of ndarrays
        List containing ray indices for each bounce. index_ray_container[i]
        maps from the original ray array to rays that hit at bounce i.
    locations_container : list of ndarrays
        List of intersection point coordinates for each bounce. Each element
        has shape (N_hits_i, 3).
    ray_origins_container : list of ndarrays
        List of ray origin positions for each bounce. Note that
        ray_origins_container[0] equals the input ray_origins parameter.
    ray_directions_container : list of ndarrays
        List of ray direction vectors for each bounce. Element [0] contains
        input directions, subsequent elements contain reflected directions.
    diffusion_pack : list or None
        If diffusion=True, contains diffuse ray tracing results:
        [index_tri_diffusion, index_ray_diffusion, ray_directions_diffusion,
         location_diffusion]. If diffusion=False, returns None.
    
    Notes
    -----
    Algorithm Overview:
    1. Initialize ray tracing kernel (Embree, CGAL, or Native)
    2. For each bounce:
       a. Add small offset to ray origins (avoids self-intersection)
       b. Trace rays to find intersections
       c. If no hits found, terminate and return results up to current bounce
       d. Compute specular reflection directions for next bounce
       e. If diffusion enabled and bounce==1, also compute diffuse reflections
    3. Return accumulated results for all bounces
    
    Performance Notes:
    - Embree is typically 10-100× faster than Native for large meshes
    - CGAL offers good robustness for edge cases but slower than Embree
    - Diffusion increases computation by factor of num_diffuse
    - Memory usage scales linearly with bounces and num_diffuse
    
    Kernel-Specific Details:
    - Embree/Embree3: Uses BVH acceleration, highly optimized for x86 CPUs
    - CGAL: Uses AABB tree, more robust numerical handling
    - Native: Pure Python/Trimesh, no special acceleration
    
    The 1e-3 offset added to ray origins prevents numerical precision issues
    where a reflected ray might re-intersect the same surface it just bounced
    from (self-intersection artifact).
    
    Examples
    --------
    >>> # Simple direct illumination
    >>> results = RTXkernel(mesh, origins, directions, bounces=1, kernel='Embree3')
    >>> hits, ray_ids, locations, _, _, _ = results
    >>> 
    >>> # Multi-bounce with diffuse scattering
    >>> results = RTXkernel(mesh, origins, directions, bounces=3, 
    ...                     kernel='Embree3', diffusion=True, num_diffuse=50)
    >>> hits, ray_ids, locs, origins, dirs, diffuse_data = results
    
    See Also
    --------
    pixel_plane_opt : Generate ray grids for illumination sources
    compute_secondary_bounce : Compute reflection directions
    diffuse : Generate diffuse reflection samples
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
    if kernel in ['Embree', 'Native']:
        for i in range(bounces):

            ray_origins_container.append(ray_origins)

            if kernel == 'Embree':
                intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_obj)

            elif kernel == 'Native':
                intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_obj)

            # Avoid numerical problems
            ray_origins = ray_origins + 1e-3 * ray_directions

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
        shape_model = Embree3_init_geometry(mesh_obj)
        context = embree.IntersectContext()

        for i in range(bounces):
            ray_origins_container.append(ray_origins)

            # Avoid numerical problems
            ray_origins = ray_origins + 1e-3 * ray_directions

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
        shape_model.scene.release()

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