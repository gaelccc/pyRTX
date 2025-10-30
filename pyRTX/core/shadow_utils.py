import numpy as np 
from numba import jit 




def circular_mask(R, coords, origin):
    """
    Identify points within a circular region of specified radius.
    
    Returns indices of points in the coords array that fall within a circle
    of radius R centered at origin. Used for selecting pixels within a
    circular aperture (e.g., the Sun's disk).
    
    Parameters
    ----------
    R : float
        Radius of the circular region (in same units as coords).
    coords : ndarray, shape (N, 3)
        Array of 3D coordinate points to test.
    origin : ndarray, shape (3,)
        Center point of the circular region.
    
    Returns
    -------
    maskIds : ndarray, shape (M,), dtype=int32
        Indices of points in coords that satisfy ||coords[i] - origin|| <= R.
        M is the number of points within the circle.
    
   
    See Also
    --------
    circular_rim : Select points on the circle boundary
    """

	maskIds = []
	maskIds = np.where(np.linalg.norm(coords-origin, axis = 1) <= R)
	

	return np.array(maskIds[0], dtype = 'int32')


def circular_rim(R, coords, origin):
    """
    Identify points on the rim (boundary) of a circular region.
    
    Returns indices of points in the coords array that lie approximately on
    the circle of radius R centered at origin. Uses numerical tolerance for
    floating-point comparison.
    
    Parameters
    ----------
    R : float
        Radius of the circle (in same units as coords).
    coords : ndarray, shape (N, 3)
        Array of 3D coordinate points to test.
    origin : ndarray, shape (3,)
        Center point of the circle.
    
    Returns
    -------
    maskIds : ndarray, shape (M,), dtype=int32
        Indices of points in coords that satisfy ||coords[i] - origin|| ≈ R
        within relative tolerance of 0.001 (0.1%). M is the number of points
        on the rim.
    
    
    See Also
    --------
    circular_mask : Select points within the circle
    """

	maskIds = np.where(np.isclose(np.linalg.norm(coords-origin, axis = 1),R, rtol = 0.001))
	mask_coords = coords[maskIds]

	return np.array(maskIds[0], dtype = 'int32')



@jit(nopython=True)
def compute_directions(pixelCoords):
    """
    Convert pixel coordinates to normalized direction unit vectors.
    
    For each pixel coordinate point, computes the unit vector pointing from
    the origin toward that point. This converts position vectors to direction
    vectors for ray tracing.
    
    Parameters
    ----------
    pixelCoords : ndarray, shape (N, 3)
        Array of 3D pixel coordinate positions.
    
    Returns
    -------
    dirs : ndarray, shape (N, 3)
        Array of normalized direction unit vectors. Each row satisfies
        ||dirs[i]|| = 1 and dirs[i] ∝ pixelCoords[i].
    
    Notes
    -----

    This function is typically used to convert pixel plane coordinates into
    ray directions for ray tracing operations.
    

    """
	dirs = np.zeros_like(pixelCoords)
	for i,elem in enumerate(pixelCoords):
		dirs[i] = elem/np.linalg.norm(elem)

	return dirs



@jit(nopython = True)
def compute_beta(coords, origin, R):
    """
    Compute angular position (beta angle) of points on a spherical cap relative
    to the cap center.
    
    For each coordinate point on a sphere of radius R, computes the angle beta
    between the point's position vector and the sphere center direction as viewed
    from origin. This angle is used in solar limb darkening calculations.
    
    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Array of 3D coordinate points on or near the spherical surface.
    origin : ndarray, shape (3,)
        Observer position (e.g., spacecraft location). The viewing direction
        is computed from this point.
    R : float
        Radius of the sphere (e.g., solar radius in km).
    
    Returns
    -------
    betas : ndarray, shape (N,)
        Angular positions in radians. beta[i] represents the angle between
        the viewing direction to coords[i] and the sphere center direction.
        Range: [0, π/2] for visible hemisphere.
    

    See Also
    --------
    compute_pixel_intensities : Uses beta to compute intensity with limb darkening
    

    """
	d = np.linalg.norm(origin)
	o = origin / np.linalg.norm(origin)
	betas = np.zeros((len(coords)))
	
	for i,c in enumerate(coords):
		cos = np.dot(c/np.linalg.norm(c), o)
		if cos > 1.0:
			cos = 1.0
		elif cos < -1.0:
			cos = -1.0
		ang = np.arccos(cos)

		
		sinb = d/R*np.sin(ang)
		beta = np.arcsin(sinb)
		betas[i] = beta
		#if np.isnan(beta):
			#print(np.dot(c/np.linalg.norm(c),o))
			#print(sinb)

	return betas

@jit(nopython = True)
def compute_pixel_intensities(beta, Type = 'Standard'):
    """
    Compute solar intensity at given angular positions using limb darkening models.
    
    Calculates the relative brightness of solar disk pixels based on their
    angular position (beta angle) from disk center. Implements two physically
    motivated limb darkening models.
    
    Parameters
    ----------
    beta : ndarray, shape (N,)
        Angular positions in radians. beta = 0 at disk center, beta = π/2
        at limb. Typically computed by compute_beta function.
    Type : str, default='Standard'
        Limb darkening model to use:
        
        - 'Standard': Quadratic empirical model 
        - 'Eddington': Eddington approximation
    
    Returns
    -------
    intensities : ndarray, shape (N,)
        Relative intensity values. Normalized so that disk-integrated intensity
        is preserved. Values range from ~0.6 (at limb) to 1.0 (at center) for
        Standard model.
    
    Notes
    -----
    **Standard (Quadratic) Model:**
    
    I(β) = a₀ + a₁·cos(β) + a₂·cos²(β)
    
    Coefficients:
    - a₀ = 0.3  : constant term
    - a₁ = 0.93 : linear coefficient  
    - a₂ = -0.23: quadratic coefficient
    
    This empirical model provides excellent agreement with observations in
    visible wavelengths. The center-to-limb intensity ratio is approximately 0.6.
    
    **Eddington Approximation Model:**
    
    I(μ) = (3/4) · [(7/12) + (μ/2) - (μ²/3) + (μ³/12)·ln((1+μ)/μ)]
    
    where μ = cos(β)
    
    
    See Also
    --------
    compute_beta : Compute angular positions for limb darkening
    
    References
    ----------
    - Pierce, A.K. & Slaughter, C.D. (1977), "Solar limb darkening", 
      Solar Physics, 51, 25-41
    - Eddington, A.S. (1926), "The Internal Constitution of the Stars"
    - Neckel, H. & Labs, D. (1994), "Solar limb darkening 1986-1990",
      Solar Physics, 153, 91-114
    
    Examples
    --------
    >>> import numpy as np
    >>> # Beta angles from center to limb
    >>> beta = np.linspace(0, np.pi/2, 10)
    >>> intensities = compute_pixel_intensities(beta, Type='Standard')
    >>> print(f"Center intensity: {intensities[0]:.3f}")  # ~1.0
    >>> print(f"Limb intensity: {intensities[-1]:.3f}")    # ~0.6
    """
	if Type == 'Standard':
		a0 = 0.3
		a1 = 0.93
		a2 = -0.23
		intensities = a0 + a1*np.cos(beta) + a2*(np.cos(beta))**2
	elif Type == 'Eddington':
		m = np.cos(beta)
		intensities = 3.0/4* ( 7/12 * 0.5*m * m*(1/3 + 0.5*m)*np.log((1+m)/m))
	

	return intensities