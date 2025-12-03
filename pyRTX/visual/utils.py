import matplotlib.colors as col
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def color_convert(color, size = 1):
    """
    Converts a color to a list of colors of a specified size.

    Parameters
    ----------
    color : str or tuple
        The color to be converted.
    size : int, default=1
        The size of the list of colors to be returned.

    Returns
    -------
    list
        A list of RGBA color values.
    """
    color = col.to_rgba_array(color)*255.0
    return list(color)*size


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_mesh(mesh, title="3D Mesh", figsize=(10, 8), alpha=0.7, 
              edge_color='k', face_color=None, labels=None,
              elev=30, azim=45):
    """
    Plots one or more trimesh meshes using matplotlib.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh
        The mesh or meshes to plot.
    title : str, default="3D Mesh"
        The title of the plot.
    figsize : tuple, default=(10, 8)
        The size of the figure in inches.
    alpha : float, default=0.7
        The transparency of the mesh faces.
    edge_color : str or list of str, default='k'
        The color of the mesh edges.
    face_color : str, list of str, or None, optional
        The color of the mesh faces. If None, automatic colors are used.
    labels : list of str, optional
        Labels for each mesh, for use in the legend.
    elev : float, default=30
        The elevation viewing angle in degrees.
    azim : float, default=45
        The azimuthal viewing angle in degrees.
    
    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axis objects.
    
    Examples
    --------
    >>> # Single mesh with default colors
    >>> fig, ax = plot_mesh(mesh)
    
    >>> # Single mesh with custom color
    >>> fig, ax = plot_mesh(mesh, face_color='lightblue')
    
    >>> # Multiple meshes with automatic colors
    >>> fig, ax = plot_mesh([mesh1, mesh2, mesh3])
    
    >>> # Multiple meshes with custom colors
    >>> fig, ax = plot_mesh([mesh1, mesh2],
    ...                     face_color=['red', 'blue'],
    ...                     labels=['Mesh 1', 'Mesh 2'])
    
    >>> # Custom viewing angle (top-down view)
    >>> fig, ax = plot_mesh(mesh, elev=90, azim=0)
    
    >>> # Side view
    >>> fig, ax = plot_mesh(mesh, elev=0, azim=0)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert single mesh to list for uniform handling
    if not isinstance(mesh, list):
        meshes = [mesh]
    else:
        meshes = mesh
    
    n_meshes = len(meshes)
    
    # Handle face colors
    if face_color is None:
        # Use automatic colors from colormap
        colors = plt.cm.tab10.colors
        face_colors = [colors[i % len(colors)] for i in range(n_meshes)]
    elif isinstance(face_color, str):
        # Single color for all meshes
        face_colors = [face_color] * n_meshes
    elif isinstance(face_color, list):
        # List of colors provided
        if len(face_color) != n_meshes:
            raise ValueError(f"Number of face_colors ({len(face_color)}) must match number of meshes ({n_meshes})")
        face_colors = face_color
    else:
        raise ValueError("face_color must be None, a string, or a list of strings")
    
    # Handle edge colors
    if isinstance(edge_color, str):
        edge_colors = [edge_color] * n_meshes
    elif isinstance(edge_color, list):
        if len(edge_color) != n_meshes:
            raise ValueError(f"Number of edge_colors ({len(edge_color)}) must match number of meshes ({n_meshes})")
        edge_colors = edge_color
    else:
        raise ValueError("edge_color must be a string or a list of strings")
    
    # Plot each mesh
    all_vertices = []
    for i, (m, fc, ec) in enumerate(zip(meshes, face_colors, edge_colors)):
        poly3d = Poly3DCollection(m.vertices[m.faces], 
                                  alpha=alpha,
                                  facecolor=fc,
                                  edgecolor=ec,
                                  linewidths=0.1)
        
        ax.add_collection3d(poly3d)
        all_vertices.append(m.vertices)
        
        # Add legend entry if labels provided
        if labels and i < len(labels):
            ax.plot([], [], 'o', color=fc, label=labels[i], markersize=10)
    
    # Set axis limits based on all meshes
    all_vertices = np.vstack(all_vertices)
    scale = all_vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend if labels were provided
    if labels:
        ax.legend()
    
    plt.tight_layout()
    return fig, ax

def visualize_planet_field(planet, field='temperature', epoch=None, cmap='viridis', 
                           figsize=(12, 10), show_colorbar=True, title=None,
                           vmin=None, vmax=None, lighting=True, 
                           show_sun=False, elev = 30, azim = 45):
    """
    Visualize a planet mesh with color-mapped field values on its surface.
    
    Creates a 3D visualization of the planet with face colors representing
    physical properties such as temperature, albedo, or emissivity. Optionally
    displays the Sun direction vector. Useful for verifying planetary models 
    and understanding spatial distributions of surface properties.
    
    Parameters
    ----------
    planet : Planet
        Planet object containing mesh geometry and surface properties.
    field : str, default='temperature'
        Surface field to visualize. Options:
        
        - 'temperature': Surface temperature (K)
        - 'albedo': Bond albedo (0-1)
        - 'emissivity': Infrared emissivity (0-1)
        - 'area': Face areas (for geometric verification)
        
    epoch : float or None, default=None
        SPICE ephemeris time for time-dependent fields. Required if
        temperature varies with time or if show_sun=True. If None, uses 
        static values.
    cmap : str, default='viridis'
        Matplotlib colormap name. Common choices:
        
        - 'viridis': Good for temperature (dark to bright)
        - 'plasma': Alternative for temperature
        - 'coolwarm': Diverging colormap
        - 'RdYlBu_r': Red-yellow-blue (reversed)
        
    figsize : tuple, default=(12, 10)
        Figure size in inches (width, height).
    show_colorbar : bool, default=True
        If True, display colorbar showing the mapping from values to colors.
    title : str or None, default=None
        Custom plot title. If None, automatically generated based on field name.
    vmin : float or None, default=None
        Minimum value for color scale. If None, uses data minimum.
    vmax : float or None, default=None
        Maximum value for color scale. If None, uses data maximum.
    lighting : bool, default=True
        If True, apply directional lighting for better 3D perception.
        If False, use flat shading showing colors directly.
    show_sun : bool, default=False
        If True, draw an arrow showing the Sun direction. Requires epoch,
        spacecraft_name, and reference_frame to be specified.
    spacecraft_name : str or None, default=None
        Name of spacecraft for SPICE queries (e.g., 'LRO'). Required if
        show_sun=True.
    reference_frame : str or None, default=None
        Reference frame for Sun direction vector (e.g., 'LRO_SC_BUS').
        Required if show_sun=True.
    elev : float
        Elevation viewing angle in degrees (default: 30)
    azim : float
        Azimuthal viewing angle in degrees (default: 45)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The 3D axes object.
    
    Notes
    -----
    The function retrieves field values using Planet object methods:
    - Temperature: planet.getFaceTemperatures(epoch)
    - Albedo: planet.getFaceAlbedo()
    - Emissivity: planet.getFaceEmissivity()
    
    Face colors are computed by mapping field values to the colormap range.
    The mesh is displayed using matplotlib's 3D projection with face colors.
    
    When show_sun=True, a yellow arrow is drawn from the planet center pointing
    toward the Sun. The arrow length is scaled to 1.5× the planet's maximum
    dimension for visibility. The Sun position is queried from SPICE using
    the provided spacecraft_name and reference_frame.
    
    For large meshes (>10000 faces), rendering may be slow. Consider using
    mesh subdivision or decimation to reduce face count.
    
    Examples
    --------
    >>> import spiceypy as sp
    >>> from pyRTX.classes import Planet
    >>> 
    >>> # Create planet with temperature map
    >>> moon = Planet(name='Moon', radius=1737.4)
    >>> moon.dayside_temperature = 400
    >>> moon.nightside_temperature = 100
    >>> 
    >>> # Visualize temperature at specific epoch with Sun direction
    >>> epoch = sp.str2et('2024-01-01T12:00:00')
    >>> fig, ax = visualize_planet_field(moon, field='temperature', 
    ...                                  epoch=epoch, cmap='plasma',
    ...                                  show_sun=True, 
    ...                                  spacecraft_name='LRO',
    ...                                  reference_frame='MOON_PA')
    >>> 
    >>> # Visualize constant albedo
    >>> moon.albedo = 0.12
    >>> fig, ax = visualize_planet_field(moon, field='albedo', cmap='gray')
    
    See Also
    --------
    Planet.getFaceTemperatures : Get temperature values for mesh faces
    Planet.getFaceAlbedo : Get albedo values for mesh faces
    Planet.getFaceEmissivity : Get emissivity values for mesh faces
    trimesh.Trimesh.visual : Trimesh visualization utilities
    """
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    import spiceypy as sp
    
    # Validate Sun direction parameters
    if show_sun:
        spacecraft_name = planet.name
        reference_frame = planet.bodyFrame
        if epoch is None:
            raise ValueError("epoch is required when show_sun=True")

    
    # Get the mesh
    if epoch is not None:
        mesh = planet.mesh(epoch=epoch)
    else:
        mesh = planet.mesh()
    
    # Get field values based on requested field
    if field == 'temperature':
        if epoch is None:
            raise ValueError("epoch is required for temperature field")
        values = planet.getFaceTemperatures(epoch)
        if title is None:
            title = f'Surface Temperature Distribution'
        default_cmap = 'plasma'
        cbar_label = 'Temperature (K)'
        
    elif field == 'albedo':
        values = planet.getFaceAlbedo(epoch)
        if title is None:
            title = 'Surface Albedo Distribution'
        default_cmap = 'gray'
        cbar_label = 'Albedo'
        
    elif field == 'emissivity':
        values = planet.getFaceEmissivity()
        if title is None:
            title = 'Surface Emissivity Distribution'
        default_cmap = 'viridis'
        cbar_label = 'Emissivity'
        
    elif field == 'area':
        # Use mesh face areas
        values = mesh.area_faces
        if title is None:
            title = 'Face Area Distribution'
        default_cmap = 'viridis'
        cbar_label = 'Area (km²)'
        
    else:
        raise ValueError(f"Unknown field: {field}. Use 'temperature', 'albedo', 'emissivity', or 'area'")
    
    # Use default colormap if not explicitly overridden for temperature
    if field == 'temperature' and cmap == 'viridis':
        cmap = default_cmap
    
    # Set up normalization
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
   
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    
    # Map values to colors
    face_colors = colormap(norm(values))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Create face polygons
    face_vertices = vertices[faces]
    
    # Create 3D polygon collection
    poly_collection = Poly3DCollection(face_vertices, 
                                       facecolors=face_colors,
                                       edgecolors='none' if not lighting else face_colors,
                                       linewidths=0.1,
                                       alpha=1.0)
    
    # Add lighting if requested
    if lighting:
        poly_collection.set_edgecolor('k')
        poly_collection.set_linewidth(0.05)
    
    ax.add_collection3d(poly_collection)
    
    # Set axis limits
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                         vertices[:, 1].max() - vertices[:, 1].min(),
                         vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Draw Sun direction if requested
    if show_sun:
        # Get Sun position relative to planet in specified reference frame
        sun_pos = sp.spkezr('Sun', epoch, reference_frame, 'LT+S', spacecraft_name)[0][0:3]
        
        # Normalize and scale for visualization
        sun_dir = sun_pos / np.linalg.norm(sun_pos)
        arrow_length = max_range * 1.5  # Scale arrow to be visible
        sun_arrow = sun_dir * arrow_length
        
        # Draw arrow from planet center to Sun direction
        ax.quiver(mid_x, mid_y, mid_z,  # Arrow origin (planet center)
                 sun_arrow[0], sun_arrow[1], sun_arrow[2],  # Arrow direction
                 color='yellow', 
                 arrow_length_ratio=0.15,
                 linewidth=3,
                 label='Sun Direction')
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10)
    
    # Labels
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    if show_colorbar:
        mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(values)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label(cbar_label, fontsize=12)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    
    return fig, ax
