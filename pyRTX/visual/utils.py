import matplotlib.colors as col
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def color_convert(color, size = 1):
    """Converts a color to a list of colors of size `size`
    Parameters
    ----------
    color : str or tuple
        Color to be converted
    size : int
        Size of the list of colors to be returned
    Returns
    -------
    color : (size,) array-like
        converted color as a list of rgba lists
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
    Plot one or more trimesh meshes using matplotlib.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh or list of trimesh.Trimesh
        The mesh(es) to plot
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    alpha : float
        Transparency of mesh faces (0-1)
    edge_color : str or list of str
        Color of mesh edges. If single string, applied to all meshes.
        If list, must match number of meshes.
    face_color : str, list of str, or None
        Color of mesh faces. If None, uses automatic colors from colormap.
        If single string, applied to all meshes.
        If list, must match number of meshes.
    labels : list of str, optional
        Labels for each mesh (for legend)
    elev : float
        Elevation viewing angle in degrees (default: 30)
    azim : float
        Azimuthal viewing angle in degrees (default: 45)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    
    Examples:
    ---------
    # Single mesh with default colors
    fig, ax = plot_mesh(mesh)
    
    # Single mesh with custom color
    fig, ax = plot_mesh(mesh, face_color='lightblue')
    
    # Multiple meshes with automatic colors
    fig, ax = plot_mesh([mesh1, mesh2, mesh3])
    
    # Multiple meshes with custom colors
    fig, ax = plot_mesh([mesh1, mesh2], 
                        face_color=['red', 'blue'],
                        labels=['Mesh 1', 'Mesh 2'])
    
    # Custom viewing angle (top-down view)
    fig, ax = plot_mesh(mesh, elev=90, azim=0)
    
    # Side view
    fig, ax = plot_mesh(mesh, elev=0, azim=0)
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