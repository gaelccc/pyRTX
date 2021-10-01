import trimesh
import numpy as np
from pyRTX.utils_rt import pixel_plane_opt, RTXkernel
from matplotlib.colors import to_rgba_array
# Scope: Perform a simple ray tracing against a sphere and visualize the impinging and reflected rays


### Object mesh definition. 
# Here we create a sphere using the trimesh library 
radius = 1.0
mesh = trimesh.creation.icosphere(radius = radius, subdivisions = 4)
faces = mesh.faces
faceNum = np.shape(faces)[0]

### Definition of the parameters of the pixel plane
d0 = 5   	# Distance of the pixel plane from the origin (in m)
dec = 45*np.pi/180		# Declination of the pixel plane wrt the body fixed frame (radians)
ra = 0*np.pi/180	# Right ascension of the pixel plane wrt the body fixed frame
rs = 0.1 	# Ray spacing [distance (in m) between rays
height = 4  	# Height and width of the pixel plane (in m)
width = 4


### Generation of the pixel plane
ray_origins, ray_directions = pixel_plane_opt(d0, ra, dec, width = width, height = height, ray_spacing = rs)

### Ray tracing
RTXresult  = RTXkernel(
		 mesh, 			# Mesh object
		 ray_origins, 		# Ray origins (output of pixel_plane_opt)
		 ray_directions, 	# Ray directions (output of pixel_plane_opt)
		 kernel = 'Embree', 	# The kernel to use (see the documentation)
		 bounces = 1		# Number of bounces to consider
		 )


# The results are contained in a 'container'. This means that index_tri[0] contains the indexes of the mesh elements impacted at first bounce, index_tri[1] at second bounce (if present) and so on
index_tri = RTXresult[0]
index_ray = RTXresult[1]
locations = RTXresult[2]
ray_origins = RTXresult[3]
ray_directions = RTXresult[4]


### Visualization
# Set visualization options
inc_l = 1.0  # Incoming rays length
mesh_color = 'white' #  The color of the base mesh

visualize_rays = False # Visualize the impinging rays
ray_color = 'gold'   # The color of the incoming rays

visualize_origin_points = True # Choose to set a color for the origin of the impinging rays
origin_color = 'black'

visualize_reflected_rays = False # Choose to visualize the reflected rays

visualize_impact = False # Choose to visualize the mesh elements impacted by the rays
impact_color = 'red' # The color of the mesh elements impacted by the rays


visualize_body_frame = False  # Choose if you want to visualize body frame
body_frame_colors = ['red', 'green', 'blue'] # The colors of the x,y,z axes


mesh_color = to_rgba_array(mesh_color)*255.0 # Convert the color to rgba array
impact_color = to_rgba_array(impact_color)*255.0
ray_color = to_rgba_array(ray_color)*255.0
###



### Scene generation
scene_elements = []

# Generate the impinging rays
if visualize_rays:
	ray_visualize = trimesh.load_path(np.hstack(( ray_origins[0], ray_origins[0] + ray_directions[0]*inc_l)).reshape(-1, 2, 3))
	ray_visualize.colors = np.full((len(ray_origins[0]),4), ray_color) 
	scene_elements.append(ray_visualize)


# Generate the reference frame (if requested)
if visualize_body_frame:
	xaxis = np.array([1,0,0])
	yaxis = np.array([0,1,0])
	zaxis = np.array([0,0,1])
	origin = np.array([0,0,0])

	xaxis = trimesh.load_path(np.hstack(( origin, origin + xaxis*10)).reshape(-1, 2, 3))
	yaxis = trimesh.load_path(np.hstack(( origin, origin + yaxis*10)).reshape(-1, 2, 3))
	zaxis = trimesh.load_path(np.hstack(( origin, origin + zaxis*10)).reshape(-1, 2, 3))

	xaxis.colors = np.full((1,4),to_rgba_array(body_frame_colors[0])*255)
	yaxis.colors = np.full((1,4),to_rgba_array(body_frame_colors[1])*255)
	zaxis.colors = np.full((1,4),to_rgba_array(body_frame_colors[2])*255)

	scene_elements.append(xaxis)
	scene_elements.append(yaxis)
	scene_elements.append(zaxis)


# Set the color of the mesh and impinging rays
mesh.visual.face_colors = mesh_color[0] 

if visualize_origin_points:
	org_visualize = trimesh.load_path(np.hstack(( ray_origins[0], ray_origins[0] + ray_directions[0]*0.1)).reshape(-1, 2, 3))
	org_visualize.colors = np.full((len(ray_origins[0]),4), to_rgba_array(origin_color)*255)
	scene_elements.append(org_visualize)

if visualize_impact:
	mesh.visual.face_colors[index_tri[0]] = impact_color 

scene_elements.append(mesh)



mesh.unmerge_vertices()
scene = trimesh.Scene(scene_elements)
scene.show()
