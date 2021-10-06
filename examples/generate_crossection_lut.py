import numpy as np
from pyRTX.pixelPlaneClass import pixelPlane
import spiceypy as sp
import matplotlib.pyplot as plt
from pyRTX.srpClass import solarPressure
from pyRTX.scClass import Spacecraft
import trimesh.transformations as tmt
import logging
from pyRTX.rayTracerClass import rayTracer

# Purpose:
# Generate the cross-section lookup table used for the atmopsheric drag computations
# In this example we will use the MAVEN shape

# Structure:
# 1) Define the spacecraft shape
# 2) Run the ray-tracing and generate the lookup table of cross sections
# 3) Visualize the lookup table
# 4) Interrogate the lookup table in points outside the sampling grid
# 5) Save the lookup table for later use as a pickled instance of the LookupTableND class and as a numpy array


METAKR = '../example_data/generic_metakernel.tm'
sp.furnsh(METAKR)

# Define the spacecraft
identity = tmt.identity_matrix()
maven = Spacecraft(     name = 'MAVEN',
                        base_frame = 'SC_FRAME',
                        units = 'm',
                        spacecraft_model = {
                                'BUS': {
                                                'file': '../example_data/maven_mat.obj',
                                                'frame_type': 'UD',
                                                'UD_rotation': identity,
                                                'center': [0,0,0],
                                                'diffuse': 0,
                                                'specular': 0,
                                                }
                                        }
                       )


# Define the parameters for the cross section sampling

sampling = 50           # Degrees of sampling in RA/DEC
conv = np.pi/180        # Conversion factor
RA = np.linspace(0, 360, int(360/sampling + 1))*conv
DEC = np.linspace(-90, 90, int(360/sampling + 1))*conv


# Initialize an empty lookup table
LUT = np.zeros((len(RA), len(DEC)))


# Deactivate nasty warnings from trimesh
log = logging.getLogger('trimesh')
log.disabled = True

# Instantiate the pixel plane
rays = pixelPlane(
        spacecraft = maven,
        mode = 'Fixed',
        width = 15,
        height = 15,
        ray_spacing = 0.1,
        lon = 0,
        lat = 0,
        distance = 30,
        )

# Create a fictitious epoch (not needed in this case, as the spacecraft shape does not depend on spice kernels
epc = "2000 jan 10 00:00:00"
epc_et0 = sp.str2et(epc)

# Run the raytracing
for i, ra in enumerate(RA):
        print(f'{i}/{len(RA)}')
        for j, dec in enumerate(DEC):
                
                rays.update_latlon(lon = ra, lat = dec)

                rtx = rayTracer( maven, rays, kernel = 'Embree', bounces = 1, diffusion = False)
                rtx.trace(epc_et0)


                hit_rays = rtx.index_ray_container

                LUT[i,j] = len(hit_rays[0])/rays.norm_factor * 1e6  # Cross section in m**2 

# Re-activate the warnings from trimesh and unload the spice kernels
log.disabled = False
sp.unload(METAKR)


# Plot the resulting Lookup Table
fig, ax = plt.subplots()
X,Y = np.meshgrid(RA, DEC)
h = ax.contourf(X/conv,Y/conv, LUT.T)
ax.set_xlabel('Right Ascension [deg]')
ax.set_ylabel('Declination [deg]')
plt.colorbar(h, label = r'Cross Section [$m^2$]', ax = ax)



# Use the pyRTX lookup table class to interpolate and resample
from pyRTX.analysis_utils import LookupTableND as LT

newLut = LT(axes = (RA, DEC), values = LUT.T)


# Interrogate the lookup table in arbitary points
ra = 12.347 * conv
dec = -28.31 *conv
value = newLut[ra, dec]
print(f'The cross-section in the direction {ra/conv} deg, {dec/conv} deg \n is: {value}')



# Save for later use
save = False
if save:
        import pickle as pkl
        np.save('maven_crossection_lut.npy', newLUT[:,:]) # Save as a numpy array
        pkl.dump(newLUT, open('maven_pickled_lut.pkl', 'wb')) # Save the class instance


plt.show()

