import numpy as np
import trimesh.transformations as tmt
import spiceypy as sp
import matplotlib.pyplot as plt
import logging 
import pickle as pkl
from tqdm import tqdm

from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.RayTracer import RayTracer
from pyRTX.core.analysis_utils import LookupTableND as LT
from pyRTX.core.analysis_utils import get_spacecraft_area

# Purpose:
# Generate the cross-section lookup table used for the atmopsheric drag computations
# In this example we will use the MAVEN shape

# Structure:
# 1) Define the spacecraft shape
# 2) Run the ray-tracing and generate the lookup table of cross sections
# 3) Visualize the lookup table
# 4) Interrogate the lookup table in points outside the sampling grid
# 5) Save the lookup table for later use as a pickled instance of the LookupTableND class



# Load the metakernel containing references to the necessary SPICE frames
METAKR = '../example_data/LRO/metakernel_lro.tm'
sp.furnsh(METAKR)


# Define a basic epoch and a time span
epc = "2010 may 10 09:25:00"
epc_et0 =  sp.str2et( epc )


# Define spacecraft properties 
mass = 2000




# Define the spacecraft


obj_path = '../example_data/LRO/'
lro = Spacecraft( name = 'LRO',
                                        base_frame = 'LRO_SC_BUS',                                      # Name of the spacecraft body-fixed frame
                                        spacecraft_model = {                                            # Define a spacecraft model
                                        'LRO_BUS': { 
                                                         'file' : obj_path + 'bus_rotated.obj',         # .obj file of the spacecraft component
                                                         'frame_type': 'Spice',                         # type of frame (can be 'Spice' or 'UD'
                                                         'frame_name': 'LRO_SC_BUS',                    # Name of the frame
                                                         'center': [0.0,0.0,0.0],                       # Origin of the component
                                                         'diffuse': 0.1,                                # Diffuse reflect. coefficient
                                                         'specular': 0.3,                               # Specular reflect. coefficient
                                                         },

                                        'LRO_SA': {     
                                                        'file': obj_path + 'SA_recentred.obj',
                                                        'frame_type': 'Spice',
                                                        'frame_name': 'LRO_SA',
                                                        'center': [-1,-1.1, -0.1],
                                                         'diffuse': 0,
                                                         'specular': 0.3,
                                                        },


                                        'LRO_HGA': {    
                                                        'file': obj_path + 'HGA_recentred.obj',
                                                        'frame_type': 'Spice',
                                                        'frame_name': 'LRO_HGA',
                                                        'center':[-0.99,    -0.3,  -3.1],
                                                         'diffuse': 0.2,
                                                         'specular': 0.1,
                                        },


                                        }
                                        
                                        )


# Define the parameters for the cross section sampling

sampling = 10           # Degrees of sampling in RA/DEC
conv = np.pi/180        # Conversion factor
RA = np.linspace(0, 360, int(360/sampling + 1))*conv
DEC = np.linspace(-90, 90, int(360/sampling + 1))*conv


# Initialize an empty lookup table
LUT = np.zeros((len(RA), len(DEC)))


# Deactivate nasty warnings from trimesh
log = logging.getLogger('trimesh')
log.disabled = True


# Run the raytracing
for i, ra in tqdm(enumerate(RA), total = len(RA)):
        for j, dec in enumerate(DEC):
                LUT[i,j] = get_spacecraft_area(lro, ra = ra, dec = dec, epoch = epc_et0) * 1e6  # Cross section in m**2 

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
newLut = LT(axes = (RA, DEC), values = LUT.T)


# Interrogate the lookup table in arbitary points
ra = 12.347 * conv
dec = -28.31 *conv
value = newLut[ra, dec]
print(f'The cross-section in the direction {ra/conv} deg, {dec/conv} deg \n is: {value}')



# Save for later use
save = True
if save:
        import pickle as pkl
        pkl.dump(newLut, open('lro_crossection.pkl', 'wb')) # Save the class instance


plt.show()

