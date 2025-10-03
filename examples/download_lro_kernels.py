import os
import subprocess

import sys
sys.path.append('../')
from pyRTX import helpers

# Scope: 
# Download the spice kernels needed for running LRO examples
# NOTE: At the moment this script works if either wget or curl are installed in the system


agent = helpers.get_download_agent()

# From PDS
todownload = [  'spk/lrorg_2010091_2010182_v01.bsp',
                'spk/de421.bsp',
                'fk/moon_assoc_pa.tf',
                'fk/moon_080317.tf',
                'ck/lrohg_2010121_2010131_v01.bc',
                'ck/lrosa_2010121_2010131_v01.bc',
                'ck/lrosc_2010121_2010131_v01.bc',
                'pck/moon_pa_de421_1900_2050.bpc',
                'sclk/lro_clkcor_2021076_v00.tsc',
                'lsk/naif0012.tls',
                ]


for f in todownload:
        fname = os.path.basename(f)
        if not os.path.exists(f'../example_data/kernels_lro/{fname}'):
                if agent == 'wget':
                        subprocess.run(f'wget -P ../example_data/kernels_lro/  https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f}', shell = True)
                elif agent == 'curl':
                        subprocess.run(f'curl https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f} -o ../example_data/kernels_lro/{fname}', shell = True)


# From NAIF
todownload = ['fk/lro_frames_2014049_v01.tf']


for f in todownload:
        fname = os.path.basename(f)
        if not os.path.exists(f'../example_data/kernels_lro/{fname}'):
                if agent == 'wget':
                        subprocess.run(f'wget -P ../example_data/kernels_lro/  https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/{f}', shell = True)
                elif agent == 'curl':
                        subprocess.run(f'curl https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/{f} -o ../example_data/kernels_lro/{fname}', shell = True)