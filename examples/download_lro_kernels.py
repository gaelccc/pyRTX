import os

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
        fname = f.split('/')[1]
        if not os.path.exists(f'../example_data/kernels_lro/{fname}'):
                os.system(f'wget -P ../example_data/kernels_lro/  https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/{f}')



# From NAIF
todownload = ['fk/lro_frames_2014049_v01.tf']


for f in todownload:
        fname = f.split('/')[1]
        if not os.path.exists(f'../example_data/kernels_lro/{fname}'):
                os.system(f'wget -P ../example_data/kernels_lro/  https://naif.jpl.nasa.gov/pub/naif/LRO/kernels/{f}')
