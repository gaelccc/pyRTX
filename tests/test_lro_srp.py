import unittest
import numpy as np
import spiceypy as sp
import xarray as xr
import os

from pyRTX.classes.Spacecraft import Spacecraft
from pyRTX.classes.Planet import Planet
from pyRTX.classes.PixelPlane import PixelPlane
from pyRTX.classes.RayTracer import RayTracer
from pyRTX.classes.SRP import SunShadow, SolarPressure
from pyRTX.classes.Precompute import Precompute
from pyRTX.core.analysis_utils import epochRange2


class TestLROSRP(unittest.TestCase):

    def test_lro_srp_computation(self):
        # Change to the examples directory to handle relative paths
        original_cwd = os.getcwd()
        os.chdir('examples')

        # Setup from lro_srp.py
        ref_epc = "2010 may 10 09:25:00"
        duration = 10000
        timestep = 100
        spacing = 0.01
        METAKR = '../example_data/LRO/metakernel_lro.tm'
        obj_path = '../example_data/LRO/'
        base_flux = 1361.5
        ref_radius = 1737.4
        n_cores = 1
        sc_mass = xr.open_dataset('mass/lro_mass.nc')
        sc_mass.load()
        sc_mass.close()


        # Load the metakernel
        sp.furnsh(METAKR)

        # Define epochs
        epc_et0 = sp.str2et(ref_epc)
        epc_et1 = epc_et0 + duration
        epochs = epochRange2(startEpoch=epc_et0, endEpoch=epc_et1, step=timestep)

        # Define Spacecraft
        lro = Spacecraft(
            name='LRO',
            mass=sc_mass,
            base_frame='LRO_SC_BUS',
            spacecraft_model={
                'LRO_BUS': {
                    'file': obj_path + 'bus_rotated.obj',
                    'frame_type': 'Spice',
                    'frame_name': 'LRO_SC_BUS',
                    'center': [0.0, 0.0, 0.0],
                    'diffuse': 0.1,
                    'specular': 0.3,
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

        # Define Moon
        moon = Planet(
            radius=ref_radius,
            name='Moon',
            bodyFrame='MOON_PA',
            sunFixedFrame='GSE_MOON',
            units='km',
            subdivs=5,
        )

        # Define Rays
        rays = PixelPlane(
            spacecraft=lro,
            mode='Dynamic',
            distance=100,
            source='Sun',
            width=10,
            height=10,
            ray_spacing=spacing,
        )

        # Define RayTracer
        rtx = RayTracer(
            lro,
            rays,
            kernel='Embree3',
            bounces=2,
            diffusion=False,
        )

        # Precomputation
        prec = Precompute(epochs=epochs,)
        prec.precomputeSolarPressure(lro, moon, correction='LT+S')
        prec.dump()

        # Shadow
        shadow = SunShadow(
            spacecraft=lro,
            body='Moon',
            bodyShape=moon,
            limbDarkening='Eddington',
            precomputation=prec,
        )

        # Solar Pressure
        srp = SolarPressure(
            lro,
            rtx,
            baseflux=base_flux,
            shadowObj=shadow,
            precomputation=prec,
        )

        # Computation
        accel = srp.compute(epochs, n_cores=n_cores) * 1e3

        # Unload kernels
        sp.unload(METAKR)

        # Change back to the original directory
        os.chdir(original_cwd)


        expected_accel = np.array(
        [[-1.60388913e-08, 3.06416082e-08, 2.71976845e-08],
         [-1.84534922e-08, 3.17936686e-08, 2.45130092e-08],
         [-2.05350283e-08, 3.18042046e-08, 2.23657423e-08],
         [-2.24971239e-08, 3.17952659e-08, 2.00704835e-08],
         [-2.43309611e-08, 3.18342078e-08, 1.76647097e-08],
         [-2.59860461e-08, 3.18643472e-08, 1.51377045e-08],
         [-2.74494461e-08, 3.19561799e-08, 1.25425774e-08],
         [-2.86597571e-08, 3.20126783e-08, 9.85516593e-09],
         [-2.95540210e-08, 3.20461535e-08, 7.08920394e-09],
         [-3.01520903e-08, 3.20324941e-08, 4.30631080e-09],
         [-3.03450709e-08, 3.19363365e-08, 1.72983771e-09],
         [-3.02742319e-08, 3.18324198e-08, -3.31590315e-10],
         [-3.00107291e-08, 3.17066975e-08, -2.10881022e-09],
         [-2.90258674e-08, 3.10936017e-08, -3.53827615e-09],
         [-2.74841708e-08, 3.00473731e-08, -5.03300638e-09],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [2.93719391e-09, 3.08032339e-09, -4.79715708e-10],
         [2.66323903e-08, 2.74313656e-08, -2.99709502e-09],
         [2.78945515e-08, 2.84314232e-08, -1.89611443e-09],
         [2.84612193e-08, 2.88541354e-08, -4.67521347e-10],
         [2.85739109e-08, 2.90046886e-08, 1.36362200e-09],
         [2.83520898e-08, 2.90354192e-08, 3.59193996e-09],
         [2.79287951e-08, 2.90680286e-08, 6.12754692e-09],
         [2.71397044e-08, 2.90385257e-08, 8.64521298e-09],
         [2.61392628e-08, 2.90251039e-08, 1.11533590e-08],
         [2.48974923e-08, 2.90458594e-08, 1.36352735e-08],
         [2.34871574e-08, 2.91316129e-08, 1.60754492e-08],
         [2.19407971e-08, 2.92394211e-08, 1.84815166e-08],
         [2.02142035e-08, 2.94200289e-08, 2.08351255e-08],
         [1.83213795e-08, 2.96336532e-08, 2.31086265e-08],
         [1.62433412e-08, 2.98696024e-08, 2.51987068e-08],
         [1.40167786e-08, 3.00985675e-08, 2.70474139e-08],
         [1.16524336e-08, 3.03098976e-08, 2.86329831e-08],
         [9.20325895e-09, 3.05862111e-08, 2.99695831e-08],
         [6.67664423e-09, 3.08379209e-08, 3.09685075e-08],
         [4.09577491e-09, 3.10960940e-08, 3.17059190e-08],
         [1.48360434e-09, 3.13490109e-08, 3.21972896e-08],
         [-1.14038523e-09, 3.15988772e-08, 3.23437076e-08],
         [-3.79456211e-09, 3.17640796e-08, 3.21783836e-08],
         [-6.43500863e-09, 3.18328485e-08, 3.16146825e-08],
         [-9.08051585e-09, 3.29197672e-08, 2.97574120e-08],
         [-1.17555455e-08, 3.68928052e-08, 2.41269542e-08],
         [-1.42946020e-08, 3.68359648e-08, 2.25750254e-08],
         [-1.66901463e-08, 3.52730171e-08, 2.29862566e-08],
         [-1.86741737e-08, 3.18321170e-08, 2.41935528e-08],
         [-2.07605361e-08, 3.18628250e-08, 2.20257537e-08],
         [-2.27013583e-08, 3.18857378e-08, 1.97299324e-08],
         [-2.45072387e-08, 3.19060896e-08, 1.73099730e-08],
         [-2.61443691e-08, 3.19497207e-08, 1.47823977e-08],
         [-2.75823205e-08, 3.20418828e-08, 1.21813679e-08],
         [-2.87643618e-08, 3.20964379e-08, 9.48599292e-09],
         [-2.96069556e-08, 3.21256535e-08, 6.71913701e-09],
         [-3.01396229e-08, 3.21007083e-08, 3.93711920e-09],
         [-3.02912684e-08, 3.19903561e-08, 1.44648245e-09],
         [-3.02251290e-08, 3.19058722e-08, -5.94059924e-10],
         [-2.98787046e-08, 3.17423451e-08, -2.29776147e-09],
         [-2.88168472e-08, 3.10752296e-08, -3.73404636e-09],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.],
         [0., 0., 0.]])

        # Assertion
        np.testing.assert_allclose(accel, expected_accel, rtol=1e-5, atol=1e-8)

if __name__ == '__main__':
    unittest.main()
