import pytest
import numpy as np
import trimesh
from pyRTX.visual import utils

from pyRTX.classes.Planet import Planet
import spiceypy as sp

def test_plot_mesh():
    # Create a sample mesh
    mesh = trimesh.creation.box()

    # Test that the function runs without errors
    try:
        utils.plot_mesh(mesh)
    except Exception as e:
        pytest.fail(f"plot_mesh raised an exception: {e}")


def test_visualize_planet_field():
    # Create a mock Planet object
    class MockPlanet(Planet):
        def __init__(self, name, radius):
            super().__init__(name=name, radius=radius)
            self._albedo = 0.5 # Set a default albedo

        def getFaceAlbedo(self, epoch):
            return np.full(self.numFaces, self._albedo)

    planet = MockPlanet(name='Earth', radius=6371)
    sp.furnsh('example_data/generic_kernels/naif0012.tls')
    epoch = sp.str2et('2024-01-01T12:00:00')

    # Test that the function runs without errors
    try:
        utils.visualize_planet_field(planet, field='albedo', epoch=epoch)
    except Exception as e:
        pytest.fail(f"visualize_planet_field raised an exception: {e}")
