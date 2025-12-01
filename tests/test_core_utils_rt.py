import numpy as np
import pytest
from unittest.mock import patch
import pyRTX.core.utils_rt as utils
import trimesh

def test_pxform_convert():
    pxform = np.eye(3)
    expected = np.eye(4)
    result = utils.pxform_convert(pxform)
    assert np.allclose(result, expected)

def test_block_normalize():
    V = np.array([[1, 1, 1], [2, 2, 2]])
    expected = np.array([[0.57735027, 0.57735027, 0.57735027], [0.57735027, 0.57735027, 0.57735027]])
    result = utils.block_normalize(V)
    assert np.allclose(result, expected)

def test_block_dot():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.array([14, 77])
    result = utils.block_dot(a, b)
    assert np.allclose(result, expected)

def test_reflected():
    incoming = np.array([[1, -1, 0]])
    normal = np.array([[0, 1, 0]])
    expected = np.array([[1, 1, 0]])
    result = utils.reflected(incoming, normal)
    assert np.allclose(result, expected)

def test_get_orthogonal():
    v = np.array([1, 0, 0], dtype=np.float64)
    result = utils.get_orthogonal(v)
    assert np.isclose(np.dot(v, result), 0)

def test_get_centroids():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    expected = np.array([[1/3, 1/3, 0]])
    result = utils.get_centroids(V, F)
    assert np.allclose(result, expected)

def test_get_cross_products():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    expected = np.array([[0, 0, 1]])
    result = utils.get_cross_products(V, F)
    assert np.allclose(result, expected)

def test_get_face_areas():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    expected = np.array([0.5])
    result = utils.get_face_areas(V, F)
    assert np.allclose(result, expected)

def test_get_surface_normals():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    expected = np.array([[0, 0, 1]])
    result = utils.get_surface_normals(V, F)
    assert np.allclose(result, expected)

def test_get_surface_normals_and_face_areas():
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    F = np.array([[0, 1, 2]])
    expected_N = np.array([[0, 0, 1]])
    expected_A = np.array([0.5])
    N, A = utils.get_surface_normals_and_face_areas(V, F)
    assert np.allclose(N, expected_N)
    assert np.allclose(A, expected_A)

@patch('pyRTX.core.utils_rt.cgal_init_geometry')
def test_RTXkernel_cgal(mock_cgal_init_geometry):
    # Create a mock mesh
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           faces=[[0, 1, 2]])

    # Create mock ray origins and directions
    ray_origins = np.array([[0.5, 0.5, 1]])
    ray_directions = np.array([[0, 0, -1]])

    # Configure the mock intersector
    mock_cgal_init_geometry.return_value.intersect1_2d_with_coords.return_value = (
        np.array([0]),  # index_tri
        np.array([[0.5, 0.5, 0]])  # location
    )

    # Call the function
    index_tri, index_ray, locations, _, _, _ = utils.RTXkernel(
        mesh, ray_origins, ray_directions, kernel='CGAL'
    )

    # Assert the results
    assert len(index_tri) == 1
    assert len(index_ray) == 1
    assert len(locations) == 1
