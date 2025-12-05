# Core routines test module
#

from context import core
import numpy as np
import pytest
import importlib.util as importlib 

# Settings
tols = {'rel':1e-6, 'abs':1e-6} # pytest.approx tolerances


def test_chunker():
	arr = np.ones(1000)
	chunk = core.chunker(arr, 10)
	assert len(chunk) == 10

def test_block_normalize():
	arr = np.ones((100,3))
	norms = core.block_normalize(arr)
	
	compare = np.ones((100,3))*0.57735027

	assert norms == pytest.approx(compare)

def test_embree3():
	# Test here that the three kernels work nominally
	# Embree 3 test
	load = importlib.find_spec('embree')
	found = load is not None
	assert found


# CGAL is not implemented
# This is a placeholder for later
#def test_CGAL():
#	# Test here that the three kernels work nominally
#	# CGAL test
#	load = importlib.find_spec('aabbtree')
#	found = load is not None
#	assert found
# ADD MORE
