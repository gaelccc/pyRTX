# Core routines test module
#
# Instructions: Every time a new core routine is written, or a functionality is added
# to an already existing core routine, write a test!!

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

def test_embree2():
	# Test here that the three kernels work nominally
	# Embree 2 test
	load = importlib.find_spec('pyembree')
	found = load is not None
	assert found

def test_embree3():
	# Test here that the three kernels work nominally
	# Embree 3 test
	load = importlib.find_spec('embree')
	found = load is not None
	assert found


def test_CGAL():
	# Test here that the three kernels work nominally
	# CGAL test
	load = importlib.find_spec('aabb')
	found = load is not None
	assert found
# ADD MORE
