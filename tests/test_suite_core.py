# Core routines test module
#
# Instructions: Every time a new core routine is written, or a functionality is added
# to an already existing core routine, write a test!!

from context import core
import numpy as np
import pytest

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



# ADD MORE
