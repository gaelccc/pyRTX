"""
pyRTX - Non-gravitational forces modelling using raytracing
"""

# Import version
__version__ = "0.1.0"

# Try to import embree
try:
    import embree
    EMBREE_AVAILABLE = True
except ImportError as e:
    EMBREE_AVAILABLE = False
    import warnings
    warnings.warn(
        f"Embree is not available: {e}\n"
        "Some features may be disabled.\n"
        "To install Embree support, run: python install_deps.py",
        ImportWarning
    )

# Export key components
from pyRTX.defaults import dFloat, dInt

__all__ = [
    '__version__',
    'EMBREE_AVAILABLE',
    'dFloat',
    'dInt',
]