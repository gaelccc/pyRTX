"""
pyRTX - Non-gravitational forces modelling using raytracing
"""
import sys

# Import version
__version__ = "0.1.0"

# Automatically configure Embree environment (Linux only)
if sys.platform == 'linux':
    from pyRTX.embree_env import _embree_configured
else:
    _embree_configured = False

# Now try to import embree
try:
    import embree
    EMBREE_AVAILABLE = True
except ImportError as e:
    EMBREE_AVAILABLE = False
    import warnings
    
    if sys.platform == 'linux':
        if _embree_configured:
            warnings.warn(
                f"Embree environment was configured but import failed: {e}\n"
                "You may need to reinstall pyRTX or check your Embree installation.\n"
                "Run: python install_deps.py",
                ImportWarning
            )
        else:
            warnings.warn(
                "Embree is not available. Embree environment could not be configured.\n"
                "Some features may be disabled.\n"
                "Run: python install_deps.py",
                ImportWarning
            )
    else:
        warnings.warn(
            f"Embree auto-configuration is only supported on Linux.\n"
            f"Current platform: {sys.platform}\n"
            "Please manually configure Embree environment or run install_deps.py",
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