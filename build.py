import subprocess
import sys
from pathlib import Path
from setuptools import build_meta as _build_meta

def run_install_deps():
    """Install external C++ dependencies via install_deps.py"""
    print("\n" + "="*60)
    print("Installing dependencies via install_deps.py...")
    print(f"Platform: {sys.platform}")
    print("="*60 + "\n")
    script_path = Path(__file__).parent / 'install_deps.py'

    if not script_path.exists():
        print("\n" + "="*60)
        print(f"Warning: install_deps.py not found at {script_path}")
        print("Skipping dependency installation.")
        print("="*60 + "\n")
        return False

    try:
        subprocess.check_call([sys.executable, str(script_path)])
        print("\n" + "="*60)
        print("Dependency installation completed successfully")
        print("="*60 + "\n")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "="*60)
        print(f"Warning: Dependency installation failed: {e}")
        print("You may need to install dependencies manually.")
        print("Run: python install_deps.py")
        print("="*60 + "\n")
        return False
    except Exception as e:
        print("\n" + "="*60)
        print(f"Error running install_deps.py: {e}")
        print("You may need to install dependencies manually.")
        print("="*60 + "\n")
        return False

# Run the dependency installation
run_install_deps()

# Delegate to the standard build backend
get_requires_for_build_wheel = _build_meta.get_requires_for_build_wheel
get_requires_for_build_sdist = _build_meta.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _build_meta.prepare_metadata_for_build_wheel
build_wheel = _build_meta.build_wheel
build_sdist = _build_meta.build_sdist
