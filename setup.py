# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
import sys
from pathlib import Path

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # Run install_deps.py BEFORE setup installation
        self._install_external_deps()
        install.run(self)

    def _install_external_deps(self):
        """Install external C++ dependencies"""
        print("\n" + "="*60)
        print("Installing dependencies via install_deps.py...")
        print("="*60 + "\n")
        script_path = Path(__file__).parent / 'install_deps.py'
        try:
            subprocess.check_call([sys.executable, str(script_path)])
        except subprocess.CalledProcessError as e:
            print(f"\nWarning: Dependency installation failed: {e}")
            print("You may need to install dependencies manually.")
            print("Run: python install_deps.py\n")

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        # Run install_deps.py BEFORE setup installation
        self._install_external_deps()
        develop.run(self)

    def _install_external_deps(self):
        """Install external C++ dependencies"""
        print("\n" + "="*60)
        print("Installing dependencies via install_deps.py...")
        print("="*60 + "\n")
        script_path = Path(__file__).parent / 'install_deps.py'
        try:
            subprocess.check_call([sys.executable, str(script_path)])
        except subprocess.CalledProcessError as e:
            print(f"\nWarning: Dependency installation failed: {e}")
            print("You may need to install dependencies manually.")
            print("Run: python install_deps.py\n")

# Read the long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name='pyRTX',
    version='0.1.0',
    author='Gael Cascioli',  
    author_email='gael.cascioli@nasa.gov',  
    description='Non grav. forces modelling for deep space probes using raytracing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gaelccc/pyRTX',  
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    
    # No dependencies listed here - all handled by install_deps.py
    setup_requires=[],
    install_requires=[],
    
    # Command classes for installation
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },
    
    # Package data
    package_data={
        'pyRTX': [
            'lib/*',
            'data/*',
        ],
    },
    
    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)