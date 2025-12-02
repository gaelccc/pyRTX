# setup.py
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
import subprocess
import sys
import platform
from pathlib import Path

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
    
    install_requires=[
        'numba',
        'numpy',
        'matplotlib',
        'pyglet==1.5.15',
        'rtree',
        'scipy',
        'shapely',
        'spiceypy',
        'tqdm',
        'trimesh==3.10.7',
        'wget',
        'pathos',
        'cython',
        'pytest',
        'xarray',
    ],
    
    # Package data - install_deps.py should be included
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
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',  # Add your license
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    
    # Python version requirement
    python_requires='>=3.8',
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
)