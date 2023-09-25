# pyRTX v0.0.2

A collection of functions, classes and tools for the computation of non-gravitational acceleration on space probes leveraging ray tracing techniques.

# Installation

Installation in a new environment called "py38" with Anaconda (suggested).

(Installation tested on Linux 3.10.0-1160.95.1.el7.x86_64) 
### Download pyRTX and setup an empty environment
1) Download pyRTX folder
2) `conda create --name py38 python=3.8  --channel default --channel anaconda`
3) `conda activate py38`
### Install dependencies
5) `pip install "PATH_TO_MAIN_FOLDER" -r requirements-pre.txt` (e.g., pip install -e ~/username/pyRTX -r requirements.txt)

### Minimal ray tracing dependencies
For the ray-tracing algorithms to work the ray tracing kernel programs need to be installed. 
Here we will detail the installation procedures for Embree 2, Embree 3 and CGAL. 
The user can decide to install only one of the three kernels.

### Installing Embree 2
Fetch and download the Embree version which suits your needs (e.g., https://github.com/embree/embree/releases/tag/v2.17.7)  
Unzip/untar the downloaded archive and place the resulting folder somewhere (e.g., in ~/usr/lib)  
Enter the Embree directory and   
`source embree-vars.sh` (or .csh depending on the shell in use)  

You can test the succesfull installation of Embree 2 by opening a python terminal and: 
`from pyembree import rtcore_scene` 

### Installing Embree3


### Installing CGAL
Download CGAL from the official website (e.g., https://github.com/CGAL/cgal/releases/tag/v5.6)  
`tar xf CGAL-5.6.tar.xz`  
Download Boost from the official website (e.g., https://www.boost.org/users/history/version_1_82_0.html)  
`tar xf boost_1_82_0.tar.gz`  
Install the `aabb` binder  (thanks @steo85it !)  
`git clone https://github.com/steo85it/py-cgal-aabb.git`  
modify the setup.py file and add the path to the `include` dirs of CGAL and Boost  
from inside the `py-cgal-aabb` folder:  
`python setup.py build_ext --inplace`  
`pip install .`  


Several examples are provided in the examples folder. See the README.txt in the examples folder for a description of the various examples.

## Quickstart and installation testing
Download the data required for running the examples running in the `examples` folder:

`python download_lro_kernels.py` 

# [Documentation](https://gaelccc.github.io/pyRTX)
(work in progress)


# Change log
Version 0.0.2 implements the same functionalities as v0.0.1 but the code structure has been heavily restructured. Backwards compatibility is guaranteed for functions and classes call signs but not for imports syntax.
