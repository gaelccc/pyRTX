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
Fetch the Embree version which suits your needs 
https://github.com/embree/embree/releases/tag/v2.17.7

### Additional dependencies (Embree 2, 3 and CGAL)
7) `pip install "PATH_TO_MAIN_FOLDER" -r requirements-post.txt`

Several examples are provided in the examples folder. See the README.txt in the examples folder for a description of the various examples.

## Quickstart and installation testing
Download the data required for running the examples running in the `examples` folder:

`python download_lro_kernels.py` 

# [Documentation](https://gaelccc.github.io/pyRTX)
(work in progress)


# Change log
Version 0.0.2 implements the same functionalities as v0.0.1 but the code structure has been heavily restructured. Backwards compatibility is guaranteed for functions and classes call signs but not for imports syntax.
