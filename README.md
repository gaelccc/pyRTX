# pyRTX v0.0.2

A collection of functions, classes and tools for the computation of non-gravitational acceleration on space probes leveraging ray tracing techniques.

# Installation

Installation in a new environment called "py38" with Anaconda (suggested).

1) Download pyRTX folder
2) conda create --name py38 python=3.8 --file requirements.txt --channel default --channel anaconda
3) conda activate py38
4) pip install "PATH_TO_MAIN_FOLDER" (e.g., pip install ~/username/pyRTX)

Several examples are provided in the examples folder. See the README.txt in the examples folder for a description of the various examples.

## Quickstart and installation testing
Download the data required for running the examples running in the `examples` folder:
`python download_lro_kernels.py` 

# [Documentation](https://gaelccc.github.io/pyRTX)
(work in progress)


# Change log
Version 0.0.2 implements the same functionalities as v0.0.1 but the code structure has been heavily restructured. Backwards compatibility is guaranteed for functions and classes call signs but not for imports syntax.
