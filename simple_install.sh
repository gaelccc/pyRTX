#!/bin/bash

## Setup configuration
ENV_NAME="prt-new" # The name of the conda environment to be created
MAIN_FLD="." # The folder where pyRTX lives
LIB_FLD="./lib/" # The folder where to download and install embree and cgal
EMBREE2_VERSION="2.17.7"
EMBREE3_VERSION="3.13.5"
CGAL_VERSION="5.6"
BOOST_VERSION="1.82.0"



## Setup script
## DO NOT MODIFY [unless you know what you're doing :) ]
######################################################################################################
# Create and activate environment
echo "...... Creating conda env ......."
conda create --name $ENV_NAME python=3.8 --channel default --channel anaconda 
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install dependencies
echo "...... Installing dependencies  ......."
pip install $MAIN_FLD -r requirements.txt

# Move into the lib directory
mkdir $LIB_FLD
cd $LIB_FLD

# Install Embree 2
echo "...... Installing Embree 2 ......."
DWN_URL="https://github.com/embree/embree/releases/download/v${EMBREE2_VERSION}/"
wget "${DWN_URL}embree-${EMBREE2_VERSION}.x86_64.linux.tar.gz"
tar -xf "embree-${EMBREE2_VERSION}.x86_64.linux.tar.gz"

cd "embree-$EMBREE2_VERSION".x86_64.linux
source embree-vars.sh
cd ..

# Install Embree 3
echo "...... Installing Embree 2 ......."
DWN_URL="https://github.com/embree/embree/releases/download/v${EMBREE3_VERSION}/"
wget "${DWN_URL}embree-${EMBREE3_VERSION}.x86_64.linux.tar.gz"
tar -xf "embree-${EMBREE3_VERSION}.x86_64.linux.tar.gz"
cd "embree-$EMBREE3_VERSION".x86_64.linux
source embree-vars.sh
cd ..

# Clone python embree 
git clone https://github.com/sampotter/python-embree
cd python-embree
# NOTE: temporary fix
LINE_TO_COMMENT="rtcSetDeviceErrorFunction(self._device, simple_error_function, NULL);"
sed -i "/$LINE_TO_COMMENT/s/^/# /" embree.pyx

# Install python embree
pip install .
cd ..

# Install CGAL & Boost
echo "...... Installing CGAL ......."
DWN_URL="https://github.com/CGAL/cgal/releases/download/v${CGAL_VERSION}/CGAL-${CGAL_VERSION}.tar.xz"
wget "${DWN_URL}"
tar -xf "CGAL-${CGAL_VERSION}.tar.xz"

VRS_UND="${BOOST_VERSION//./_}"
DWN_URL="https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${VRS_UND}.tar.gz"
wget "${DWN_URL}"
tar -xf "boost_${VRS_UND}.tar.gz"


# Download and install the aabb binder
git clone https://github.com/steo85it/py-cgal-aabb.git
cd py-cgal-aabb

BOOST_INCLUDE="../boost_${VRS_UND}"
CGAL_INCLUDE="../CGAL-${CGAL_VERSION}/include"
BOOST_ABS="$(realpath $BOOST_INCLUDE)"
CGAL_ABS="$(realpath $CGAL_INCLUDE)"

sed  -i "13a\"$BOOST_ABS\",\"$CGAL_ABS\"" setup.py
python setup.py build_ext --inplace
pip install .

cd ..
rm -f *.tar*


# Run tests
echo "...... Running Tests ......."
cd ../tests
bash run_tests.sh

