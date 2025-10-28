### ------------------------------------------------------------------------------------------------------- ###

#									  LRO MASS OBJECT BUILDING

# Example purpose:
# The spacecraft mass must be defined to compute non-gravitational accelerations.
# It can be a float, int or an xarray.
# Here we show how to store the spacecraft mass as an xarray.

# To compute accelerations, the mass is interpolated by simply returning the previous value of the point.

### ------------------------------------------------------------------------------------------------------- ###

import spiceypy as sp
import xarray as xr
import numpy as np
from pyRTX.core.analysis_utils import epochRange2

ref_epc		=  "2010 may 10 09:25:00"
duration    =  50000  									  # seconds
timestep    =  100
METAKR      = '../example_data/LRO/metakernel_lro.tm'     # metakernel

sp.furnsh(METAKR)

# Define epochs (they must be sorted)
epc_et0 = sp.str2et( ref_epc )
epc_et1 = epc_et0 + duration
times   = epochRange2(startEpoch = epc_et0, endEpoch = epc_et1, step = timestep)

# Define values (here we use a constant value in kg for every epoch)
values = [2000.] * len(times)

# Create the xarray
MASS = xr.Dataset(
    data_vars = dict( mass=("time", values),),
    coords = dict(time = times,),
    attrs = dict(description="LRO mass related data."),	
	)

# Save 
MASS.to_netcdf('mass/lro_mass.nc', encoding = MASS.encoding.update({'zlib': True, 'complevel': 1}))

sp.unload(METAKR)

### ------------------------------------------------------------------------------------------------------- ###