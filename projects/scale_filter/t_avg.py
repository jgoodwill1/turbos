from turbos.sim.p3d.p3d_2 import P3D
from turbos.average import time_average
import numpy as np


# %%
rc = P3D(
    "/archive/tulasi/149p6_rs",
    filenum="000",
)
nslice = np.arange(rc.numslices)

# %%


avg = time_average(
    rc,
    nslice,
    variables='all',
)

# %%
print(avg)

# %%
avg.to_netcdf('/home/goodwill/turbos/projects/scale_filter/data/149p6_rs.nc')



