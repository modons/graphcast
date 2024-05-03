"""

make the tendency file for steady mean state

"""
import numpy as np
import xarray as xr

# paths to model weights, model stats files, and input ERA5 datasets
#rpath = '/home/disk/ice4/hakim/data/ai-models/graphcast/' # root path to everything
rpath = '/glade/work/hakim/data/ai-models/graphcast/'
path_input = rpath+'input/'
path_output = rpath+'output/'
#graphcast_version = 'small' # small version (1 degree, precip input/output)
graphcast_version = 'oper'  # "operational" version (0.25 degree, precip output only)
zonal_mean = True
#zonal_mean = False

if graphcast_version == 'small':
    fm = path_input+'graphcast_pw_DJF_mean_small.nc'
    fo = path_output+'graphcast_pw_DJF_mean_dt_small.nc'
else:
    if zonal_mean:
        fm = path_input+'graphcast_pw_DJF_zm_mean_oper.nc'
        fo = path_output+'graphcast_pw_DJF_zm_mean_dt_oper.nc'
    else:
        fm = path_input+'graphcast_pw_DJF_mean_oper.nc'
        fo = path_output+'graphcast_pw_DJF_mean_dt_oper.nc'

mean_state = xr.load_dataset(fm).compute()
ds_dt = xr.load_dataset(fo).compute()
print(ds_dt)

# this is the tendency to subtract in the inference step
tmp = ds_dt - mean_state

# drop datetime (this is the only method that seems to work; from graphcast code)
del tmp.coords["datetime"]

print(tmp)

# save and test in inference code
if graphcast_version == 'small':
    ft = path_input+'graphcast_DJF_6h_dt_small.nc'
else:
    if zonal_mean:
        ft = path_input+'graphcast_DJF_zm_6h_dt_oper.nc'
    else:
        ft = path_input+'graphcast_DJF_6h_dt_oper.nc'

print('saving here:',ft)
tmp.to_netcdf(ft)
