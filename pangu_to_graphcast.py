"""

convert the pangu mean state file to a graphcast-compatible input file

"""

# paths to model weights, model stats files, and input ERA5 datasets
#rpath = '/home/disk/ice4/hakim/data/ai-models/graphcast/' # root path to everything
rpath = '/glade/work/hakim/data/ai-models/graphcast/'
path_model_weights = rpath
path_model_stats = rpath+'stats/'
path_input = rpath+'input/'
path_output = rpath+'output/'
#perturb_ic = True
perturb_ic = False
#geoadjust = True
geoadjust = False
graphcast_version = 'small' # small version (1 degree, precip input/output)
#graphcast_version = 'oper'  # "operational" version (0.25 degree, precip output only)
#zonal_mean = True
zonal_mean = False
#ntims = 20 # for perturbed IC experiments
#ntims = 9
#ntims = 6 # for mean state test
ntims = 2 # convert mean state

import numpy as np
import xarray as xr
import h5py
import sys
sys.path.append('../panguweather-dynamics')
import panguweather_utils as pw
import datetime
from datetime import date

def pw_to_gc_pl(gc_dims,pw_array,lat_stride,lon_stride):

    # make sure gc_dims is configured correctly!
    ntims = gc_dims[1]
    
    # select every 4th point, move the displaced levels axis, and flip latitude
    if len(pw_array.shape)>2:
        # pl vars
        gc_array = np.zeros(gc_dims) #[ntims,nsamp,nlevs,nlat,nlon]
        # flip latitude & vertical axis
        tmp = np.flip(np.flip(pw_array[:,lat_stride[:,None],lon_stride[None,:]],axis=1),axis=0)
        for k in range(ntims):
            gc_array[0,k,:,:,:] = tmp

    else:
        # surface vars
        gc_array = np.zeros(gc_dims) #[ntims,nsamp,nlat,nlon]
        tmp = np.flip(pw_array[lat_stride[:,None],lon_stride[None,:]],axis=0)
        for k in range(ntims):
            gc_array[0,k,:,:] = tmp
    
    return gc_array

# pangu definitions
pw_config = {}
pw_config['var_sfc'] = ["msl", "u10m", "v10m", "t2m"]
pw_config['var_pl'] = ["z", "q", "t", "u", "v"]
# missing in pw, and in gc:
pw_config['missing_pl'] = ["vertical_velocity"] 
pw_config['missing_sfc'] = ["total_precipitation_6hr"]
pw_config['levels'] = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
Nvars_pl = len(pw_config['var_pl'])
Nvars_sfc = len(pw_config['var_sfc'])
Nlevels = len(pw_config['levels'])

# graphcast definitions in terms of pw (pw:gc)
gc_config = {}
gc_config['var_sfc'] = {"msl":"mean_sea_level_pressure", "u10m":"10m_u_component_of_wind",
                        "v10m":"10m_v_component_of_wind", "t2m":"2m_temperature","total_precipitation_6hr":"total_precipitation_6hr"}
gc_config['var_pl'] = {"z":"geopotential", "q":"specific_humidity", "t":"temperature", 
                       "u":"u_component_of_wind", "v":"v_component_of_wind","vertical_velocity":"vertical_velocity"}
gc_config['levels'] = list(np.flipud(np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50])))
gc_config['static'] = ['geopotential_at_surface','land_sea_mask','toa_incident_solar_radiation']
# -

print('checking pw-gc mappings...\n')
for v in pw_config['var_sfc']+pw_config['missing_sfc']:
    print(v,':',gc_config['var_sfc'][v])
print('----')
for v in pw_config['var_pl']+pw_config['missing_pl']:
    print(v,':',gc_config['var_pl'][v])

# read pw time-mean state
#pmean = '/glade/work/hakim/data/ai-models/panguweather/mean_state/orig/'
pmean = '/glade/work/hakim/data/ai-models/graphcast/climo/ERA5_climo_june_1979_2020.h5'
#mean_pl,mean_sfc,lat_pw,lon_pw = pw.fetch_mean_state(pmean,zm=zonal_mean)
h5f = h5py.File(pmean,'r')
mean_pl = h5f['mean_pl'][:]
mean_sfc = h5f['mean_sfc'][:]
lat_pw = h5f['lat'][:]
lon_pw = h5f['lon'][:]

# option to perturb the mean state
if perturb_ic:
    print('perturbing the initial condition...')
    infile_iv = '/glade/work/hakim/data/ai-models/panguweather/input/cyclone_DJF_40N_150E_regression.h5'
    h5f = h5py.File(infile_iv, 'r')
    regf_pl = h5f['regf_pl'][:]
    regf_sfc = h5f['regf_sfc'][:]
    lat = h5f['lat'][:]
    lon = h5f['lon'][:]
    iminlat = h5f['iminlat'][()]
    imaxlat = h5f['imaxlat'][()]
    iminlon = h5f['iminlon'][()]
    imaxlon = h5f['imaxlon'][()]
    h5f.close()
    # add the perturbation to the mean state
    ivp_reg_pl = np.copy(mean_pl)
    for var in range(Nvars_pl):
        for k in range(Nlevels):
            # z500 only:
            if geoadjust and (var != 0 or k !=5):
                ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] + 0.*regf_pl[var,k,:,:]
            # all but u,v 
            #if geoadjust and (var == 3 or var == 4):
            #    ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] + 0.*regf_pl[var,k,:,:]
            else:
                ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] + regf_pl[var,k,:,:]

    ivp_reg_sfc = np.copy(mean_sfc)
    for var in range(Nvars_sfc):
        if geoadjust:
            ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] + 0.*regf_sfc[var,:,:]
        else:
            ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] + regf_sfc[var,:,:]

    # rename back to mean state vars for mapping and output
    mean_pl = ivp_reg_pl
    mean_sfc = ivp_reg_sfc
    
# Load the graphcast template dataset
if graphcast_version == 'small':
    fname = path_input+'dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc'
else:
    fname = path_input+'dataset_source-era5_date-2022-01-01_res-0.25_levels-13_steps-04.nc'
    
with open(fname, "rb") as f:
    ds_template = xr.load_dataset(f).compute()

mslp_gc = ds_template['mean_sea_level_pressure'].to_numpy()
z_gc = ds_template['geopotential'].to_numpy()
lat_gc = ds_template['lat'].to_numpy()
lon_gc = ds_template['lon'].to_numpy()
level_gc = ds_template['level'].to_numpy()

# hack for the time variable
#tims = ds_template['time'].to_numpy()[:ntims]
tims = ds_template['time'].data[:ntims]#.astype('int32')
dt = tims[1]
tims = np.arange (0,dt*ntims,dt)
print(type(tims),type(dt))
print('tims=',tims)
# add datetime array
#ds['datetime'] = (["batch","time"], ds_template['datetime'].to_numpy()[:,:ntims])
#ds['datetime'] = (["batch","time"], ds_template['datetime'].data[:,:ntims].astype('int32'))
dtims = ds_template['datetime'].data[:,:ntims]#.astype('int32')
print(dtims)
#tdelta=datetime.timedelta(microseconds=dt/1000.)
ndtims = []
k=-1
for t in np.arange(ntims):
    print(t,dt*t,dtims[0,0]+dt*t)
    k+=1
    ndtims.append(dtims[0,0]+dt*t)
dtims = np.array([ndtims])
print('dtims=',dtims)

if graphcast_version == 'small':
    lat_stride = np.arange(0, len(lat_pw), 4)
    lon_stride = np.arange(0, len(lon_pw), 4)
else:
    lat_stride = np.arange(0, len(lat_pw), 1)
    lon_stride = np.arange(0, len(lon_pw), 1)

# [batch,time,level,lat,lon]]
gc_dims_pl = tuple(list([1,ntims])+list(z_gc.shape[2:]))
# [batch,time,lat,lon]]
gc_dims_sfc = tuple(list([1,ntims])+list(mslp_gc.shape[2:]))

print('gc_dims:',gc_dims_pl)

k = -1
for v in pw_config['var_pl']:
    k+=1
    print('working on '+v+' '+gc_config['var_pl'][v]+' '+str(k))
    gc_array = pw_to_gc_pl(gc_dims_pl,mean_pl[k,:,:,:],lat_stride,lon_stride)
    print(v,gc_array.shape)
    if k == 0:
        ds = xr.Dataset(
            data_vars={
                gc_config['var_pl'][v]:(["batch","time","level","lat","lon"], gc_array)
                },
            coords={'lat':lat_gc,'lon':lon_gc,'level':level_gc,'time':tims,'datetime':(["batch","time"],dtims)},
            attrs=dict(description="Pangu-Weather DJF mean state on Graphcast grid."),
        )
    else:
        ds[gc_config['var_pl'][v]] = (["batch","time","level","lat","lon"], gc_array)

k = -1
for v in pw_config['var_sfc']:
    k+=1
    print('working on '+v+' ',str(k))
    gc_array = pw_to_gc_pl(gc_dims_sfc,mean_sfc[k,:,:],lat_stride,lon_stride)
    ds[gc_config['var_sfc'][v]] = (["batch","time","lat","lon"], gc_array)

# add the missing variables: 'vertical_velocity' 'total_precipitation_6hr'
for v in pw_config['missing_pl']:
    print('working on '+v)
    #ds[v] = (["batch","time","level","lat","lon"], 0.*z_gc[:,:ntims,:,:,:])
    ds[v] = (["batch","time","level","lat","lon"], 0.*ds['geopotential'].data[:])

for v in pw_config['missing_sfc']:
    #ds[v] = (["batch","time","lat","lon"], 0.*mslp_gc[:,:ntims,:,:])
    ds[v] = (["batch","time","lat","lon"], 0.*ds['mean_sea_level_pressure'].data[:])

# add the static variables: 'geopotential_at_surface' 'land_sea_mask' 'toa_incident_solar_radiation'
for v in gc_config['static']:
    print('working on '+v)
    if v == 'toa_incident_solar_radiation':
        # zero everywhere:
        #ds[v] = (["batch","time","lat","lon"], 0.*ds['mean_sea_level_pressure'].data[:])
        # fixed at the initial time
        gc_array = np.zeros(gc_dims_sfc)
        for k in range(ntims):
            gc_array[0,k,:,:] = ds_template[v].to_numpy()[0,0,:,:]
        ds[v] = (["batch","time","lat","lon"], gc_array)
    else:
        ds[v] = (["lat","lon"], ds_template[v].to_numpy())

# add the coordinates
ds['lat'] = (["lat"],lat_gc)
ds['lon'] = (["lon"],lon_gc)
ds['level'] = (["level"],level_gc)
ds['time'] = (["time"],tims)
ds['datetime'] = (["batch","time"],dtims)

print(ds)

# save the dataset
if graphcast_version == 'small':
    if perturb_ic:
        of = rpath+'/input/graphcast_pw_DJF_perturbed_small.nc'
    else:
        of = rpath+'/input/graphcast_pw_DJF_mean_small.nc'
        of = '/glade/work/hakim/data/ai-models/graphcast/climo/ERA5_climo_june_1979_2020_gc_small.h5'
else:
    if perturb_ic:
        if geoadjust:
            of = rpath+'/input/graphcast_pw_DJF_perturbed_geoadjust_oper.nc'
#            of = rpath+'/input/graphcast_pw_DJF_perturbed_geoadjust_oper_windonly.nc'
        else:
            if zonal_mean:                
                of = rpath+'/input/graphcast_pw_DJF_zm_perturbed_oper.nc'
            else:
                of = rpath+'/input/graphcast_pw_DJF_perturbed_oper.nc'
    else:
        if zonal_mean:
            #of = rpath+'/input/graphcast_pw_DJF_zm_mean_oper.nc'
            of = rpath+'/input/graphcast_pw_DJF_zm_mean_oper_test.nc'
        else:
            of = rpath+'/input/graphcast_pw_DJF_mean_oper.nc'
            
ds.to_netcdf(of)
    
# check as in code:
example_check = xr.load_dataset(of).compute()
print(example_check)


