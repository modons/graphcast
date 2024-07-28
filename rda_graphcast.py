"""
read rda and save in graphcast xarray .nc files
"""

#graphcast_version = 'small' # small version (1 degree, precip input/output)
graphcast_version = 'oper'  # "operational" version (0.25 degree, precip output only)
ntims = 3
rpath = '/glade/work/hakim/data/ai-models/graphcast/'
path_model_weights = rpath
path_model_stats = rpath+'stats/'
path_input = rpath+'input/'
path_output = rpath+'output/'

import numpy as np
import datetime as dt
import rda
import xarray as xr

#date =  "2022121500" # YYYYMMDDHH 
#date =  "2021062000" # YYYYMMDDHH 
#date =  "2021062100" # YYYYMMDDHH 
#date =  "2021062200" # YYYYMMDDHH 
#date =  "2021062300" # YYYYMMDDHH 
date =  "2021062500" # YYYYMMDDHH 
#date =  "2021063000" # YYYYMMDDHH 
start_time = dt.datetime(int(date[:4]),int(date[4:6]),int(date[6:8]),int(date[8:10]))
print(start_time+dt.timedelta(days=3))

#
# upper level data
#

pw_pl = ['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150', 'z100', 'z50', 'q1000',
        'q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', 'q100', 'q50', 't1000', 't925',
        't850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50', 'u1000', 'u925', 'u850',
        'u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50', 'v1000', 'v925', 'v850', 'v700',
        'v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50']

# surface variables
pw_sfc = ["msl", "u10m", "v10m", "t2m"]

print('reading pl rda:')
ds = rda.DataSource(pw_pl)
print(ds)
data = ds[start_time]
print(type(data))
data_pl=np.squeeze(data.to_numpy())
print(type(data_pl))
print(data_pl.shape)
input_pl = np.reshape(data_pl,[5,13,721,1440])
print(input_pl.shape)
print('this is the 500hPa pl data:')
print(input_pl[0,5,100:120,100]/9.81)
lat_pw = data['lat'].to_numpy()
lon_pw = data['lon'].to_numpy()
print('lat_pw:',lat_pw)

print('----')
print('z dat at all levels:')
print(input_pl[0,:,100,100]/9.81)
print('T dat at all levels:')
print(input_pl[2,:,100,100])

print('reading sfc rda:')
ds_sfc = rda.DataSource(pw_sfc)
data = ds_sfc[start_time]
data_sfc=np.squeeze(data.to_numpy())
input_sfc = np.reshape(data_sfc,[4,721,1440])
print('this is the t2m data:')
print(input_sfc[3,100:120,100])

#----------------------------------------
# conversion code follows
#----------------------------------------

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
    gc_array = pw_to_gc_pl(gc_dims_pl,input_pl[k,:,:,:],lat_stride,lon_stride)
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
    gc_array = pw_to_gc_pl(gc_dims_sfc,input_sfc[k,:,:],lat_stride,lon_stride)
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
    of = rpath+'/input/graphcast_small_'+date+'.nc'
else:
    of = rpath+'/input/graphcast_oper_'+date+'.nc'
            
ds.to_netcdf(of)
    
# check as in code:
example_check = xr.load_dataset(of).compute()
print(example_check)
