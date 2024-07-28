"""
this version truncates perturbations in the initial condition

convert a graphcast nc file to a pangu-weather compatible input file

"""

# perturbed regional optimal
#pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/reg/40_steps_99_epochs_10_patient_[42.0, 60.0, 230.0, 250.0]_times_all_.nc'
# perturbed global optimal
pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/gl/40_steps_95_epochs_10_patient_all_times_all_.nc'
# control
cfile = '/glade/work/tvonich/inputs/10day_verification'

# name the converted output file
#ofile = 'graphcast_optimal_ic_on_pangu_10day_control_check.h5'
#ofile = 'graphcast_optimal_ic_on_pangu_10day_control_check2.h5'
#ofile = 'graphcast_control_on_pangu'
#ofile = 'graphcast_reg_optimal_on_pangu_T'
#ofile = 'graphcast_reg_optimal_on_pangu_NH_'
ofile = 'graphcast_gl_optimal_on_pangu_NH_'

# path to write the converted file
icpath = '/glade/work/hakim/data/ai-models/panguweather/graphcast_input/'

# truncation
#ntrunc = 60
#ntrunc = 30
ntrunc = None

import numpy as np
import xarray as xr
import h5py
import sys
sys.path.append('../panguweather-dynamics')
sys.path.append('../DL_DA/')
import panguweather_utils as pw
import DL_DA_verify as DLv
import datetime
from datetime import date
import spharm

def graphcast_to_pangu(gc_config,pw_config,ds,time_gc,specob_gc,specob_pw,lev=None,vars_pl=None,verbose=True):

    # given a GraphCast xarray dataset, convert to a Pangu-Weather numpy array
    # option lev chooses a single pressure level
    # option vars_pl specifies a list of pangu pressure-level variables

    if vars_pl == None:
        vars_pl = pw_config['var_pl']
        
    Nvars_pl = len(vars_pl)
    Nvars_sfc = len(pw_config['var_sfc'])
    Nlevels = len(pw_config['levels'])
    Nlat_pw = pw_config['Nlat_pw']
    Nlon_pw = pw_config['Nlon_pw']
    
    #
    # pressure level variables
    #
    k = -1
    for v in vars_pl:
        k+=1
        if verbose: print(v,gc_config['var_pl'][v])
        if k == 0:
            # initialize
            if lev == None:
                dims = np.flip(np.flip(ds[gc_config['var_pl'][v]].isel(batch=0,time=time_gc).to_numpy(),axis=0),axis=1).shape
            else:
                ilev = np.where(ds['level'].to_numpy()==lev)[0]
                if verbose: print('ilev=',ilev)
                dims = np.flip(np.flip(ds[gc_config['var_pl'][v]].isel(level=ilev,batch=0,time=time_gc).to_numpy(),axis=0),axis=1).shape
            tmp = tuple([Nvars_pl]+list(dims))
            pw_pl_on_gc = np.zeros(tmp)
            if verbose: print('pw shape:',pw_pl_on_gc.shape)                
        # flip latitude & vertical axis
        if lev == None:
            pw_pl_on_gc[k,:,:,:] = np.flip(np.flip(ds[gc_config['var_pl'][v]].isel(batch=0,time=time_gc).to_numpy(),axis=0),axis=1)
        else:
            pw_pl_on_gc[k,:,:,:] = np.flip(np.flip(ds[gc_config['var_pl'][v]].isel(level=ilev,batch=0,time=time_gc).to_numpy(),axis=0),axis=1)

    # spectral interpolation from GC onto PW grid
    if lev == None:
        pw_pl_on_pw = np.zeros([Nvars_pl,Nlevels,Nlat_pw,Nlon_pw],dtype=np.float32)
    else:
        pw_pl_on_pw = np.zeros([Nvars_pl,Nlat_pw,Nlon_pw],dtype=np.float32)
        
    for k in range(Nvars_pl):
        if verbose: print('working on var: ',k)
        if lev == None:
            for l in range(Nlevels):
                gcdat = pw_pl_on_gc[k,l,:,:].astype(dtype=np.float32)
                pwdat = spharm.regrid(specob_gc, specob_pw, gcdat, ntrunc=None, smooth=None)
                pw_pl_on_pw[k,l,:,:] = pwdat
        else:
            gcdat = pw_pl_on_gc[k,0,:,:].astype(dtype=np.float32)
            pwdat = spharm.regrid(specob_gc, specob_pw, gcdat, ntrunc=None, smooth=None)
            pw_pl_on_pw[k,:,:] = pwdat
        
    #
    # surface variables
    #
    pw_sfc_on_gc = np.zeros([Nvars_sfc,Nlat_gc,Nlon_gc])
    k = -1
    for v in pw_config['var_sfc']:
        k+=1
        if k == 0:
            # initialize
            dims = np.flip(np.flip(ds[gc_config['var_sfc'][v]].isel(batch=0,time=time_gc).to_numpy(),axis=0),axis=1).shape
            tmp = tuple([Nvars_sfc]+list(dims))
            pw_sfc_on_gc = np.zeros(tmp)
        
        # flip latitude axis
        pw_sfc_on_gc[k,:,:] = np.flip(ds[gc_config['var_sfc'][v]].isel(batch=0,time=time_gc).to_numpy(),axis=0)


    # spectral interpolation from GC onto PW grid
    pw_sfc_on_pw = np.zeros([Nvars_sfc,Nlat_pw,Nlon_pw],dtype=np.float32)
    for k in range(Nvars_sfc):
        if verbose: print('working on var: ',k)
        gcdat = pw_sfc_on_gc[k,:,:].astype(dtype=np.float32)
        pwdat = spharm.regrid(specob_gc, specob_pw, gcdat, ntrunc=None, smooth=None)
        pw_sfc_on_pw[k,:,:] = pwdat
    return pw_pl_on_pw,pw_sfc_on_pw

def config_setup():
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

    return pw_config,gc_config

# load configs
pw_config,gc_config = config_setup()

# load the gc source perturbed file
p_ds = xr.load_dataset(pfile)
print(p_ds)
# load the gc source control file
c_ds = xr.load_dataset(cfile)
print(c_ds)

# set up two grids: pw on gc grid and pw on pw grid (ERA5)
lat_gc = p_ds['lat'].to_numpy()
lon_gc = p_ds['lon'].to_numpy()
Nlat_gc = len(lat_gc)
Nlon_gc = len(lon_gc)
# ERA5 lat,lon grid
lat_pw = 90 - np.arange(721) * 0.25
lon_pw = np.arange(1440) * 0.25
Nlat_pw = len(lat_pw)
Nlon_pw = len(lon_pw)
# put in the configs
pw_config['Nlat_pw'] = Nlat_pw
pw_config['Nlon_pw'] = Nlon_pw
gc_config['Nlat_gc'] = Nlat_gc
gc_config['Nlon_gc'] = Nlon_gc

print(Nlat_gc,Nlon_gc)
print(Nlat_pw,Nlon_pw)

# make spectral objects for interpolation from graphcast to pangu
specob_gc = spharm.Spharmt(Nlon_gc,Nlat_gc,gridtype='regular',legfunc='computed')
specob_pw = spharm.Spharmt(Nlon_pw,Nlat_pw,gridtype='regular',legfunc='computed')

# test single level, single variable option
#time_gc_v = 0
#vars_pl = ['z']
#verif_pl,verif_sfc = graphcast_to_pangu(gc_config,pw_config,p_ds,time_gc_v,specob_gc,specob_pw,lev=500,vars_pl=vars_pl,verbose=False)
#print(verif_pl.shape)

# perturbed file
time_gc = 1 # perturbation file
pert_pl,pert_sfc = graphcast_to_pangu(gc_config,pw_config,p_ds,time_gc,specob_gc,specob_pw)
# control file
time_gc = 1 # 10_day_verification file (control)
control_pl,control_sfc = graphcast_to_pangu(gc_config,pw_config,c_ds,time_gc,specob_gc,specob_pw)

nppert_pl = pert_pl - control_pl
nppert_sfc = pert_sfc - control_sfc
np_pl = np.zeros_like(control_pl)
np_sfc = np.zeros_like(control_sfc)
Nvars_pl = len(pw_config['var_pl'])
Nvars_sfc = len(pw_config['var_sfc'])
Nlevels = len(pw_config['levels'])
print(ntrunc)
if ntrunc != None:
    for k in range(Nvars_pl):
        print('k=',k)
        for lev in range(Nlevels):
            perts = pert_pl[k,lev,:,:] - control_pl[k,lev,:,:]
            pert_spec = specob_pw.grdtospec(perts,ntrunc=ntrunc)
            np_pl[k,lev,:,:] = control_pl[k,lev,:,:] + specob_pw.spectogrd(pert_spec)

    for k in range(Nvars_sfc):
        print('k=',k)
        perts = pert_sfc[k,:,:] - control_sfc[k,:,:]
        pert_spec = specob_pw.grdtospec(perts,ntrunc=ntrunc)
        np_sfc[k,:,:] = control_sfc[k,:,:] + specob_pw.spectogrd(pert_spec)

else:
    print('using all spherical harmonics; truncating by another strategy...')
    #--- other pert strategies here ---
    # tropics only
    #np_pl = np.copy(control_pl)
    #np_sfc = np.copy(control_sfc)
    #np_pl[:,:,60*4:120*4+1,:] = nppert_pl[:,:,60*4:120*4+1,:]
    #np_sfc[:,60*4:120*4+1,:] = nppert_sfc[:,60*4:120*4+1,:]
    # extratropics only:
    #np_pl = np.copy(nppert_pl)
    #np_sfc = np.copy(nppert_sfc)
    #np_pl[:,:,60*4:120*4+1,:] = control_pl[:,:,60*4:120*4+1,:]
    #np_sfc[:,60*4:120*4+1,:] = control_sfc[:,60*4:120*4+1,:]
    # NH only
    np_pl = np.copy(pert_pl)
    np_sfc = np.copy(pert_sfc)
    np_pl[:,:,90*4+1:,:] = control_pl[:,:,90*4+1:,:]
    np_sfc[:,90*4+1:,:] = control_sfc[:,90*4+1:,:]
    
#--
# write the pangu IC file
#rgfile = icpath+ofile+str(ntrunc)+'.h5'
dd = c_ds['datetime'].isel(batch=0,time=time_gc).to_numpy()
dds = np.datetime_as_string(dd, unit='h')
rgfile = icpath+ofile+str(ntrunc)+'_'+dds+'.h5'
print('time from datetime in nc file: ',dds)

#rgfile = icpath+ofile
print('writing file: ',rgfile)    
h5f = h5py.File(rgfile, 'w')
h5f.create_dataset('input_pl',data=np_pl)
h5f.create_dataset('input_sfc',data=np_sfc)
h5f.create_dataset('lat',data=lat_pw)
h5f.create_dataset('lon',data=lon_pw)
h5f.close()

