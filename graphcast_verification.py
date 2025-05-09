"""
# need to run in conda environment shtools

test pangu solutions for Trent's optimals

started from paper plotting code

"""
#----start config
ndays = 350 # can't go further because no verification for long leads!
#ndays = 10 # testing
verbose = False
#verbose = True
#ftype = 'control'
ftype = 'optimal'
lev = 500

# option to specify custom forecast lead times (do not include zero; it is added below)
#custom = False # use this for all lead times
lead_times = [12,24,36,48,72,96,120,240,360]; custom = True

# 00 or 12UTC initialized forecast files
#ssdate = '2020-01-01T00'; HH = '00'
ssdate = '2020-01-01T12'; HH = '12'

# spacing between initialization times in hours
init_dt = 24 

# path to GC control IC files; use for verification
gc_c_path = '/glade/campaign/univ/uwas0139/paper2/control_ic/'
if ftype == 'control':
    gc_f_path = '/glade/work/tvonich/paper2/gc_control_predictions/'
elif ftype == 'optimal':
    gc_f_path = '/glade/work/tvonich/paper2/gc_optimal_predictions/'
    gc_o_path = '/glade/campaign/univ/uwas0139/paper2/optimal/BEST/'

#----end config

import numpy as np
import xarray as xr
import h5py
import sys
import datetime
import matplotlib
from pyshtools.expand import SHExpandDH
from pyshtools.spectralanalysis import spectrum
import rda
import glob

def even_grids(u,v):
    # for grids having odd number of latitudes, put on even number by averaging
    u_even = (u[:-1]+u[1:])/2.
    v_even = (v[:-1]+v[1:])/2.
    return u_even,v_even

def compute_spectrum(f):
    sh = SHExpandDH(f,sampling=2) # sampling=2 means nlon = 2*nlat
    spec = spectrum(sh)
    return spec

def process_forecast(uf,vf,uv,vv):
    """
    compute forecast statistics
    uf,vf: forecast u and v wind components
    uv,vv: verification u and v wind components
    """
    # return results in a dictionary
    forecast_info = {}
    forecast_info['KE'] = {}

    # forecast error
    uerr = (uf - uv)#/np.sqrt(2) # Boffetta & Musacchio (2017) eqn (2)
    verr = (vf - vv)#/np.sqrt(2) # Boffetta & Musacchio (2017) eqn (2)
    
    # put u,v data on even grids
    ufe,vfe = even_grids(uf,vf)
    uve,vve = even_grids(uv,vv)
    uee,vee = even_grids(uerr,verr)

    # forecast, verification, and error kinetic energy
    KE_f = 0.5*(ufe**2 + vfe**2)
    KE_v = 0.5*(uve**2 + vve**2)
    KE_e = 0.5*(uee**2 + vee**2)

    # power spectra
    forecast_info['KE']['f'] = compute_spectrum(KE_f)
    forecast_info['KE']['v'] = compute_spectrum(KE_v)
    forecast_info['KE']['e'] = compute_spectrum(KE_e)
    
    return forecast_info

def spec_avg_by_lead(fdict,stype):
    # compute KE spectra averages over all initalization times as a function of lead time
    # stype is the quantity to average ('f','v','e'; forecast, verification, error)

    inits = list(fdict.keys())
    ninits = len(inits)
    leads = list(fdict[list(fdict.keys())[0]].keys())
    nleads = len(leads)
    nsh = len(fdict[inits[0]][leads[0]]['KE'][stype])

    results = {}
    all_leads = np.zeros([nleads,nsh])
    it = -1
    for lead in leads:
        it+=1
        all_dat = np.zeros([ninits,nsh])
        n = -1
        for init in fdict.keys():
            n+=1
            all_dat[n,:] = fdict[init][lead]['KE'][stype]
        # average over all inits
        results[lead] = np.mean(all_dat,axis=0)
        all_leads[it] = results[lead]

    # grand average over all 
    grand_avg = np.mean(all_leads,axis=0)
    return results,grand_avg

"""
GC control initial conditions: /glade/campaign/univ/uwas0139/paper2/control_ic
GC optimal initial conditions: /glade/campaign/univ/uwas0139/paper2/optimal/BEST
15 day optimal GC forecasts: /glade/work/tvonich/paper2/gc_optimal_predictions
15 day control GC forecasts: /glade/work/tvonich/paper2/gc_control_predictions

what to compute:
* forecast error as a function of lead time 
* forecast error KE spectrum as a function of lead time
* KE spectrum as a function of lead time
*---
* eventually: cospectrum and coherence between GC & PW
"""
if ftype == 'optimal':
    # need to make a list with all filenames because numbers vary (nonstandard filenames)
    fils_input_o = glob.glob(gc_o_path+'BEST_*')
    # add zero to the lead time list
    lead_times.insert(0,0)

# numpy datetime version of starting date
nsdate = np.datetime64(ssdate)

# dictionary with all forecast verification info
forecast_info_all = {}

# loop over initialization 
for k in range(ndays):
    print('day=',k)
    if ftype == 'control':
        gc_forecast_file = gc_f_path+np.datetime_as_string(nsdate)[:-3]+'_'+HH+'_lead_60.zarr'
    elif ftype == 'optimal':
        gc_forecast_file = gc_f_path+np.datetime_as_string(nsdate)+'_lead_60.zarr'
    if verbose: print('forecast file: ',gc_forecast_file)
    ds_gc_f = xr.open_dataset(gc_forecast_file)
    if not custom and k == 0:
        # fetch all lead times on first pass
        lead_times = ds_gc_f['time'].to_numpy()/np.timedelta64(1,'h')
        nleads = len(lead_times)
        
    # loop over forecast lead times: note only have 00&12UTC files!
    ilead = -1
    forecast_info_leads = {}
    for lead in lead_times[:]:
        ilead+=1
        valid_time = nsdate+np.timedelta64(int(lead),'h')
        if verbose: print('lead=',int(lead),ilead,'valid time=',np.datetime_as_string(valid_time,unit='D'))
        hour = valid_time.astype('datetime64[h]').astype(int) % 24
        # 00 and 12 are the only valid times right now
        if hour==0:
            gc_verif_file = gc_c_path+np.datetime_as_string(valid_time,unit='D')+'_00_2.nc'
        elif hour==12:
            gc_verif_file = gc_c_path+np.datetime_as_string(valid_time,unit='D')+'_12_2.nc'
        else:
            # skip rest of this loop for invalid times
            continue
            
        ds_gc_v = xr.open_dataset(gc_verif_file)
        # trent saved these with two times: t-6, t. verification is at t (i.e., index 1)
        udat_gc_v = ds_gc_v['u_component_of_wind'].sel(level=lev,batch=0,time=ds_gc_v['time'][1]).to_numpy()
        vdat_gc_v = ds_gc_v['v_component_of_wind'].sel(level=lev,batch=0,time=ds_gc_v['time'][1]).to_numpy()
        # forecast
        if ftype == 'optimal' and lead == 0:
            # optimal initial condition rather than a forecast
            substring = np.datetime_as_string(nsdate)
            gc_ic_o = [s for s in fils_input_o if substring in s][0]
            tmp = xr.open_dataset(gc_ic_o)
            udat_gc_f = tmp['u_component_of_wind'].sel(level=lev,batch=0,time=tmp['time'][1]).to_numpy()
            vdat_gc_f = tmp['v_component_of_wind'].sel(level=lev,batch=0,time=tmp['time'][1]).to_numpy()
        else:
            # time is jacked up in Trent's optimal forecast files
            if ftype == 'optimal':
                ttime = ds_gc_f['time'][0]+np.timedelta64(lead-6,'h')
            else:
                ttime = np.timedelta64(lead,'h')
            udat_gc_f = ds_gc_f['u_component_of_wind'].sel(level=lev,batch=0,time=ttime).to_numpy()
            vdat_gc_f = ds_gc_f['v_component_of_wind'].sel(level=lev,batch=0,time=ttime).to_numpy()
        forecast_info_leads[int(lead)] = process_forecast(udat_gc_f,vdat_gc_f,udat_gc_v,vdat_gc_v)
           
    # increment forecast initialization time
    forecast_info_all[np.datetime_as_string(nsdate)] = forecast_info_leads
    nsdate = nsdate+np.timedelta64(init_dt,'h')

# save results to a file
ofile = 'graphcast_KE_spectra_'+ftype+'_'+str(lev)+'_'+HH+'UTC.npy'
forecast_info_all['lead_times'] = lead_times
np.save(ofile,forecast_info_all) 
    
"""
To Do:
- pull code out to a script DONE
- run full year for both optimal and control DONE
- save dictionaries to a file DONE
- need to handle the optimal IC as a special case, but include in the dictionary DONE
"""
