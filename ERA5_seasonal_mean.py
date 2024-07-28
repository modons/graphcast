"""
Read and time average monthly ERA5 data from NCAR RDA

Originator: Greg Hakim
            ghakim@uw.edu
            University of Washington

"""

import numpy as np
import rda_month as rdam
import datetime as dt
import h5py

# define range of years and months for the time average
years = list(range(1979,2020))
#months = [12,1,2]
#months = [6]
months = [7]

# write data here
opath = '/glade/work/hakim/data/ai-models/graphcast/climo/'

# ERA5 grid
lat = 90 - np.arange(721) * 0.25
lon = np.arange(1440) * 0.25

# pangu-weather variable definitions ("channels")
pw_sfc = ['msl', 'u10m', 'v10m', 't2m']
pw_pl = ['z1000', 'z925', 'z850', 'z700', 'z600', 'z500', 'z400', 'z300', 'z250', 'z200', 'z150','z100', 'z50','q1000','q925', 'q850', 'q700', 'q600', 'q500', 'q400', 'q300', 'q250', 'q200', 'q150', \
'q100', 'q50','t1000', 't925','t850', 't700', 't600', 't500', 't400', 't300', 't250', 't200', 't150', 't100', 't50','u1000', 'u925', 'u850','u700', 'u600', 'u500', 'u400', 'u300', 'u250', 'u200', 'u150', 'u100', 'u50','v1000', 'v925', 'v850', 'v700','v600', 'v500', 'v400', 'v300', 'v250', 'v200', 'v150', 'v100', 'v50']

# instances of rda objects
ds_pl = rdam.DataSource(pw_pl)
ds_sfc = rdam.DataSource(pw_sfc)

# loop over years
it = 0
for year in years:
    print('year=',year)
    # loop over months
    for month in months:
        it+=1
        newtime = dt.datetime(year,month,1,0)
        print('time=',newtime)
        data = ds_pl[newtime]
        data_pl = np.reshape(np.squeeze(data.to_numpy()),[5,13,721,1440])
        data = ds_sfc[newtime]
        data_sfc = np.squeeze(data.to_numpy())
        if it == 1:
            tavg_pl = data_pl
            tavg_sfc = data_sfc
        else:
            tavg_pl = tavg_pl + data_pl
            tavg_sfc = tavg_sfc + data_sfc

print('it=',it)
tavg_pl = tavg_pl/float(it)
tavg_sfc = tavg_sfc/float(it)

# test save file
#outfile = opath+'mean_DJF.h5'
#outfile = opath+'mean_june.h5'
outfile = opath+'mean_july.h5'
h5f = h5py.File(outfile, 'w')
h5f.create_dataset('mean_pl',data=tavg_pl)
h5f.create_dataset('mean_sfc',data=tavg_sfc)
h5f.create_dataset('lat',data=lat)
h5f.create_dataset('lon',data=lon)
h5f.create_dataset('years',data=years)
h5f.create_dataset('months',data=months)
h5f.create_dataset('nmonths',data=it)
h5f.close()



