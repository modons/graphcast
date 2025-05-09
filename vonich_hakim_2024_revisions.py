# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python [conda env:earth2mip]
#     language: python
#     name: conda-env-earth2mip-py
# ---

"""
test figures in revision

production figures involving spatial plots for the paper

"""

import numpy as np
import xarray as xr
import h5py
import sys
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
# need these?
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MultipleLocator


# ERA verification
v_era = '/glade/work/hakim/data/ai-models/graphcast/input/graphcast_oper_2021062100.nc'
v_ds_full = xr.load_dataset(v_era)
z500_era5=np.flip(v_ds_full['geopotential'].isel(batch=0,level=7,time=0).to_numpy()/9.81,axis=0)


print(z500_era5.shape)
print(v_ds_full)
print(v_ds_full['time'][0])
print(v_ds_full['time'])


# +
# lat,lon boundaries of the regional optimizition domain
# add the optimization domain
# Define the coordinates of the square
lat1, lon1 = 42, -110  # bottom left corner
lat2, lon2 = 60, -130  # top right corner
lat3, lon3 = 60, -110  # top left corner
lat4, lon4 = 42, -130  # bottom right corner

# colormap
ccmap = plt.get_cmap('coolwarm').copy()
ccmap.set_extremes(under='yellow', over='yellow')
norm = TwoSlopeNorm(vmin=-25, vcenter=0, vmax=25)

# ERA5 lat,lon grid
lat_pw = 90 - np.arange(721) * 0.25
lon_pw = np.arange(1440) * 0.25
Nlat_pw = len(lat_pw)
Nlon_pw = len(lon_pw)

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 12  # Set the font size

# +
# set geographical limits for the plots; use to subset data so contours appear correctly
lat_limits = [30,70]
spw = np.where(lat_pw == lat_limits[0])[0][0]
npw = np.where(lat_pw == lat_limits[1])[0][0]
print(spw,lat_pw[spw])         
print(npw,lat_pw[npw])         
lon_limits =[200,280] # need a little extra
wpw = np.where(lon_pw == lon_limits[0])[0][0]
epw = np.where(lon_pw == lon_limits[1])[0][0]
print(wpw,lon_pw[wpw])         
print(epw,lon_pw[epw])         

# limited coords for plotting subsets
latpw = lat_pw[npw:spw]
lonpw = lon_pw[wpw:epw]


# +
# control
cfile = '/glade/work/tvonich/inputs/10day_verification'

# perturbed regional optimal
pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/reg/40_steps_99_epochs_10_patient_[42.0, 60.0, 230.0, 250.0]_times_all_.nc'
#pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/reg/20_steps_97_epochs_10_patient_[42.0, 60.0, 230.0, 250.0]_times_all_.nc'

# load the gc source control file
c_ds = xr.load_dataset(cfile)
#print(c_ds)

# load the gc source perturbed file
p_ds = xr.load_dataset(pfile)
#print(p_ds)

z500_ctrl = c_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81 # 40 steps
#z500_ctrl = c_ds['geopotential'].isel(batch=0,time=21,level=7).to_numpy()/9.81 # 20 steps
z500_reg = p_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81
# perturbation from the control forecast:
#z500_reg_pert = z500_reg - z500_ctrl
# anomaly relative to ERA5:
z500_reg_pert = z500_reg - c_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81

lat_gc = c_ds['lat'].to_numpy()
lon_gc = c_ds['lon'].to_numpy()


# +
# set geographical limits for the plots; use to subset data so contours appear correctly
#lat_limits = [30,70]; lon_limits =[200,280] # need a little extra # original
#lat_limits = [10,70]; lon_limits =[60,280] # testing
lat_limits = [-70,70]; lon_limits =[0,359] # testing

sgc = np.where(lat_gc == lat_limits[0])[0][0]
ngc = np.where(lat_gc == lat_limits[1])[0][0]
print(sgc,lat_gc[sgc])         
print(ngc,lat_gc[ngc])         
wgc = np.where(lon_gc == lon_limits[0])[0][0]
egc = np.where(lon_gc == lon_limits[1])[0][0]
print(wgc,lon_gc[wgc])         
print(egc,lon_gc[egc])         

# limited coords for plotting subsets
latgc = lat_gc[sgc:ngc]
longc = lon_gc[wgc:egc]

# -

print(c_ds['time'][1])
print(p_ds['time'][1])
print(c_ds['datetime'][0,1])
#print(c_ds['datetime'][0,21])


# +
zcints = np.arange(4800,6000,60.)

proj = ccrs.Robinson(central_longitude=-120.)
#fig,ax = plt.subplots(2,2,figsize=(8,8),subplot_kw=dict(projection=proj))
fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
#fig,ax = plt.subplots(1,2,figsize=(8,8),subplot_kw=dict(projection=proj))
#fig.tight_layout(w_pad=0.5,h_pad=-0.) # w_pad:horizontal; h_pad:vertical

#tfield = z500_reg_pert[sgc:ngc,wgc:egc]
#zfield = z500_ctrl[sgc:ngc,wgc:egc]
tfield = z500_reg_pert
zfield = z500_ctrl
#plat = latgc
#plon = longc
plat = lat_gc
plon = lon_gc
ttl = '500Z'
label = 'B'
i=0;j=0

# custom here
#ax[i,j].set_extent([lon_limits[0]+45,lon_limits[1]-20,lat_limits[0],lat_limits[1]],crs=ccrs.PlateCarree()) 
ax.coastlines(color='gray')
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
# Plot the regional optimal domain edge
ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())

# customized colormap here...
ccmap = plt.get_cmap('coolwarm').copy()
#ccmap.set_extremes(under='yellow', over='yellow')
norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10) 

alpha=1.0
vmax = 20
vmin = - vmax
#temp_plot = ax.pcolormesh(plon,plat,tfield,cmap=ccmap,edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,norm=norm)
temp_plot = ax.pcolormesh(plon,plat,tfield,cmap='bwr',edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,vmin=vmin,vmax=vmax)
geo_contours = ax.contour(plon,plat,zfield,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)
#ax[i,j].clabel(geo_contours,inline=True, fontsize=10)
#ax[i,j].title.set_text(ttl)
#ax[i,j].title.set_size(12)
#tx = ax[i,j].text(211,35,label,transform=ccrs.PlateCarree(),fontsize=10,fontweight='bold', va='top')
#tx.set_bbox(dict(facecolor='white',edgecolor='white',alpha=0.75,linewidth=0))

#plt.savefig('paper_graphcast_perts_t_0.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
# next steps:
# 1) plot GC forecasts for 6, and 24 h
# 2) plot PW 24 h forecast
# ---makes a 4 panel plot

# +
# graphcast solutions

# control
gc_ctrl = '/glade/u/home/tvonich/graph_repo/results/predictions/jun30_right/right_justify_40_ctrl_forecast.nc'
gcc_ds = xr.load_dataset(gc_ctrl)
lat_gc = gcc_ds['lat'].to_numpy()
lon_gc = gcc_ds['lon'].to_numpy()
z500_ctrl = gcc_ds['geopotential'].isel(batch=0,level=7).to_numpy()/9.81
print(z500_ctrl.shape)

# regional optimal
gc_reg = '/glade/u/home/tvonich/graph_repo/results/predictions/jun30_right/right_justify_40_reg_forecast.nc'
gcc_ds = xr.load_dataset(gc_reg)
z500_reg = gcc_ds['geopotential'].isel(batch=0,level=7).to_numpy()/9.81
print(z500_reg.shape)

# -

for it in range(4):
    print(c_ds['datetime'].to_numpy()[0][it+2])

# +
it = 0
zcints = np.arange(4800,6000,60.)

# first four forecast times, every 6 hours
for it in range(4):
# 24-240h, every 24h
#for it in range(3,40,4):
    
    proj = ccrs.Robinson(central_longitude=-120.)
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
    
    # anomalies from the control simulation:
    #tfield = z500_reg[it,:,:] - z500_ctrl[it,:,:]
    # anomalies from ERA5 on GC grid
    tfield = z500_reg[it,:,:] - c_ds['geopotential'].isel(batch=0,time=it+2,level=7).to_numpy()/9.81
    # contours of control solution
    #zfield = z500_ctrl[it,:,:]
    # contours of perturbed solution
    zfield = z500_reg[it,:,:]
    zfield_wrap,plon_wrap = add_cyclic_point(zfield,coord=plon,axis=1)

    plat = lat_gc
    plon = lon_gc
    
    ax.coastlines(color='gray')
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
    # Plot the regional optimal domain edge
    ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())
    
    # customized colormap here...
    ccmap = plt.get_cmap('coolwarm').copy()
    #ccmap.set_extremes(under='yellow', over='yellow')
    norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10) 
    
    alpha=1.0
    vmax = 20
    vmin = - vmax
    #temp_plot = ax.pcolormesh(plon,plat,tfield,cmap=ccmap,edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,norm=norm)
    temp_plot = ax.pcolormesh(plon,plat,tfield,cmap='bwr',edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,vmin=vmin,vmax=vmax)
    geo_contours = ax.contour(plon_wrap,plat,zfield_wrap,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)
    #temp_plot = ax.pcolormesh(plon,plat,tfield,cmap='bwr',edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,norm=colors.CenteredNorm())
    #geo_contours = ax.contour(plon,plat,zfield,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)
    cbar = fig.colorbar(temp_plot,fraction=0.046,orientation='horizontal',pad =0.01,shrink=0.75,extend='both')#,boundaries=[260,320])

    #plt.savefig('paper_graphcast_perts_t_'+str((it+1)*6)+'.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
for it in range(3,4,4):
    print(it,str((it+1)*6))

it = 3
tfield = z500_reg[it,:,:] - c_ds['geopotential'].isel(batch=0,time=it+2,level=7).to_numpy()/9.81

vmax = np.max(np.abs(tfield))
vmin = -vmax
print(vmax)
# -

"""

Pangu weather perturbations from the PW control simulation

"""

# +
rpath = '/glade/work/hakim/data/ai-models/panguweather/graphcast_testing/'

# lead time in a string
#st = '0'
st = '24'

# PW control forecast
sfile = 'graphcast_control_on_pangu_2021-06-20T00_solution_'
infile = rpath+sfile+st+'h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
h5f.close()
z500_pw_ctrl = ivp_pl_save[0,5,:,:]/9.81
print(z500_pw_ctrl.shape)

# PW PNW coptimal forecast
sfile = 'graphcast_reg_optimal_on_pangu_2021-06-20T00_solution_'
infile = rpath+sfile+st+'h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
h5f.close()
z500_pw_reg = ivp_pl_save[0,5,:,:]/9.81
print(z500_pw_reg.shape)

# pangu needs full ERA5 resolution to compute 24h forecast errors at 00 UTC 21 June 2021
v_era = '/glade/work/hakim/data/ai-models/graphcast/input/graphcast_oper_2021062100.nc'
v_ds_full = xr.load_dataset(v_era)
z500_era5=np.flip(v_ds_full['geopotential'].isel(batch=0,level=7,time=0).to_numpy()/9.81,axis=0)


# +
zcints = np.arange(4800,6000,60.)

proj = ccrs.Robinson(central_longitude=-120.)
fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
#tfield = z500_pw_reg - z500_pw_ctrl
#zfield = z500_pw_ctrl
tfield = z500_pw_reg - z500_era5
zfield = z500_pw_reg

plat = lat_pw
plon = lon_pw
ax.coastlines(color='gray')
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
# Plot the regional optimal domain edge
ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())

# customized colormap here...
ccmap = plt.get_cmap('coolwarm').copy()
norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10) 

alpha=1.0
temp_plot = ax.pcolormesh(plon,plat,tfield,cmap=ccmap,edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,norm=norm)
geo_contours = ax.contour(plon,plat,zfield,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)

#plt.savefig('paper_pangu_perts_t_'+st+'.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
# option to run this cell to load all data for perturbed IC files
# perturbed regional optimal
pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/reg/40_steps_99_epochs_10_patient_[42.0, 60.0, 230.0, 250.0]_times_all_.nc'
#pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/reg/20_steps_97_epochs_10_patient_[42.0, 60.0, 230.0, 250.0]_times_all_.nc'
#pfile = '/glade/u/home/tvonich/graph_repo/results/optimization/optimal_input/jun30_right_justify/reg/40_threshold_1e-1_steps_99_epochs_20_patient_[42.0, 60.0, 230.0, 250.0]_times_all_.nc'
p_ds = xr.load_dataset(pfile)

# graphcast forecasts from these perturbed ICs:
gc_reg = '/glade/u/home/tvonich/graph_repo/results/predictions/jun30_right/right_justify_40_reg_forecast.nc'
#gc_reg = '/glade/u/home/tvonich/graph_repo/results/predictions/jun30_right/40_PNW_THRESH.nc'
gcc_ds = xr.load_dataset(gc_reg)
z500_reg = gcc_ds['geopotential'].isel(batch=0,level=7).to_numpy()/9.81


# +
"""

4 panel figure for the paper supplement

"""

# perturbationfigure (4 panel: (a) ERA5, (b) ctrl optim, (c) global opt (d) reg opt)

proj = ccrs.Robinson(central_longitude=-120.)
fig,ax = plt.subplots(2,2,figsize=(8,8),subplot_kw=dict(projection=proj))
fig.tight_layout(w_pad=0.5,h_pad=-10.) # w_pad:horizontal; h_pad:vertical

# set up for each panel
ii = -1
for i in range(2):
    for j in range(2):
        ii +=1
        print('ii=',ii)
        ax[i,j].coastlines(color='gray')
        ax[i,j].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
        # Plot the regional optimal domain edge
        ax[i,j].plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=2,transform=ccrs.PlateCarree())

        if ii == 0:
            tfield = p_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81 - c_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81
            zfield = p_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81
            plat = lat_gc
            plon = lon_gc
            ttl = 'GraphCast optimal (t = 0h)'
            label = 'A'
        elif ii == 1:
            tfield = z500_reg[0,:,:] - c_ds['geopotential'].isel(batch=0,time=2,level=7).to_numpy()/9.81
            zfield = z500_reg[0,:,:]
            plat = lat_gc
            plon = lon_gc
            ttl = 'GraphCast optimal (t = 6h)'
            label = 'B'
        elif ii == 2:
            tfield = z500_reg[2,:,:] - c_ds['geopotential'].isel(batch=0,time=4,level=7).to_numpy()/9.81
            zfield = z500_reg[2,:,:]
            plat = lat_gc
            plon = lon_gc
            ttl = 'GraphCast optimal (t = 24h)'
            label = 'C'
        else:
            tfield = z500_pw_reg - z500_era5
            zfield = z500_pw_reg
            plat = lat_pw
            plon = lon_pw 
            ttl = 'Pangu-Weather (t = 24h)'
            label = 'D'

        zfield_wrap,plon_wrap = add_cyclic_point(zfield,coord=plon,axis=1)
        alpha=1.0
        vmax = 20
        vmin = - vmax
        temp_plot = ax[i,j].pcolormesh(plon,plat,tfield,cmap='bwr',edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,vmin=vmin,vmax=vmax)
        geo_contours = ax[i,j].contour(plon_wrap,plat,zfield_wrap,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=0.5)
        #ax[i,j].clabel(geo_contours,inline=True, fontsize=10)
        ax[i,j].set_title(ttl,fontdict={'fontsize': 10, 'fontweight': 'medium'})
        #ax[i,j].title.set_size(12)
        #tx = ax[i,j].text(90,-70,label,transform=ccrs.PlateCarree(),fontsize=10,fontweight='bold', va='top')
        #tx = ax[i,j].text(xpos,ypos,label,transform=ccrs.PlateCarree(),fontsize=10,fontweight='bold', va='top')
        xpos = ax[i,j].get_xbound()[0]*.95
        ypos = ax[i,j].get_ybound()[0]*.85
        tx = ax[i,j].text(xpos,ypos,label,fontsize=10,fontweight='bold', va='top')
        tx.set_bbox(dict(facecolor='white',edgecolor='white',alpha=0.75,linewidth=0))
       
# shared colorbar
cbar = fig.colorbar(temp_plot,ax=ax.ravel().tolist(),fraction=0.046,orientation='horizontal',pad =0.01,shrink=0.5,extend='both')#,boundaries=[260,320])
cbar.set_label('500Z Anomalies (m)')
cbar.ax.tick_params(labelsize=10)

plt.savefig('paper_perturbation_figure.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
proj = ccrs.Robinson(central_longitude=-120.)
fig,ax = plt.subplots(2,2,figsize=(8,8),subplot_kw=dict(projection=proj))
fig.tight_layout(w_pad=0.5,h_pad=-14.) # w_pad:horizontal; h_pad:vertical

# set up for each panel
ii = -1
for i in range(2):
    for j in range(2):
        ii +=1
        print('ii=',ii)
        ax[i,j].coastlines(color='gray')
        ax[i,j].add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
        # Plot the regional optimal domain edge
        #ax[i,j].plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=2,transform=ccrs.PlateCarree())
        #xpos = ax[i,j].get_position().bounds[0]
        #ypos = ax[i,j].get_position().bounds[1]
        xpos = ax[i,j].get_xbound()[0]*.95
        ypos = ax[i,j].get_ybound()[0]*.85
        label= 'A'
        #print(xpos,ypos)
        tx = ax[i,j].text(xpos,ypos,label,fontsize=10,fontweight='bold', va='top')
        tx.set_bbox(dict(facecolor='white',edgecolor='white',alpha=0.75,linewidth=0))
        #
        print(ax[i,j].get_xbound())
        print(ax[i,j].get_ybound())
#print(dir(ax[0,0]))


# +
#
# PW T60/T30 solution plots
#

# +
rpath = '/glade/work/hakim/data/ai-models/panguweather/graphcast_testing/'

#trunc = 'T60'
trunc = 'T30'

# lead time in a string
#st = '0'
st = '24'

# PW control forecast
sfile = 'graphcast_control_on_pangu_2021-06-20T00_solution_'
infile = rpath+sfile+st+'h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
h5f.close()
z500_pw_ctrl = ivp_pl_save[0,5,:,:]/9.81
u500_pw_ctrl = ivp_pl_save[3,5,:,:]
v500_pw_ctrl = ivp_pl_save[4,5,:,:]
print(z500_pw_ctrl.shape)

# PW PNW coptimal forecast
sfile = 'graphcast_reg_optimal_on_pangu_'+trunc+'_2021-06-20T00_solution_'
infile = rpath+sfile+st+'h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
h5f.close()
z500_pw_reg = ivp_pl_save[0,5,:,:]/9.81
u500_pw_reg = ivp_pl_save[3,5,:,:]
v500_pw_reg = ivp_pl_save[4,5,:,:]
print(z500_pw_reg.shape)


# +
zcints = np.arange(4800,6000,60.)

proj = ccrs.Robinson(central_longitude=-120.)
fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
tfield = z500_pw_reg - z500_pw_ctrl
zfield = z500_pw_ctrl

plat = lat_pw
plon = lon_pw
ax.coastlines(color='gray')
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
# Plot the regional optimal domain edge
ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())

# customized colormap here...
ccmap = plt.get_cmap('coolwarm').copy()
norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10) 

alpha=1.0
temp_plot = ax.pcolormesh(plon,plat,tfield,cmap=ccmap,edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,norm=norm)
geo_contours = ax.contour(plon,plat,zfield,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)

plt.savefig('paper_pangu_'+trunc+'_perts_t_'+st+'.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
#
# zoom in on a region and plot winds with heights
#

# +
rpath = '/glade/work/hakim/data/ai-models/panguweather/graphcast_testing/'

# lead time in a string
#st = '0'
st = '24'

# PW control forecast
sfile = 'graphcast_control_on_pangu_2021-06-20T00_solution_'
infile = rpath+sfile+st+'h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
h5f.close()
z500_pw_ctrl = ivp_pl_save[0,5,:,:]/9.81
u500_pw_ctrl = ivp_pl_save[3,5,:,:]
v500_pw_ctrl = ivp_pl_save[4,5,:,:]
print(z500_pw_ctrl.shape)

# PW PNW coptimal forecast
sfile = 'graphcast_reg_optimal_on_pangu_2021-06-20T00_solution_'
infile = rpath+sfile+st+'h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
h5f.close()
z500_pw_reg = ivp_pl_save[0,5,:,:]/9.81
u500_pw_reg = ivp_pl_save[3,5,:,:]
v500_pw_reg = ivp_pl_save[4,5,:,:]
print(z500_pw_reg.shape)


# +
udat = u500_pw_reg - u500_pw_ctrl
vdat = v500_pw_reg - v500_pw_ctrl
zfield = z500_pw_reg - z500_pw_ctrl

proj = ccrs.Robinson(central_longitude=-120.)
fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
ax.coastlines(color='red',linewidths=2)
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
# Plot the regional optimal domain edge
ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())

if st == '24':
    vscale = 150 # vector scaling (counterintuitive:smaller=larger arrows)
elif st == '0':
    vscale = 5 # vector scaling (counterintuitive:smaller=larger arrows)
latskip = 2
lonskip = 2
#zcints = [-10,-8,-6,-4,-2,-1,1,2,4,6,8,10]
zcints = list(np.arange(-5,0,1))+list(np.arange(1,6,1))
alpha = 1.0
col = 'g'
cs = ax.quiver(plon[::lonskip],plat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
geo_contours = ax.contour(plon,plat,zfield,levels=zcints,colors='black',transform=ccrs.PlateCarree(),linewidths=1)

#lat_limits = [10,70]; lon_limits =[180,300] # testing
lat_limits = [10,40]; lon_limits =[180,220] # testing
ax.set_extent([lon_limits[0]+10,lon_limits[1]-0,lat_limits[0],lat_limits[1]],crs=ccrs.PlateCarree()) 
if st == '24':
    qk = ax.quiverkey(cs, 0.75, 0.15, 10, r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)
elif st == '0':
    qk = ax.quiverkey(cs, 0.75, 0.15, 1, r'$1~ m/s$', labelpos='E',coordinates='figure',color=col)

#plt.savefig('paper_pangu_T60_wind_perts_zoom_t_'+st+'.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)
plt.savefig('paper_pangu_wind_perts_zoom_t_'+st+'.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
#
# GC forecast diffs on polar stereographic grid
#

# +
it = 0
zcints = np.arange(4800,6000,60.)

# first four times, every 6 hours
#for it in range(4):
# 24-240h, every 24h
for it in range(3,40,4):

    proj = ccrs.NorthPolarStereo(central_longitude=180.0)
    #ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=180.0))
    fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
    # Limit the map to 40 degrees latitude and above\n",
    ax.set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
    ax.coastlines()
    # Compute a circle in axes coordinates, which we can use as a boundary",
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
    tfield = z500_reg[it,:,:] - z500_ctrl[it,:,:]
    #zfield = z500_ctrl[it,:,:]
    zfield = z500_reg[it,:,:]
    zdat_wrap,plon_wrap = add_cyclic_point(zfield,coord=plon,axis=1)
    plat = lat_gc
    plon = lon_gc
    
    ax.coastlines(color='gray')
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
    # Plot the regional optimal domain edge
    ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())
    
    vmax = np.max(np.abs(tfield[90:,:]))
    vmin = -vmax
    alpha=1.0
    temp_plot = ax.pcolormesh(plon,plat,tfield,cmap='bwr',edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,vmin=vmin,vmax=vmax)
    geo_contours = ax.contour(plon_wrap,plat,zdat_wrap,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)
    cbar = fig.colorbar(temp_plot,fraction=0.05,orientation='horizontal',pad =0.01,shrink=0.75,extend='both')#,boundaries=[260,320])
    plt.text(140,-10,'t='+str((it+1)*6)+'h',transform=ccrs.PlateCarree())
    plt.savefig('paper_graphcast_perts_t_'+str((it+1)*6)+'_NPS.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)

# +
# t = 0 on the NPS projection
z500_ctrl_init = c_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81 # 40 steps
#z500_ctrl = c_ds['geopotential'].isel(batch=0,time=21,level=7).to_numpy()/9.81 # 20 steps
z500_reg_init = p_ds['geopotential'].isel(batch=0,time=1,level=7).to_numpy()/9.81

proj = ccrs.NorthPolarStereo(central_longitude=180.0)
#ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=180.0))
fig,ax = plt.subplots(figsize=(8,8),subplot_kw=dict(projection=proj))
# Limit the map to 40 degrees latitude and above\n",
ax.set_extent([-180, 180, 0, 90], ccrs.PlateCarree())
ax.coastlines()
# Compute a circle in axes coordinates, which we can use as a boundary",
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

tfield = z500_reg_init - z500_ctrl_init
#zfield = z500_ctrl[it,:,:]
zfield = z500_reg_init
zdat_wrap,plon_wrap = add_cyclic_point(zfield,coord=plon,axis=1)
plat = lat_gc
plon = lon_gc

ax.coastlines(color='gray')
ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='gray')
# Plot the regional optimal domain edge
ax.plot([lon1, lon3, lon2, lon4, lon1], [lat1, lat3, lat2, lat4, lat1],color='m',linestyle='--',lw=1,transform=ccrs.PlateCarree())

vmax = np.max(np.abs(tfield[90:,:]))
vmin = -vmax
alpha=1.0
temp_plot = ax.pcolormesh(plon,plat,tfield,cmap='bwr',edgecolors='none',shading='nearest',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True,vmin=vmin,vmax=vmax)
geo_contours = ax.contour(plon_wrap,plat,zdat_wrap,levels=zcints,transform=ccrs.PlateCarree(),colors='k',linewidths=1)
cbar = fig.colorbar(temp_plot,fraction=0.05,orientation='horizontal',pad =0.01,shrink=0.75,extend='both')#,boundaries=[260,320])
plt.text(140,-10,'t=0h',transform=ccrs.PlateCarree())
plt.savefig('paper_graphcast_perts_t_0_NPS.pdf',bbox_inches='tight',pad_inches=0.25,dpi=300)
# -


