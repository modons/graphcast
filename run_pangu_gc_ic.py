"""
Adapted from run_cycle.py for panguweather. This version runs the GraphCast optimal IC generated by Trent.

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            May 2024

Revisions:

"""

import numpy as np
import logging
import yaml
import h5py
import sys
sys.path.append('../panguweather-dynamics')
import panguweather_utils as pw
import torch
import onnxruntime as ort

# initialize logger
logger = pw.get_logger()

# load config
logger.info('reading configuration file...')
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# convenience vars
Nvars_pl = len(config['var_pl'])
Nvars_sfc = len(config['var_sfc'])
Nlevels = len(config['levels'])
nhours = config['nhours']
ohr = config['ohr']
only_500 = config['only_500']

# model time step defaults to 24h unless 6h output is requested
if ohr == 6:
    dhr = 6
else:
    dhr = 24

# output path
opath = config['path_output']

# output data file name
#outfile = opath+'pangu_graphcast_ic.h5'
#outfile = opath+'pangu_graphcast_control_ic.h5'

# read initial condition
# old
#infile_iv = config['path_input']+'graphcast_ic_on_pangu.h5'
#infile_iv = config['path_input']+'graphcast_control_ic_on_pangu.h5'
# new
#infile = 'graphcast_control_ic_on_pangu_95_epochs.h5'
#infile = 'graphcast_optimal_ic_on_pangu_95_epochs.h5'
#infile = 'graphcast_optimal_ic_on_pangu_95_epochs_T21.h5'
#infile = 'graphcast_optimal_ic_on_pangu_95_epochs_T62.h5'
#infile = 'graphcast_optimal_ic_on_pangu_95_epochs_T100.h5'
#infile = 'graphcast_optimal_ic_on_pangu_95_epochs_T160.h5'
#infile = 'graphcast_optimal_ic_on_pangu_95_epochs_T180.h5'
# regional optimals:
#infile = 'graphcast_optimal_ic_on_pangu_99_epochs_region.h5'
#infile = 'graphcast_optimal_ic_on_pangu_99_epochs_region_T62.h5'
#infile = 'graphcast_optimal_ic_on_pangu_99_epochs_region_T21.h5'
#infile = 'graphcast_optimal_ic_on_pangu_99_epochs_region_tropics_only.h5'
#infile = 'graphcast_optimal_ic_on_pangu_99_epochs_region_extratropics_only.h5'
#infile = 'graphcast_optimal_ic_on_pangu_93_epochs_15days_region.h5'
#infile = 'graphcast_optimal_ic_on_pangu_10day_control_check.h5'
#infile = 'graphcast_control_on_pangu_2021-06-20T00.h5'
#infile = 'graphcast_reg_optimal_on_pangu_2021-06-20T00.h5'
#infile = 'graphcast_reg_optimal_on_pangu_T60_2021-06-20T00.h5'
#infile = 'graphcast_reg_optimal_on_pangu_T30_2021-06-20T00.h5'
#infile = 'graphcast_reg_optimal_on_pangu_15day.h5'
#infile = 'graphcast_control_on_pangu_15day_2021-06-15T00.h5'
infile = 'graphcast_reg_optimal_on_pangu_NH_None_2021-06-20T00.h5'
infile = 'graphcast_gl_optimal_on_pangu_NH_None_2021-06-20T00.h5'
#
infile_iv = config['path_input']+infile
outfile = opath+infile[:-3]+'_solution.h5'

logger.info('reading perturbation file...'+infile_iv)
h5f = h5py.File(infile_iv, 'r')
ivp_reg_pl = h5f['input_pl'][:]
ivp_reg_sfc = h5f['input_sfc'][:]
lat = h5f['lat'][:]
lon = h5f['lon'][:]
h5f.close()

logger.info('checking on GPU availability...')
try:
    device_index = torch.cuda.current_device()
    providers = [("CUDAExecutionProvider",{"device_id": device_index,},)]
    logger.info('Got a GPU---no waiting!')
except:
    providers = ["CPUExecutionProvider"]
    logger.info('Using CPU---no problem!')

# paths to model weights
pangu24 = config['path_model']+'pangu_weather_24.onnx'
pangu6 = config['path_model']+'pangu_weather_6.onnx'

num_threads = 1
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = num_threads

logger.info('starting ONNX session for 24h model...')
ort_session_24 = ort.InferenceSession(pangu24,sess_options=options,providers=providers)
if ohr == '6':
    logger.info('starting ONNX session for 6h model...')
    ort_session_6 = ort.InferenceSession(pangu6,sess_options=options,providers=providers)
    
t = 0
ofile = outfile[:-3]+'_'+str(t)+'h.h5'
logger.info('writing the initial condition to: '+ofile)

h5f = h5py.File(ofile, 'w')
if only_500:
    h5f.create_dataset('ivp_pl_save',data=ivp_reg_pl[:,5,:,:])    
else:
    h5f.create_dataset('ivp_pl_save',data=ivp_reg_pl)
    h5f.create_dataset('ivp_sfc_save',data=ivp_reg_sfc)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
h5f.close()
 
# initialize 'old' states for each model with the IC for their first step
pl_last_24 = np.copy(ivp_reg_pl)
sfc_last_24 = np.copy(ivp_reg_sfc)

# loop over forecast lead time
for t in np.arange(dhr,nhours+1,dhr):

    if t == 6:
        logger.info('first step: 6h model')
        ivp_pl_run = ivp_reg_pl
        ivp_sfc_run = ivp_reg_sfc
        ort_session = ort_session_6
    elif np.mod(t,24)==0:
        logger.info(str(t)+' 24h model')
        ivp_pl_run = pl_last_24
        ivp_sfc_run = sfc_last_24
        ort_session = ort_session_24
    else:
        logger.info(str(t)+' 6h model')
        ivp_pl_run = pl_last
        ivp_sfc_run = sfc_last
        ort_session = ort_session_6
  
    pl_tmp,sfc_tmp = pw.run_panguweather(ort_session,1,ivp_pl_run,ivp_sfc_run)
    
    pl_last = pl_tmp[-1,:]
    sfc_last = sfc_tmp[-1,:]
    
    if np.mod(t,24)==0:
        print('copying 24 hour output for the next 24 step IC...')
        pl_last_24 = np.copy(pl_last)
        sfc_last_24 = np.copy(sfc_last)
        
    # write to a file (no lat,lon; that's in the IC file)
    ofile = outfile[:-3]+'_'+str(t)+'h.h5'
    print('writing to: ',ofile)
    h5f = h5py.File(ofile, 'w')
    if only_500:
        h5f.create_dataset('ivp_pl_save',data=pl_last[:,5,:,:])        
    else:
        h5f.create_dataset('ivp_pl_save',data=pl_last)
        h5f.create_dataset('ivp_sfc_save',data=sfc_last)
    h5f.close()
