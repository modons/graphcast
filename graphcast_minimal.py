# isolate the minimal code to run inference

#graphcast_version = 'small' # small version (1 degree, precip input/output)
graphcast_version = 'oper'  # "operational" version (0.25 degree, precip output only)
# run inference
#run = False
run = True
#perturb_ic = True
perturb_ic = False # use this for mean-state tendency
geoadjust = False
#geoadjust = True
#eval_steps = 1 # use this for mean-state tendency
eval_steps = 6 #geoadjust
#eval_steps = 17 # BW IVP
#eval_steps = 3 # tendency testing
make_steady = False # use this for mean-state tendency
#make_steady = True # BW IVP
#zonal_mean = True
zonal_mean = False

# paths to model weights, model stats files, and input ERA5 datasets
#rpath = '/home/disk/ice4/hakim/data/ai-models/graphcast/' # root path to everything
rpath = '/glade/work/hakim/data/ai-models/graphcast/' # root path to everything
path_model_weights = rpath
path_model_stats = rpath+'stats/'
path_input = rpath+'input/'
path_output = rpath+'output/'

import os
import dataclasses
import datetime
import functools
import math
import pandas as pd
import re
from typing import Optional

from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import losses
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree
import haiku as hk
import jax #this is 0.4.16 do not upgrade
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray

print('done importing...')
if perturb_ic:
    if graphcast_version == 'small':
        f = path_input+'graphcast_pw_DJF_perturbed_small.nc'
    else:
        if geoadjust:
            f = path_input+'graphcast_pw_DJF_perturbed_geoadjust_oper.nc'
#            f = path_input+'graphcast_pw_DJF_perturbed_geoadjust_oper_windonly.nc'
        else:
            if zonal_mean:
                f = path_input+'graphcast_pw_DJF_zm_perturbed_oper.nc'
            else:
                f = path_input+'graphcast_pw_DJF_perturbed_oper.nc'
else:
    if graphcast_version == 'small':
        f = path_input+'graphcast_pw_DJF_mean_small.nc'
    else:
        if zonal_mean:
            #f = path_input+'graphcast_pw_DJF_zm_mean_oper.nc'
            f = path_input+'graphcast_pw_DJF_zm_mean_oper_test.nc'
        else:
            #f = path_input+'graphcast_pw_DJF_mean_oper.nc'
            f = path_input+'graphcast_oper_2022121500.nc'

print('reading ic file:',f)
example_batch = xarray.load_dataset(f).compute()
print('input initial condition:\n',example_batch)

if make_steady:
    print('loading dt dataset...')
    if graphcast_version == 'small':
        filen = '/glade/work/hakim/data/ai-models/graphcast/output/graphcast_DJF_6h_dt.nc'
    else:
        if zonal_mean:
            filen = path_input+'graphcast_DJF_zm_6h_dt_oper.nc'
        else:
            filen = path_input+'graphcast_DJF_6h_dt_oper.nc'
        
    ds_dt = xarray.load_dataset(filen).compute()
else:
    ds_dt = None
    
# Update to deal with dataset_source split issue
def parse_file_parts(file_name):
    return dict(part.split("-", 1) for part in file_name.split("_") if "-" in part)

# Load the model
if graphcast_version == 'small':
    params_file = 'params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz' # Small
else:
    params_file = 'params_GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz' #Operational
   
#params_file = 'params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz' #37 LVL, 0.25 Degree
#params_file_path = f"/Users/tre_von/Desktop/graph_repo/params/{params_file}"
#params_file_path = f"/Users/hakim/data/ai-models/graphcast/{params_file}"
params_file_path = path_model_weights+params_file
with open(params_file_path, "rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)
params = ckpt.params
state = {}

model_config = ckpt.model_config
task_config = ckpt.task_config
print("Model description:\n", ckpt.description, "\n")
#print("Model license:\n", ckpt.license, "\n")

#model_config


# + cellView="form" id="ke2zQyuT_sMA"
# @title Build jitted functions

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True) #Can turn this to false for single step forecast probably.
  return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state  # DING DING DING
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config) #added per_timestep_loss
  loss, diagnostics = predictor.loss(inputs, targets, forcings) #defines loss -- subfunction of predictor
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

@hk.transform_with_state  
def timestep_loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config) #added per_timestep_loss
  per_timestep_losses, per_timestep_diagnostics = predictor.per_timestep_loss_and_diagnostics(inputs, targets, forcings) #defines loss -- subfunction of predictor
  return per_timestep_losses, per_timestep_diagnostics

#DING DING DING
def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Dont forget input_grads_fn at bottom!
def input_grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  # Function to compute loss with respect to inputs
  def compute_loss_wrt_inputs(inputs):
      # Prevent gradients for parameters and state
      params_no_grad = jax.lax.stop_gradient(params)
      #state_no_grad = jax.lax.stop_gradient(state)

      # Apply the loss function with no gradients for params and state
      # Add back state_no_grad after params_no_grad below if necessary
      (loss, _), _ = loss_fn.apply(
          params_no_grad, jax.random.PRNGKey(0), model_config, task_config, inputs, targets, forcings
      )
      return loss

  # Compute gradients with respect to inputs only
  grads_wrt_inputs = jax.grad(compute_loss_wrt_inputs)(inputs)
  return grads_wrt_inputs


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn):
  return functools.partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))

# if params is None:
#   params, state = init_jitted(
#       rng=jax.random.PRNGKey(0),
#       inputs=train_inputs,
#       targets_template=train_targets,
#       forcings=train_forcings)

input_grads_fn_jitted = jax.jit(input_grads_fn) #linked to input mod grad
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
timestep_loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(timestep_loss_fn.apply)))) #removed drop_state
grads_fn_jitted = with_params(jax.jit(with_configs(grads_fn)))
run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

# + [markdown] id="WEtSV8HEkHtf"
# # Load Data and Initialize Model

# + cellView="form" id="Yz-ekISoJxeZ"
# @title Load Weather Data and Normalization Data

#fname = 'dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc'
fname = 'graphcast_pw_DJF_mean.nc'
dataset_file_path = path_input+fname

# Path to the dataset file in the local directory
#dataset_file_path = '/Users/tre_von/Desktop/graph_repo/data/dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc'
#f"/Users/tre_von/Desktop/graph_repo/data/{dataset_file.value}"

# Load the dataset from the local file
#print('reading: ',dataset_file_path)
#with open(dataset_file_path, "rb") as f:
#with open(dataset_file_path) as f:
#    example_batch = xarray.load_dataset(f).compute()

#example_check = xarray.load_dataset(f,engine='netcdf4').compute()

# Check if the dataset has the required dimensions
assert example_batch.dims["time"] >= 3  # 2 for input, >=1 for targets


# Path to the local directory for normalization data
#local_stats_path = "/Users/tre_von/Desktop/graph_repo/stats"
local_stats_path = path_model_stats

# Load diffs_stddev_by_level from the local file
with open(f"{local_stats_path}/stats_diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xarray.load_dataset(f).compute()

# Load mean_by_level from the local file
with open(f"{local_stats_path}/stats_mean_by_level.nc", "rb") as f:
    mean_by_level = xarray.load_dataset(f).compute()

# Load stddev_by_level from the local file
with open(f"{local_stats_path}/stats_stddev_by_level.nc", "rb") as f:
    stddev_by_level = xarray.load_dataset(f).compute()

example_batch

# + cellView="form" id="tPVy1GHokHtk"
# @title Choose training and eval data to extract

# Removed training steps which would go here

#eval_steps = widgets.IntSlider(
#    value=example_batch.sizes["time"]-2, min=1, max=example_batch.sizes["time"]-2, description="Eval steps")
#train_steps = widgets.IntSlider(
#    value=example_batch.sizes["time"]-2, min=1, max=example_batch.sizes["time"]-2, description="Train steps")

#widgets.VBox([
#    eval_steps, train_steps,
#    widgets.Label(value="Run the next cell to extract the data. Rerunning this cell clears your selection.")
#])
#eval_steps = 16 # first good BW test case
#eval_steps = 1
train_steps = 1

# +
# @title Extract training and eval data

year = parse_file_parts(dataset_file_path.split('/')[-1].removesuffix(".nc"))

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
#    example_batch, target_lead_times=slice("6h", f"{train_steps.value*6}h"),
    example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"),
    **dataclasses.asdict(task_config))#, justify='left')

eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
#    example_batch, target_lead_times=slice("6h", f"{eval_steps.value*6}h"),
    example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"),
    **dataclasses.asdict(task_config))#, justify='left')

print("All Examples:  ", example_batch.dims.mapping)
print("Train Inputs:  ", train_inputs.dims.mapping)
print("Train Targets: ", train_targets.dims.mapping)
print("Train Forcings:", train_forcings.dims.mapping)
print("Eval Inputs:   ", eval_inputs.dims.mapping)
print("Eval Targets:  ", eval_targets.dims.mapping)
print("Eval Forcings: ", eval_forcings.dims.mapping)

if graphcast_version == 'small':
    eval_inputs['total_precipitation_6hr'] = 0.*eval_inputs['total_precipitation_6hr']

# # Run

# + cellView="form" id="7obeY9i9oTtD"
# @title Autoregressive rollout (loop in python) #Non-differentiable trajectories (at least for most) using rollout.py

assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
  "Model resolution doesn't match the data resolution. You likely want to "
  "re-filter the dataset list, and download the correct data.")

print("Inputs:  ", eval_inputs.dims.mapping) #time dimension input (2)
print("Targets: ", eval_targets.dims.mapping) #time dimension matches eval steps
print("Forcings:", eval_forcings.dims.mapping) #time dimension matches eval steps

if run:
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
        steady_dt = ds_dt)

    # save to a file
    #predictions.to_netcdf('/glade/work/hakim/data/ai-models/graphcast/output/graphcast_no_prcp.nc')
    #predictions.to_netcdf('/glade/work/hakim/data/ai-models/graphcast/output/graphcast_IVP.nc')
    #predictions.to_netcdf('/glade/work/hakim/data/ai-models/graphcast/output/graphcast_steady.nc')
    #predictions.to_netcdf('/glade/work/hakim/data/ai-models/graphcast/output/graphcast_one_step.nc')

    # save the dataset
    if graphcast_version == 'small':
        if perturb_ic:
            of = path_output+'graphcast_pw_DJF_IVP_small.nc'
        else:
            of = path_output+'graphcast_pw_DJF_mean_dt_small.nc'
    else:
        if perturb_ic:
            if geoadjust:
                of = path_output+'graphcast_pw_DJF_geoadjust_oper.nc'
#                of = path_output+'graphcast_pw_DJF_geoadjust_oper_windonly.nc'
            else:
                if zonal_mean:
                    of = path_output+'graphcast_pw_DJF_zm_IVP_oper.nc'
                else:
                    of = path_output+'graphcast_pw_DJF_IVP_oper.nc'
        else:
            if zonal_mean:
                of = path_output+'graphcast_pw_DJF_zm_mean_dt_oper.nc'
                #of = path_output+'graphcast_pw_DJF_zm_mean_dt_oper_check.nc'
            else:
                #of = path_output+'graphcast_pw_DJF_mean_dt_oper.nc'
                of = path_output+'graphcast_inference_oper_2022121500.nc'
            
    print('writing output here: ',of)
    predictions.to_netcdf(of)
