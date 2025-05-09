#!/bin/bash -l
#PBS -N graphcast
#PBS -A NMMM0015
#PBS -l select=1:ncpus=1:mpiprocs=1:ngpus=1:mem=3GB
#PBS -l gpu_type=v100
#PBS -l walltime=00:15:00
#PBS -q casper
#PBS -j oe

### Load newest CUDA version
module load cuda/11.8

### Load your Earth2MIP conda library
module load conda
conda activate graphcast

### Debugging
#nvidia-smi
#module list
#python -c "import torch; print(torch.cuda.is_available())"

### Run inference (variable emem sent by calling script)
python /glade/u/home/hakim/gitwork/graphcast/graphcast_minimal.py
