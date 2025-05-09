#!/bin/bash -l
#PBS -N pw_gc_ic
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
#conda activate earth2mip
conda activate dlwd

### Debugging
#nvidia-smi
#module list
#python -c "import torch; print(torch.cuda.is_available())"

python /glade/u/home/hakim/gitwork/graphcast/run_pangu_gc_ic.py
