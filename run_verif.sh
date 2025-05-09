#!/bin/bash -l
#PBS -N verification
#PBS -A NMMM0015
#PBS -l select=1:ncpus=1:ompthreads=4:mem=32GB
#PBS -l walltime=02:15:00
#PBS -q casper
#PBS -j oe

module load conda
conda activate shtools
python /glade/u/home/hakim/gitwork/graphcast/graphcast_verification.py
