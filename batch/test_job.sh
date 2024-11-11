#!/bin/bash

# Name of the job
#SBATCH --job-name=nengo_job

# Number of compute nodes
#SBATCH --nodes=1

# Number of cores, in this case one
#SBATCH --ntasks-per-node=1

# Walltime (job duration)
#SBATCH --time=00:20:00

cd ..
python run.py WM 1 0.8