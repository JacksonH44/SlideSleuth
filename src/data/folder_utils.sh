#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --mem=102400M
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=folder_utils
#SBATCH --output=../../SLURM_DEFAULT_OUT/folder_utils-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/folder_utils-%j.err

rm -r ../../data/interim/CK7