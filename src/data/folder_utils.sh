#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --mem=65536M
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=folder_utils
#SBATCH --output=../../SLURM_DEFAULT_OUT/folder_utils-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/folder_utils-%j.err

rm -r ../../models/HNE_classifier*