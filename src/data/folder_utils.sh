#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --mem=102400M
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=folder_utils
#SBATCH --output=../../SLURM_DEFAULT_OUT/folder_utils-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/folder_utils-%j.err

cp -r ../../data/processed/CK7/CK7_cvae ../../../../darah_jackson_data_transfer/processed_data/CK7/CK7_cvae