#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --job-name=rm_file
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/rm_file-%j.out

rm ../outputs/SLURM_DEFAULT_OUT/data_exploration-7666581.err