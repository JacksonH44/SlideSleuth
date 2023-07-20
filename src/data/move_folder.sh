#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --mem=65536M
#SBATCH --time=0-08:00:00
#SBATCH --nodes
#SBATCH --ntasks-per-node=32
#SBATCH --job-name=move_folder
#SBATCH --output=../../SLURM_DEFAULT_OUT/move_folder-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/move_folder-%j.err

mv ../../data/interim/HNE_images ../../data/processed/HNE