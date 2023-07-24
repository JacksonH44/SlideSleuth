#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=0-00:10:00
#SBATCH --output=../../SLURM_DEFAULT_OUT/downloadPaad-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/downloadPaad-%j.err

module load gcc r
module load r-bundle-bioconductor/3.16
module load openmpi

Rscript downloadPaad.R