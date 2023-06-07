#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=4
#SBATCH --mem=256M
#SBATCH --time=00-00:05
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/filterScores-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/filterScores-%j.err

module load gcc r
module load r-bundle-bioconductor/3.16
module load openmpi

Rscript ../src/filterScores.R