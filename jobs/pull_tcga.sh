#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=1-00:0
#SBATCH -p himem
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/pull_tcga-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/pull_tcga-%j.err

module load gcc r
module load r-bundle-bioconductor/3.16
module load openmpi

Rscript download_luad.R
Rscript download_lusc.R