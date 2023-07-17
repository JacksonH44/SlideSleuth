#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=4
#SBATCH --mem=100M
#SBATCH --time=0-00:03
#SBATCH --job-name=smooth_labels
#SBATCH --output=../../outputs/SLURM_DEFAULT_OUT/smooth_labels-%j.out
#SBATCH --error=../../outputs/SLURM_DEFAULT_OUT/smooth_labels-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index openpyxl
python ../../src/supervised/smooth_labels.py
deactivate
rm -rf $ENVDIR