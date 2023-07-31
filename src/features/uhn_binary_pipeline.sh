#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-16:00:00
#SBATCH --mem=65536M
#SBATCH --job-name=uhn_binary_pipeline
#SBATCH --output=../../SLURM_DEFAULT_OUT/uhn_binary_pipeline-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/uhn_binary_pipeline-%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7 cudnn
module load scipy-stack
source $ENVDIR/bin/activate
python uhn_binary_pipeline.py
deactivate
rm -rf $ENVDIR