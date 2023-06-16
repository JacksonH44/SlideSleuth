#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22000M
#SBATCH --time=0-04:00
#SBATCH --job-name=transfer_learning
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/transfer_learning-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/transfer_learning-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index tensorflow
pip install --no-index sklearn
python ../src/transfer_learning.py
deactivate
rm -rf $ENVDIR