#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-14:30:00
#SBATCH --mem=70G
#SBATCH --job-name=tune_hp
#SBATCH --output=../../outputs/SLURM_DEFAULT_OUT/tune_hp-%j.out
#SBATCH --error=../../outputs/SLURM_DEFAULT_OUT/tune_hp-%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7 cudnn
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index tensorflow
pip install --no-index scikit-learn
python ../../src/supervised/tune_hp.py
deactivate
rm -rf $ENVDIR