#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=3-00:00:00
#SBATCH --mem=80G
#SBATCH --job-name=transfer_learning
#SBATCH --output=../../outputs/SLURM_DEFAULT_OUT/transfer_learning-%j.out
#SBATCH --error=../../outputs/SLURM_DEFAULT_OUT/transfer_learning-%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7 cudnn
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install -q --no-index tensorflow
pip install -q --no-index pillow
pip install -q --no-index scikit-learn
python ../../src/supervised/transfer_learning.py
deactivate
rm -rf $ENVDIR