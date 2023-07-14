#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=5000M
#SBATCH --time=0-00:40
#SBATCH --job-name=train_cvae
#SBATCH --output=../../outputs/SLURM_DEFAULT_OUT/train_cvae-%j.out
#SBATCH --error=../../outputs/SLURM_DEFAULT_OUT/train_cvae-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda/11.7
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../../src/unsupervised/train_cvae.py
deactivate
rm -rf $ENVDIR