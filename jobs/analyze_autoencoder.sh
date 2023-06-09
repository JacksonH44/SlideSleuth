#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=5000M
#SBATCH --time=0-00:20
#SBATCH --job-name=train_autoencoder
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/train_autoencoder-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/train_autoencoder-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda/11.7
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../src/train_autoencoder.py
deactivate
rm -rf $ENVDIR