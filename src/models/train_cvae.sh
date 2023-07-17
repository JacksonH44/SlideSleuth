#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:4
#SBATCH --ntasks-per-node=32
#SBATCH --mem=65536M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=train_cvae
#SBATCH --output=../../outputs/SLURM_DEFAULT_OUT/train_cvae-%j.out
#SBATCH --error=../../outputs/SLURM_DEFAULT_OUT/train_cvae-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda/11.7 cudnn
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../../src/unsupervised/train_cvae.py
deactivate
rm -rf $ENVDIR