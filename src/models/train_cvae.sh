#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=81920M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=train_cvae
#SBATCH --output=../../SLURM_DEFAULT_OUT/train_cvae-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/train_cvae-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda/11.7 cudnn
source $ENVDIR/bin/activate
pip install -q --no-index --upgrade pip
pip install -q --no-index tensorflow
python train_cvae.py
deactivate
rm -rf $ENVDIR