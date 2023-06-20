#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=5000M
#SBATCH --time=0-00:10
#SBATCH --job-name=cvae
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/cvae-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/cvae-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda cudnn
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../src/cvae.py
deactivate
rm -rf $ENVDIR