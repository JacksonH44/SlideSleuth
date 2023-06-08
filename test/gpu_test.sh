#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3000M
#SBATCH --time=0-00:10
#SBATCH --job-name=gpu_test
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/gpu_test-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/gpu_test-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda cudnn
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python gpu_test.py
deactivate
rm -rf $ENVDIR