#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-10:00:00
#SBATCH --mem=64G
#SBATCH --job-name=data_exploration
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/data_exploration-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/data_exploration-%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7 cudnn
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index tensorflow
pip install --no-index pillow
pip install --no-index scikit-learn
python ../src/data_exploration.py
deactivate
rm -rf $ENVDIR