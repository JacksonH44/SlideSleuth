#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --time=2-00:00:00
#SBATCH --mem=60G
#SBATCH --job-name=transfer_learning_small
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/transfer_learning_small-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/transfer_learning_small-%j.err
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
pip install -q -U --no-index scikit-learn
python ../src/transfer_learning_small.py
deactivate
rm -rf $ENVDIR