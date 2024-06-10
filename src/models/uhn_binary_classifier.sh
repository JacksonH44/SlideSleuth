#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --gpus-per-node=p100:2
#SBATCH --ntasks-per-node=32
#SBATCH --mem=81920M
#SBATCH --time=3-00:00:00
#SBATCH --job-name=uhn_binary_classifier
#SBATCH --output=../../SLURM_DEFAULT_OUT/uhn_binary_classifier-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/uhn_binary_classifier-%j.err
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

python uhn_binary_classifier.py

deactivate
rm -rf $ENVDIR