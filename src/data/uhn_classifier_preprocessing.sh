#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=0-16:00:00
#SBATCH --mem=32G
#SBATCH --job-name=uhn_classifier_preprocessing
#SBATCH --output=../../SLURM_DEFAULT_OUT/uhn_classifier_preprocessing-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/uhn_classifier_preprocessing-%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7 cudnn
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install -q --no-index pillow
python uhn_classifier_preprocessing.py
deactivate
rm -rf $ENVDIR