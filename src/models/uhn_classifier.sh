#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gpus-per-node=p100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=65536M
#SBATCH --time=2-08:00:00
#SBATCH --job-name=uhn_classifier
#SBATCH --output=../../SLURM_DEFAULT_OUT/uhn_classifier-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/uhn_classifier-%j.err
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
pip install -q --no-index seaborn
python uhn_classifier.py
deactivate
rm -rf $ENVDIR