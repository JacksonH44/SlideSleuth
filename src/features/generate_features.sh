#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=36G
#SBATCH --time=0-16:00
#SBATCH --nodes=3
#SBATCH --job-name=generate_features
#SBATCH --output=../../SLURM_DEFAULT_OUT/generate_features-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/generate_features-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda cudnn
module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install -q --no-index tensorflow
pip install -q --no-index openslide-python
python generate_features.py ../../data/interim/HNE ../../data/processed/HNE_features
deactivate
rm -rf $ENVDIR