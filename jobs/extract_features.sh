#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=5000M
#SBATCH --time=0-00:10
#SBATCH --job-name=extract_features
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/extract_features-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/extract_features-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda/11.7
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../src/extract_features.py ../outputs/HNE/001_files/5.0/10_20.jpeg ../outputs/features.csv
deactivate
rm -rf $ENVDIR