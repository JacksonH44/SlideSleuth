#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=4
#SBATCH --mem=5000M
#SBATCH --time=0-00:05
#SBATCH --job-name=analyze_autoencoder
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/analyze_autoencoder-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/analyze_autoencoder-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda/11.7
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../src/analyze_autoencoder.py
deactivate
rm -rf $ENVDIR