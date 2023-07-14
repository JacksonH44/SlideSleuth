#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --job-name=cvae_data_pipeline
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/cvae_data_pipeline-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/cvae_data_pipeline-%j.err
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
python ../src/cvae_data_pipeline.py
deactivate
rm -rf $ENVDIR