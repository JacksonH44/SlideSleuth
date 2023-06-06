#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=3000M
#SBATCH --time=0-00:30
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/transfer_learning-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/transfer_learning-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda cudnn
module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index tensorflow
pip install --no-index openslide-python
python ../src/transfer_learning.py /scratch/jhowe4/results/2023-06-01/001_files/5.0/10_20.jpeg features.csv
deactivate
rm -rf $ENVDIR