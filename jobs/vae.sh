#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=0-16:00:00
#SBATCH --job-name=vae
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/vae-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/vae-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

source $ENVDIR/bin/activate
module load scipy-stack
module load cuda/11.7 cudnn
pip install -q -U --no-index tensorflow
python ../src/vae.py
deactivate
rm -rf $ENVDIR