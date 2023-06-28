#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=0-02:00
#SBATCH --job-name=vae
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/vae-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/vae-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load cuda/11.7 cudnn
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python ../src/vae.py
deactivate
rm -rf $ENVDIR