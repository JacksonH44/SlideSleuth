#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=4
#SBATCH --mem=4000M
#SBATCH --time=0-00:30
#SBATCH --job-name=keras_tuner_test
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/keras_tuner_test-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/keras_tuner_test-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda cudnn
module load scipy-stack
source $ENVDIR/bin/activate
pip install --no-index tensorflow
python keras_tuner_test.py
deactivate
rm -rf $ENVDIR