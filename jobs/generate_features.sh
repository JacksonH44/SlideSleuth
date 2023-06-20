#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=12000M
#SBATCH --time=0-02:30
#SBATCH --job-name=generate_features
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/generate_features-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/generate_features-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda cudnn
module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index tensorflow
pip install --no-index openslide-python
python ../src/generate_features.py ../outputs/HNE ../outputs/HNE_features
deactivate
rm -rf $ENVDIR