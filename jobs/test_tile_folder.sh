#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=0-01:00
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/tile_folder-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/tile_folder-%j.err

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load cuda cudnn
module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index tensorflow
pip install --no-index openslide-python
python ../src/tile_folder.py -s 229 -e 0 -j 32 -B 50 --output=/scratch/jhowe4/data/2023-06-02/1 ../test/test_inputs
deactivate
rm -rf $ENVDIR
