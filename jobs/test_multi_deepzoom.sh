#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20000M
#SBATCH --time=0-03:00
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/test_multi_deepzoom_%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/test_multi_deepzoom_%j.err

module load python/3.8.2
module load openslide/3.4.1
module load scipy-stack/2022a
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index openslide-python
curDate=$(date '+%Y-%m-%d')
directory="/scratch/jhowe4/results/$curDate/4"
mkdir -p "$directory"
python ../src/deepzoom_tile.py -s 229 -e 0 -j 32 -B 50 --output="$directory" ../inputs/HNE
deactivate
rm -rf $ENVDIR