#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH -t 0-00:30
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=test_deepzoom
#SBATCH --output=test_deepzoom_%j.out
#SBATCH --error=test_deepzoom_%j.err

module load python/3.8.2
module load openslide/3.4.1
module load scipy-stack/2022a
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index openslide-python
pip freeze --local > requirements.txt
curDate=$(date '+%Y-%m-%d')
directory="/scratch/jhowe4/results/$curDate/3"
mkdir -p "$directory"
python ../src/deepzoom_tile.py -s 229 -e 0 -j 32 -B 50 --output="$directory" 001.svs
deactivate
rm -rf $ENVDIR
