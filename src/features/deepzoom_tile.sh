#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=3000M
#SBATCH --time=0-00:10
#SBATCH --job-name=deepzoom_tile
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/deepzoom_tile-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/deepzoom_tile-%j.err

module load python/3.8.2
module load openslide/3.4.1
module load scipy-stack/2022a
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
pip install --no-index --upgrade pip
pip install --no-index openslide-python
# curDate=$(date '+%Y-%m-%d')
# directory="/scratch/jhowe4/results/$curDate/3"
# mkdir -p "$directory"
# python ../src/deepzoom_tile.py -s 229 -e 0 -j 32 -B 50 --output="$directory" ../inputs/HNE/003.svs
python ../src/deepzoom_tile.py -s 229 -e 0 -j 32 -B 50 --output="../outputs/HNE_53" ../inputs/raw/HNE/53.svs
deactivate
rm -rf $ENVDIR
