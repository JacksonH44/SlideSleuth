#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=0-02:30
#SBATCH --job-name=tile_folder
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/tile_folder-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/tile_folder-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index openslide-python

input_folder="../inputs/raw/HNE"
output_folder="../outputs/HNE"

mkdir -p $output_folder

# loop through all image files in the input folder and call deepzoom
# on them
for file in $(ls $input_folder); do
  python ../src/deepzoom_tile.py -s 229 -e 0 -j 32 -B 50 --output="$output_folder" "../inputs/raw/HNE/$file"
done

deactivate
rm -rf $ENVDIR
