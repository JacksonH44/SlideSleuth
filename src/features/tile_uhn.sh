#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=65536M
#SBATCH --time=0-16:00:00
#SBATCH --job-name=tile_uhn
#SBATCH --output=../../SLURM_DEFAULT_OUT/tile_uhn-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/tile_uhn-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index openslide-python

input_folder="../../data/raw/HNE"
train_folder="../../data/interim/HNE/train"
valid_folder="../../data/interim/HNE/valid"
test_folder="../../data/interim/HNE/test"

mkdir -p $train_folder
mkdir -p $valid_folder
mkdir -p $test_folder

# loop through all image files in the input folder and call deepzoom
# on them
output_folder=$train_folder
for file in $(ls $input_folder); do
  if [ "$file" = "82.svs" ]; then
    output_folder=$valid_folder
  elif [ "$file" = "92.svs" ]; then
    output_folder=$test_folder
  fi
  python deepzoom_tile.py -s 224 -e 0 -j 32 -B 50 --output="$output_folder" "$input_folder/$file"
done

deactivate
rm -rf $ENVDIR
