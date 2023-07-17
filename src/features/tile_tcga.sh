#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-24:00:00
#SBATCH --job-name=tile_tcga
#SBATCH --output=../outputs/SLURM_DEFAULT_OUT/tile_tcga-%j.out
#SBATCH --error=../outputs/SLURM_DEFAULT_OUT/tile_tcga-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install --no-index openslide-python

input_folder="/scratch/jhowe4/inputs/GDC/brca_example"
train_folder="../../data/external/brca_5x/train"
valid_folder="../../data/external/brca_5x/valid"
test_folder="../../data/external/brca_5x/test"

mkdir -p $train_folder
mkdir -p $valid_folder
mkdir -p $test_folder

validation_break="$(ls ${input_folder} | head -205 | tail -1)"
test_break="$(ls ${input_folder} | head -231 | tail -1)"

# loop through all image files in the input folder and call deepzoom
# on them
output_folder=$train_folder
for file in $(ls $input_folder); do
  if [ "$file" = "${validation_break}" ]; then
    output_folder=$valid_folder
  elif [ "$file" = "${test_break}" ]; then
    output_folder=$test_folder
  fi
  python deepzoom_tile.py -s 224 -e 0 -j 32 -B 50 --output="$output_folder" "$input_folder/$file"
done

deactivate
rm -rf $ENVDIR