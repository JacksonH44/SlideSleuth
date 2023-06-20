#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=0-04:00
#SBATCH --job-name=tile_tcga
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

input_folder="/scratch/jhowe4/inputs/GDC/paad_example"
train_folder="/scratch/jhowe4/outputs/GDC/paad_example/train"
valid_folder="/scratch/jhowe4/outputs/GDC/paad_example/valid"
test_folder="/scratch/jhowe4/outputs/GDC/paad_example/test"

mkdir -p $train_folder
mkdir -p $valid_folder
mkdir -p $test_folder

validation_break="$(ls ${input_folder} | head -207 | tail -1)"
test_break="$(ls ${input_folder} | head -230 | tail -1)"

# loop through all image files in the input folder and call deepzoom
# on them
output_folder=$train_folder
for file in $(ls $input_folder); do
  if [ "$file" = "${validation_break}" ]; then
    output_folder=$valid_folder
  elif [ "$file" = "${test_break}" ]; then
    output_folder=$test_folder
  fi
  python ../src/deepzoom_tile.py -s 224 -e 0 -j 32 -B 50 --output="$output_folder" "$input_folder/$file"
done

deactivate
rm -rf $ENVDIR