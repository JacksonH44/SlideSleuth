#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=0-04:00
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

input_folder="/home/jhowe4/projects/def-sushant/jhowe4/TissueTango/inputs/raw/CK7"
train_folder="/home/jhowe4/projects/def-sushant/jhowe4/TissueTango/outputs/CK7/train"
valid_folder="/home/jhowe4/projects/def-sushant/jhowe4/TissueTango/outputs/CK7/valid"
test_folder="/home/jhowe4/projects/def-sushant/jhowe4/TissueTango/outputs/CK7/test"

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
  python ../src/deepzoom_tile.py -s 224 -e 0 -j 32 -B 50 --output="$output_folder" "$input_folder/$file"
done

deactivate
rm -rf $ENVDIR
