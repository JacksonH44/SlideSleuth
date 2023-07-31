#!/bin/bash
#SBATCH --account=def-sushant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=81920M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=tile_uhn_binary
#SBATCH --output=../../SLURM_DEFAULT_OUT/tile_uhn_binary-%j.out
#SBATCH --error=../../SLURM_DEFAULT_OUT/tile_uhn_binary-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=j2howe@uwaterloo.ca

module load python
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR

module load scipy-stack
module load openslide
source $ENVDIR/bin/activate
pip install -q -U --no-index openslide-python

input_folder="../../data/raw/CK7"
output_folder="/scratch/jhowe4/outputs/uhn/CK7"

mkdir -p $output_folder

# loop through all image files in the input folder and call deepzoom
# on them
for file in $(ls $input_folder); do
  python deepzoom_tile.py -s 224 -e 0 -j 32 -B 50 --output="$output_folder" "$input_folder/$file"
done

deactivate
rm -rf $ENVDIR