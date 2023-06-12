#!/bin/bash
# This script requires 2 positional arguments, the path to the lowest directory
# that holds all images ($1), and the path to the desired output directory ($2)

# We assume that the directory provided as an argument has n subdirectories, 
# each with 1 image in it, for a total of n images

# Make output directory if it doesn't exist
mkdir -p $2

for dir in $(ls $1); do
  file=$(ls "$1/$dir")
  mv $1/$dir/$file $2
done