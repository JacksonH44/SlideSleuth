#!/bin/bash

# A script that splits training and validation data into different directories, 
# mainly so that cases with multiple WSIs don't have one WSI in training and 
# one in validation if we just use a naive splitting function (i.e. 
# train_test_split). Assumes a 15% test split and 153 feature files
#
# Author: Jackson Howe
# Date Created: July 5, 2023
# Last Updated: July 5, 2023

features_dir="HNE_features"
train_dir="${features_dir}/train"
test_dir="${features_dir}/test"
output_dir=$train_dir

mkdir -p $train_dir
mkdir -p $test_dir

# Get file splitter
split_file=$(ls ${features_dir} | head -130 | tail -1)

for file in $(ls $features_dir); do
  if [ "${file}" == "${split_file}" ]; then
    output_dir=$test_dir
  fi

  # Move files
  if [ $file == $train_dir ]; then
    # do nothing
  elif [ $file == $test_dir ]; then
    # do nothing
  else 
    mv "${features_dir}/${file}" $output_dir
  fi
done