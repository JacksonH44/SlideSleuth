"""A program that organizes all images of the UHN dataset due to a predefined split: 70% training, 15% validation, 15% testing

Date Created: July 31, 2023
Last Updated: July 31, 2023
"""

__author__ = 'Jackson Howe'

from os import makedirs, listdir, remove
from os.path import exists, isdir, join
import string
import sys
from shutil import move

sys.path.insert(0, '/home/jhowe4/projects/def-sushant/jhowe4/SlideSleuth/src/data')

import pandas as pd

from cvae_data_pipeline import clean_directory

LABELS = '../../data/interim/uhn_labels.csv'
INPUT_DIR = '/scratch/jhowe4/outputs/uhn/CK7'
OUTPUT_DIR = '/scratch/jhowe4/outputs/uhn/CK7_images'

def _create_dir(dir_path):
  if not exists(dir_path):
    makedirs(dir_path)
    
def _binarize_class(class_name):
  if class_name == 'acinar':
    return 1
  else:
    return 0
    
def organize_dataset(labels, input_dir, output_dir):
  """A function that organizes a dataset from the structure from:
  
  root
  |
  |-- case1_files
  |    |-- 5.0
  |    |   |-- 10_10.jpeg
  |    |   |-- 10_11.jpeg
  ...
  |    |-- 20.0
  |-- case1.dzi
  ...
  |-- caseN_files
  |    |-- 5.0
  |    |    |--10_10.jpeg
  |    |    |--10_11.jpeg
  ...
  |-- caseN.dzi
  
  to:
  
  root
  |
  |-- train
  |-- |-- case1-5.0-10_10.jpeg
  |-- |-- case1-5.0-10_10.jpeg
  ...
  |-- |-- case1-20.0-10_20.jpeg
  ...
  |-- |-- caseP-5.0-10_10.jpeg
  |-- valid
  |-- ...
  |-- test
  |-- ...

  Args:
      labels (str): Path to the csv containing the case labels.
      input_dir (str): Path to the directory where the raw tiled files are.
      output_dir (str): Path to the intended target directory where the dataset will be
  """
  
  # Create the output directory as well as all of the subdirectories.
  train_dir = join(output_dir, 'train')
  valid_dir = join(output_dir, 'valid')
  test_dir = join(output_dir, 'test')
  _create_dir(join(train_dir, '0'))
  _create_dir(join(train_dir, '1'))
  _create_dir(join(valid_dir, '0'))
  _create_dir(join(valid_dir, '1')) 
  _create_dir(join(test_dir, '0'))
  _create_dir(join(test_dir, '1'))
  
  # Each of the lepidic and acinar types have 54 cases, so defined a manual split for 70% training, 15% validation, 15% testing.
  num_training = 38
  num_validation = 8
  num_testing = 8
  label_col = (['train']*num_training + ['valid']*num_validation + ['test']*num_testing) * 2
  
  # Read in the labels file into a dataframe.
  labels_df = pd.read_csv(labels)
  labels_df['split'] = label_col
  labels_df = labels_df.rename(columns={'0': 'case', '1': 'class'})
  print(labels_df)
  
  # Loop through all files containing tiles in the input directory and find their corresponding label.
  for file in listdir(input_dir):
    if isdir(join(input_dir, file)):
      # Find both the case label and the split (training, testing, validation) the case belongs to.
      case_id = file.split('_')[0].lstrip('0').rstrip(string.ascii_letters + string.whitespace)
      case_label = _binarize_class(labels_df[labels_df['case'] == int(case_id)]['class'].values[0])
      case_set = labels_df[labels_df['case'] == int(case_id)]['split'].values[0]
      
      for mag in listdir(join(input_dir, file)):
        for img_file in listdir(join(input_dir, file, mag)):
          move(f'{join(input_dir, file, mag, img_file)}', f'{join(output_dir, case_set, str(case_label), file.split("_")[0] + "-" + mag + "-" + img_file)}')
    else:
      # Remove .dzi files from directory
      remove(f'{join(input_dir, file)}')
  

if __name__ == '__main__':
  organize_dataset(LABELS, INPUT_DIR, OUTPUT_DIR)