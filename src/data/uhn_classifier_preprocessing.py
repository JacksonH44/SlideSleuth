"""A program that preprocesses UHN slide image data into a format that can be used as input for a Tensorflow Dataset or Generator object.
  
Date Created: July 20, 2023
Last Updated: July 20, 2023
"""
  
__author__ = "Jackson Howe"

from os.path import join, exists, isdir
from os import listdir, remove, stat, makedirs
from shutil import move, rmtree

import pandas as pd

DIR_PATH = "../../data/interim/HNE"
LABEL = join(DIR_PATH, "labels.csv")
ERR_FILE = join(DIR_PATH, "error_log.txt")

def _create_folder(dir_path):
  """Private helper function that creates a directory if it doesn't already exist."""
  
  if not exists(dir_path):
    makedirs(dir_path)
    
def _categorize_labels(label):
  """Private helper function to group raw label values into categories.

  Args:
      label (float): The raw label value between 0 and 1 inclusive.

  Returns:
      str: One of three categories: "invasive", "noninvasive", or "undefined" based on the raw label value.
  """
  
  if label <= 0.40:
    return 'invasive'
  elif label >= 0.60:
    return 'noninvasive'
  else:
    return 'undefined'
    
def organize_dir(dir_path):
  """A function that changes the directory structure from:
  
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
  |-- case1-5.0-10_10.jpeg
  |-- case1-5.0-10_10.jpeg
  ...
  |-- case1-20.0-10_20.jpeg
  ...
  |-- caseN-5.0-10_10.jpeg

  Args:
      dir_path (str): The path to the directory you wish to reorganize.
  """
  
  # Read in the labels from the csv file.
  label_df = pd.read_csv(LABEL)
  
  # Create the necessary folders for each category if they don't already exist.
  invasive_path = join(dir_path, 'invasive')
  noninvasive_path = join(dir_path, 'noninvasive')
  undefined_path = join(dir_path, 'undefined')
  category_paths = [invasive_path, noninvasive_path, undefined_path]
  _create_folder(invasive_path)
  _create_folder(noninvasive_path)
  _create_folder(undefined_path)
  
  # Find all directories that contain images
  for file in listdir(dir_path):
    if isdir(join(dir_path, file)) and not join(dir_path, file) in category_paths:
      print(f"Moving {file}.")
      
      # Extract the raw label value from the data frame that corresponds to the 
      # current file.
      label = label_df[label_df['case'] == f"{file.split('_')[0].lstrip('0')}"]['score'].values[0]
      label = _categorize_labels(label)
      for mag in listdir(join(dir_path, file)):
        for img in listdir(join(dir_path, file, mag)):
          move(f"{join(dir_path, file, mag, img)}", f"{join(dir_path, label, file.split('_')[0] + '-' + mag + '-' + img)}")
    elif join(dir_path, file) not in category_paths:
      remove(join(dir_path, file))
      
  for file in listdir(dir_path):
    if isdir(join(dir_path, file)):
      if not join(dir_path, file) in category_paths:
        rmtree(join(dir_path, file))
          
def clean_directory(clean_dir):
  """removes any corrupt images from a directory.

  Args:
      clean_dir (str): expects the directory in which the images are
  """
  
  for file in listdir(clean_dir):
    statfile = stat(join(clean_dir, file))
    filesize = statfile.st_size
    if filesize == 0 and not join(clean_dir, file) == ERR_FILE:
      with open(ERR_FILE, 'a') as err:
        err.write(f"{join(clean_dir, file)} removed because it was corrupt.\n")
      remove(join(clean_dir, file))
      
def preprocess(root_dir):
  """A program that preprocesses data that is in the output format of tile_uhn into the input format of tf.keras.preprocessing.image.ImageDataGenerator.flow_from_directory function.

  Args:
      root_dir (str): The path to the root directory of the data, the directory which contains "train", "valid", and "test" subdirectories.
  """
  
  train_path = join(root_dir, "train")
  test_path = join(root_dir, "test")
  valid_path = join(root_dir, "valid")
  
  organize_dir(train_path)
  organize_dir(test_path)
  organize_dir(valid_path)
  
  clean_directory(train_path)
  clean_directory(test_path)
  clean_directory(valid_path)
  
if __name__ == '__main__':
  preprocess(DIR_PATH)