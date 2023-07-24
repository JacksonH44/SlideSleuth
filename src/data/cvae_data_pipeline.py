"""The data pipeline for the images fed into the Convolutional Variational Autoencoder

Date Created: July 12, 2023
Last Updated: July 12, 2023
"""

__author__ = "Jackson Howe"

from os import listdir, remove, environ, stat, makedirs
from os.path import join, isdir, exists
from shutil import move, rmtree

from PIL import ImageFile

BATCH_SIZE = 64
IMG_SIZE = 224
DIR_PATH = '../../data/processed/CK7/CK7_cvae'
ERR_FILE = join(DIR_PATH, 'error_log.txt')

ImageFile.LOAD_TRUNCATED_IMAGES = True
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _create_folder(folder):
  """A function that creates a folder if need be

  Args:
      folder (str): Path to the folder
  """
  if not exists(folder):
      makedirs(folder)

def organize_dir(root_dir):
  """A function that changes directory structure from:
  
  root_dir/
    001_files/
      5.0/
        img1.jpeg
        ...
        imgN.jpeg
    ...
    100_files/
      ...
    
  to:
  
  root_dir/
    folder/
      001-5.0-im1.jpeg
      ...
      100-5.0-imN.jpeg

  Args:
      root_dir (str): The path to the root directory you wish to be reorganized
  """
  
  # ImageDataGenerator requires a subfolder inside of each train, test, valid 
  # folder
  _create_folder(f"{join(root_dir, 'folder')}")
  for sub_dir in listdir(root_dir):
    
    # Make sure we don't include the subfolder when moving images
    if isdir(join(root_dir, sub_dir)) and not (sub_dir == 'folder'):
      for mag in listdir(join(root_dir, sub_dir)):
        for img in listdir(join(root_dir, sub_dir, mag)):
          new_img_name = f"{sub_dir.split('_')[0]}-{mag}-{img}"
          move(f'{join(root_dir, sub_dir, mag, img)}', f'{join(root_dir, "folder", new_img_name)}')
          
  # Go through and remove the other files from the directory that are not in #
  # use anymore (keep any .jpeg and the error log file)
  for file in listdir(root_dir):
    full_file = join(root_dir, file)
    if not (full_file == ERR_FILE or full_file.split(".")[-1] == "jpeg"):
      # If directory, remove the whole tree
      if isdir(full_file):
        rmtree(full_file)
      else:
        remove(full_file)

def clean_directory(clean_dir, log_file=ERR_FILE):
  """removes any corrupt images from a directory.

  Args:
      clean_dir (str): expects the directory in which the images are
  """
  
  print(f"cleaning {clean_dir}")
  for file in listdir(clean_dir):
    statfile = stat(join(clean_dir, file))
    filesize = statfile.st_size
    if filesize == 0 and not join(clean_dir, file) == log_file:
      with open(log_file, 'a') as err:
        err.write(f"{join(clean_dir, file)} removed because it was corrupt.\n")
      remove(join(clean_dir, file))
      
def preprocess_directory(root_path):
  """A function that cleans and reorganizes a directory for convolutional 
  variational autoencoder. Assumes a structure of:
  
  root_path/
    train/
    test/
    valid/
    
  With each of train, test, and valid having directory structure as assumed in 
  organize_root

  Args:
      root_path (str): The path to the root directory
  """
  train_path = join(root_path, 'train')
  valid_path = join(root_path, 'valid')
  test_path = join(root_path, 'test')
  
  organize_dir(train_path)
  organize_dir(test_path)
  organize_dir(valid_path)
  
  clean_directory(join(train_path, 'folder'))
  clean_directory(join(test_path, 'folder'))
  clean_directory(join(valid_path, 'folder'))

if __name__ == '__main__':
  dir_path = DIR_PATH
  preprocess_directory(dir_path)