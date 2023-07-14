"""The data pipeline for the images fed into the Convolutional Variational Autoencoder

Author: Jackson Howe
Date Created: July 12, 2023
Last Updated: July 12, 2023
"""

import tensorflow as tf
from PIL import ImageFile
from os import listdir, remove, environ, stat
from os.path import join, isdir
from shutil import move, rmtree

## Global variables
BATCH_SIZE = 64
IMG_SIZE = 224
ERR_FILE = '../outputs/HNE/train/corrupt_images.txt'
AUTOTUNE = tf.data.AUTOTUNE

ImageFile.LOAD_TRUNCATED_IMAGES = True
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def organize_root(root_dir):
  for sub_dir in listdir(root_dir):
    if isdir(join(root_dir, sub_dir)):
      for mag in listdir(join(root_dir, sub_dir)):
        for img in listdir(join(root_dir, sub_dir, mag)):
          new_img_name = f"{sub_dir.split('_')[0]}-{mag}-{img}"
          move(f'{join(root_dir, sub_dir, mag, img)}', f'{join(root_dir, new_img_name)}')
          
  # Go through and remove the other files from the directory that are not in use anymore
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
  
  for file in listdir(clean_dir):
    statfile = stat(join(clean_dir, file))
    filesize = statfile.st_size
    if filesize == 0 and not join(clean_dir, file) == log_file:
      with open(log_file, 'a') as err:
        err.write(f"{join(clean_dir, file)} removed because it was corrupt.")
      remove(join(clean_dir, file))
        
def clean_dataset_from_root(clean_root):
  """cleans a whole dataset of corrupt images. Expects a directory structure of:
  
  root_dir/
    001_files/
      5.0/
        10_10.jpeg
        10_11.jpeg
        ...
      10.0/
        ...
    002A_files/
      ...
    98C_files/
      ...

  Args:
      clean_root (str): expects the root of the directory 
  """
  
  for file in listdir(clean_root):
    if isdir(join(clean_root, file)):
      for mag in listdir(join(clean_root, file)):
        print(f"Cleaning {join(clean_root, file, mag)}")
        clean_directory(join(clean_root, file, mag))
  
def make_dataset(dir_path, img_size=(IMG_SIZE, IMG_SIZE), shuffle=True):
  """makes the dataset for a directory of images. Assumes the directory has already been cleaned, or else the error: 'InvalidArgumentError: Input is empty.' may pop up

  Returns:
      tf.data.Dataset: a tensorflow dataset object representing all images in the directory
  """
  
  datagen = tf.keras.preprocessing.image.ImageDataGenerator()
  return datagen.flow_from_directory(
    dir_path,
    target_size=img_size,
    classes=None,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=shuffle
  )

if __name__ == '__main__':
  dir_path = '../outputs/HNE'
  dataset = make_dataset(dir_path)