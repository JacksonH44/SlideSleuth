"""
  A helper program to generate a dataset in eager execution mode, then save and pickle the dataset to be used in the analyze_cvae module.
  
  Date Created: August 3, 2023
  Last Updated: August 3, 2023
"""

__author__ = 'Jackson Howe'

import pickle

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 32
IMG_SIZE = 224
DATA_PATH = '../../data/processed/CK7/CK7_cvae/test'
SAVE_PATH = '../../data/processed/datasets/cvae_dataset.pkl'

def select_images(images, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    return sample_images

def make_dataset(dir_path, img_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE):
  # Create data generator
  datagen = ImageDataGenerator(
    rescale=1./255
  )
  
  return datagen.flow_from_directory(
    dir_path,
    target_size=img_size,
    class_mode='input',
    batch_size=batch_size
  )

if __name__ == "__main__":
  # Load CK7 CVAE test dataset.
  test_ds = make_dataset(DATA_PATH)
  
  # Sample images
  num_sample_show = 8
  sample_images = None
  
  img_batch = [x for _, x in zip(range(1), test_ds)]
  img_batch = np.array(img_batch)
  img_batch = img_batch[0][0]
  sample_images = select_images(img_batch, num_sample_show)

  # Save datasets
  with open(SAVE_PATH, 'wb') as file:
    pickle.dump(sample_images, file)