'''
  A program that trains a user-built autoencoder

  Author: YouTube Channel: Valerio Velardo, with edits by Jackson Howe
  link: https://www.youtube.com/watch?v=6fZdJKm-fSk&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=6

  Date Created: June 8, 2023
  Last Updated: July 10, 2023
'''

from cvae import CVAE
from tensorflow.keras.utils import image_dataset_from_directory
import tensorflow as tf
import os
from os.path import join
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LEARNING_RATE = 0.0005
BATCH_SIZE = 128
EPOCHS = 100
IMG_SIZE = 224

LATENT_SPACE_DIM = 2
NUM_TRAINING_TILES = None
NUM_VALID_TILES = None

STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / BATCH_SIZE)

def train(X_train, learning_rate, batch_size, epochs, steps_per_epoch, validation_data, validation_steps):
  """A function that trains a convolutional variational autoencoder

  Args:
      x_train (np.array): Input and output data (since it is an autoencoder)

  Returns:
      tf.keras.History: A trained CVAE history
  """
  # YOU must specify the model parameters
  vae = CVAE(
      input_shape=(224, 224, 3),
      conv_filters=(32, 64, 64, 64),
      conv_kernels=(3, 3, 3, 3),
      conv_strides=(1, 2, 2, 1),
      latent_space_dim=LATENT_SPACE_DIM
    )
  
  # Train model
  vae.summary()
  vae.compile(learning_rate)
  history = vae.train(X_train, batch_size, epochs, validation_data)
  return history

def make_dataset(dir_path, img_size=(IMG_SIZE, IMG_SIZE), shuffle=True):
  """A function that makes a dataset suitable for the CVAE from a directory 
  given.

  Args:
      dir_path (str): The path to the required directory
  """
  return image_dataset_from_directory(
    dir_path,
    labels=None,
    label_mode=None,
    batch_size=None,
    image_size=img_size,
    shuffle=shuffle
  )

if __name__ == '__main__':
  # Load dataset
  ds_path = '../outputs/CK7'
  test_path = join(ds_path, 'test')
  train_path = join(ds_path, 'train')
  valid_path = join(ds_path, 'valid')
  
  test_ds = make_dataset(test_path)
  train_ds = make_dataset(train_path)
  valid_ds = make_dataset(valid_path)
  
  history = train(
    train_ds, 
    LEARNING_RATE, 
    BATCH_SIZE, 
    EPOCHS, 
    STEPS_PER_EPOCH, 
    valid_ds, 
    VALIDATION_STEPS
  )