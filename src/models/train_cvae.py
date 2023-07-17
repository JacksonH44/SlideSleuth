'''
  A program that trains a user-built autoencoder

  Author: YouTube Channel: Valerio Velardo, with edits by Jackson Howe
  link: https://www.youtube.com/watch?v=6fZdJKm-fSk&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=6

  Date Created: June 8, 2023
  Last Updated: July 14, 2023
'''

from cvae import CVAE
from train_vae import plot_loss

import tensorflow as tf
import math

from os.path import join

LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10
IMG_SIZE = 224

LATENT_SPACE_DIM = 200
NUM_TRAINING_TILES = 121277
NUM_VALID_TILES = 18508

STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / BATCH_SIZE)

DIR_PATH = '/scratch/jhowe4/outputs/uhn/CK7'
SAVE_PATH = '../../model/cvae-2023-07-17'

def make_dataset(dir_path, img_size=(IMG_SIZE, IMG_SIZE), shuffle=True):
  """makes the dataset for a directory of images. Assumes the directory has already been cleaned, or else the error: 'InvalidArgumentError: Input is empty.' may pop up

  Returns:
      tf.data.Dataset: a tensorflow dataset object representing all images in the directory
  """
  
  datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255.0
  )
  
  return datagen.flow_from_directory(
    dir_path,
    target_size=img_size,
    class_mode='input',
    batch_size=BATCH_SIZE,
    shuffle=shuffle
  )

if __name__ == '__main__':
  # Load dataset
  train_path = join(DIR_PATH, 'train')
  valid_path = join(DIR_PATH, 'valid')
  train_ds= make_dataset(train_path)
  valid_ds= make_dataset(valid_path)
  
  # Train on multiple GPUs
  strategy = tf.distribute.MirroredStrategy()
  
  # Check the number of GPUs
  print(f"Number of devices: {strategy.num_replicas_in_sync}")
  
  with strategy.scope():
  
    # You must specify the model parameters
    vae = CVAE(
      input_shape=(224, 224, 3),
      conv_filters=(32, 64, 64, 64),
      conv_kernels=(3, 3, 3, 3),
      conv_strides=(1, 2, 2, 1),
      latent_space_dim=LATENT_SPACE_DIM
    )
  
    # Train model
    vae.summary()
    vae.compile(LEARNING_RATE)
      
    # Train the model
    history = vae.train(
      train_ds,  
      num_epochs=EPOCHS, 
      steps_per_epoch=STEPS_PER_EPOCH,
      validation_data=valid_ds, 
      validation_steps=VALIDATION_STEPS,
      cp_path=join(SAVE_PATH, 'checkpoint')
    )
  
    # Save the model
    vae.save(SAVE_PATH)
  
  # Plot loss history
  plot_loss(
    history, 
    'calculate_reconstruction_loss',
    '../../img/cvae_reconstruction_loss.png'
  )
  
  plot_loss(
    history, 
    '_calculate_kl_loss',
    '../../img/cvae_kl_loss.png'
  )
  
  plot_loss(
    history, 
    'loss',
    '../../img/cvae_loss.png'
  )