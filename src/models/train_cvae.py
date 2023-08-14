"""A program that trains a deep convolutional autoencoder

  Author: YouTube Channel: Valerio Velardo, with edits by Jackson Howe
  link: https://www.youtube.com/watch?v=6fZdJKm-fSk&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=6

  Date Created: June 8, 2023
  Last Updated: July 21, 2023
"""

import math
from os.path import join, exists
from os import makedirs
from datetime import datetime
import pickle

import tensorflow as tf

from cvae import CVAE
from train_vae import plot_loss

LEARNING_RATE = 1e-6
BATCH_SIZE = 16
EPOCHS = 50
IMG_SIZE = 224

LATENT_SPACE_DIM = 2000
NUM_TRAINING_TILES = 121277
NUM_VALID_TILES = 18508

STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / BATCH_SIZE)

AUTOTUNE = tf.data.AUTOTUNE

unique_id = datetime.now().strftime('%Y%m%d-%H%M%S')

DIR_PATH = "../../data/processed/CK7/CK7_cvae"
SAVE_PATH = f"../../models/cvae-{unique_id}"
FIG_PATH = f"../../reports/figures/cvae-{unique_id}"

def change_inputs(images, _):
  """A function that transforms a dataset from an image-label pair to an image-image pair for autoencoder training

  Args:
      iages, labels (tf.data.Dataset): A dataset consisting of image-label pairs
  """
  
  return images, images

def make_dataset(dir_path, training, batch_size=BATCH_SIZE, img_size=(IMG_SIZE, IMG_SIZE), shuffle=True):
  """makes the dataset for a directory of images. Assumes the directory has already been cleaned, or else the error: 'InvalidArgumentError: Input is empty.' may pop up

  Returns:
      tf.data.Dataset: a tensorflow dataset object representing all images in the directory
  """
  
  if training:
    # Augment the training data.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255.0,
      horizontal_flip=True,
      rotation_range=20,
      shear_range=0.2,
      width_shift_range=0.1
    )
  else:
    # Do not augment the testing or validation data.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255.0
    )
  
  return datagen.flow_from_directory(
    dir_path,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='input',
    batch_size=batch_size,
    shuffle=shuffle
  )
  
def change_inputs(image, _):
    return image, image

def normalize(image, mirrored_image):
    return (tf.cast(image, tf.float32) / 255.0, tf.cast(mirrored_image, tf.float32) / 255.0)
  
def data_pipeline(dir_path, batch_size=BATCH_SIZE, img_size=(IMG_SIZE, IMG_SIZE), shuffle=True):
  
  dataset = tf.keras.utils.image_dataset_from_directory(
    dir_path,
    batch_size=batch_size,
    image_size=img_size,
    shuffle=shuffle
  )
  dataset = dataset.map(change_inputs, num_parallel_calls=AUTOTUNE)
  dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
  dataset = dataset.cache().prefetch(AUTOTUNE)
  
  return dataset

if __name__ == '__main__':
  # Load dataset
  train_path = join(DIR_PATH, 'train')
  valid_path = join(DIR_PATH, 'valid')
  train_ds= make_dataset(train_path, training=True)
  valid_ds= make_dataset(valid_path, training=False)
  
  # You must specify the model parameters.
  vae = CVAE(
      input_shape=(224, 224, 1),
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
      cp_path=SAVE_PATH
  )
  
  # Save the model
  vae.save(SAVE_PATH)
  
  with open(join(SAVE_PATH, "history.pkl"), "wb") as file:
    pickle.dump(history.history, file)
  
  # Plot loss history
  reconstruction_loss_path = join(FIG_PATH, f"reconstruction_loss.png")
  kl_loss_path = join(FIG_PATH, f"kl_loss.png")
  total_loss_path = join(FIG_PATH, f"total_loss.png")
  
  if not exists(FIG_PATH):
    makedirs(FIG_PATH)
    
  plot_loss(
    history, 
    'calculate_reconstruction_loss',
    reconstruction_loss_path
  )
  plot_loss(
    history, 
    '_calculate_kl_loss',
    kl_loss_path
  )
  plot_loss(
    history, 
    'loss',
    total_loss_path
  )