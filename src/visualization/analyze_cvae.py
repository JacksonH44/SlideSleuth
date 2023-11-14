"""
  A program to analyze the effectiveness of an autoencoder

  Author: YouTube Channel: Valerio Velardo
  link: https://www.youtube.com/watch?v=-HqG2s4dxJ0&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=8

  Date Created: June 9, 2023
  Last Updated: August 1, 2023
"""

import sys
sys.path.insert(0, '/home/jhowe4/projects/def-sushant/jhowe4/SlideSleuth/src/models')
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cvae import CVAE

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 224
DATASET_PATH = '../../data/processed/datasets/cvae_dataset.pkl'
SAVE_PATH = '../../reports/figures/cvae-20230728-102327/reconstructed_images.svg'
MODEL_PATH = '../../models/cvae-20230728-102327/best'
  
def plot_reconstructed_images(images, reconstructed_images):
  num_images = len(images)
  num_rows = num_images
  num_cols = 2
  
  fig, axes = plt.subplots(num_rows, num_cols, figsize=(3,12))
  
  for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
    image = image.squeeze()
    ax = axes[i, 0]
    ax.axis("off")
    ax.imshow(image)
    
    reconstructed_image = reconstructed_image.squeeze()
    ax = axes[i, 1]
    ax.axis("off")
    ax.imshow(reconstructed_image, cmap='plasma')

  fig.tight_layout()
  plt.subplots_adjust(wspace=0)
  plt.savefig(SAVE_PATH, format='svg', transparent=True)

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
  plt.figure(figsize=(20, 20))
  plt.scatter(
    latent_representations[:, 0],
    latent_representations[:, 1],
    cmap="rainbow",
    c=sample_labels,
    alpha=0.5,
    s=2
  )
  plt.colorbar()
  plt.savefig('../img/latent_space.pdf', format='png', transparent=True)
  
def _change_inputs(image, _):
    return image, image

def _normalize(image, mirrored_image):
    return (tf.cast(image, tf.float32) / 255.0, tf.cast(mirrored_image, tf.float32) / 255.0)
  
def data_pipeline(dir_path, batch_size=BATCH_SIZE, img_size=(IMG_SIZE, IMG_SIZE), shuffle=True):
  
  dataset = tf.keras.utils.image_dataset_from_directory(
    dir_path,
    batch_size=batch_size,
    image_size=img_size,
    shuffle=shuffle
  )
  dataset = dataset.map(_change_inputs, num_parallel_calls=AUTOTUNE)
  dataset = dataset.map(_normalize, num_parallel_calls=AUTOTUNE)
  dataset = dataset.cache().prefetch(AUTOTUNE)
  return dataset

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
  # Load saved autoencoder
  model_path = MODEL_PATH
  vae = CVAE.load(model_path)
  
  # Load sample images dataset with pickle
  sample_images = None
  with open(DATASET_PATH, 'rb') as file:
    sample_images = pickle.load(file)

  # Reconstruct the image with the autoencoder and plot them
  reconstructed_images, _ = vae.reconstruct(sample_images)
  plot_reconstructed_images(sample_images, reconstructed_images)
  
  # NOTE: Experimental code below
  # Make generator dataset from specified directory.
  # test_ds = make_dataset('../../data/processed/CK7/CK7_cvae/test')
  # vae.generate_latent_representations(test_ds, '../../data/processed/CK7/CK7_cvae/test/latent_representations.tsv', '../../data/interim/uhn_labels.csv')