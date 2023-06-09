"""
  A program to analyze the effectiveness of an autoencoder

  Author: YouTube Channel: Valerio Velardo
  link: https://www.youtube.com/watch?v=-HqG2s4dxJ0&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=8

  Date Created: June 9, 2023
  Last Updated: June 9, 2023
"""

import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
import pickle

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
  fig = plt.figure(figsize=(15,3))
  num_images = len(images)
  for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
    image = image.squeeze()
    ax = fig.add_subplot(2, num_images, i + 1)
    ax.axis("off")
    ax.imshow(image, cmap="gray_r")
    reconstructed_image = reconstructed_image.squeeze()
    ax = fig.add_subplot(2, num_images, i + num_images + 1)
    ax.axis("off")
    ax.imshow(reconstructed_image, cmap="gray_r")
  plt.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
  plt.figure(figsize=(10, 10))
  plt.scatter(
    latent_representations[:, 0],
    latent_representations[:, 1],
    cmap="rainbow",
    c=sample_labels,
    alpha=0.5,
    s=2
  )
  plt.colorbar()
  plt.show()

if __name__ == "__main__":
  # Load saved autoencoder
  model_path = '../outputs/model'
  autoencoder = Autoencoder.load(model_path)

  # Load mnist data
  data_path = '../outputs/ae_data.pkl'
  data = None
  with open(data_path, 'rb') as file:
    data = pickle.load(file)
  x_train, y_train, x_test, y_test = data

  # Sample images
  num_sample_show = 8
  sample_images, _ = select_images(x_test, y_test, num_sample_show)

  # Reconstruct the image with the autoencoder and plot them
  reconstructed_images, _ = autoencoder.reconstruct(sample_images)
  plot_reconstructed_images(sample_images, reconstructed_images)
