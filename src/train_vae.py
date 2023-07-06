from vae import VAE
from vae import load_csv_files
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from os import listdir
from os.path import join
import os
import pickle

LATENT_SPACE_DIM = 2
TRAIN_DIR = '../outputs/HNE_features/train'
TEST_DIR = '../outputs/HNE_features/test'

BATCH_SIZE = 32
NUM_EPOCHS = 100

def plot_loss(history, metric):
  """A function that plots the loss metric over time for a vae training session

  Args:
      history (History): The history of a fitted model
  """
  plt.plot(
    history.epoch, 
    history.history[metric], 
    color='deepskyblue', 
    label = f'Train {metric}'
  )
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(f'Training {metric}')
  plt.savefig(f'../img/vae_{metric}')
  plt.close()
  
if __name__ == '__main__':
  # Generate training data from feature vectors
  train_data = load_csv_files(TRAIN_DIR)
  test_data = load_csv_files(TEST_DIR)
  
  input_shape = [train_data.shape[1]]
  
  # Build the vae
  vae = VAE(
    input_shape,
    latent_space_dim=LATENT_SPACE_DIM
  )
  vae.summary()
  vae.compile()
  print("...Compiled!")
  
  # Fit the model
  history = vae.train(
    X_train=train_data,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS
  )
  
  # Plot the loss
  plot_loss(history, 'calculate_reconstruction_loss')
  plot_loss(history, '_calculate_kl_loss')
  plot_loss(history, 'loss')