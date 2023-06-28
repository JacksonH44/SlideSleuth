'''
  A simple variational autoencoder (not convolutional) meant to reduce
  dimension of a 1D feature vector as input.
  
  Note: https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-ii-24b9cca69bd6 could be interesting for optimizations
  
  Author: Jackson Howe
  Date Created: June 20, 2023
  Last Updated: June 27, 2023
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
from os import listdir
from os.path import join

tf.compat.v1.disable_eager_execution()

latent_dim = 2 # Dimensionality of the latent space

def load_csv_files(directory):
  """A function that transforms csv files into a numpy array for vae training

  Args:
      directory (String): path to the root directory holding all the feature 
      csv files

  Returns:
      np.ndarray: A numpy array of each training observation
  """
  
  # Get filenames of all feature files
  all_files = []
  for file in listdir(directory):
    for mag in listdir(join(directory, file)):
      for feature in listdir(join(directory, file, mag)):
        all_files.append(join(directory, file, mag, feature))
        
  data = []
  
  # Concatenate into a list of numpy arrays
  for file in all_files:
    df = pd.read_csv(file, header=None)
    data.append(df.values)
    
  return np.concatenate(data, axis=0)

# Reparameterization trick
def sampling(args):
  z_mean, z_log_var = args
  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
  return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

def build_autoencoder_parts(input_dim):
  """Build the autoencoder architecture and return a compiled model

  Returns:
      tf.keras.Model: The compiled autoencoder
  """
  # Define the input shape of the autoencoder
  input_shape = (input_dim,)
  
  # Define the encoder architecture
  inputs = Input(shape=input_shape)
  x = Dense(128, activation='relu')(inputs)
  x = Dense(64, activation='relu')(x)
  x = Dense(32, activation='relu')(x)
  
  # Define sampling layers
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  
  encoder_output = Lambda(sampling, name="encoder_output")([z_mean, z_log_var])
  
  # Make encoder
  encoder = Model(inputs, encoder_output, name="encoder")
  encoder.summary()
  
  # Define the decoder architecture
  latent_inputs = Input(shape=(latent_dim,))
  x = Dense(32, activation='relu')(latent_inputs)
  x = Dense(64, activation='relu')(x)
  x = Dense(128, activation='relu')(x)
  decoder_outputs = Dense(input_dim, activation='sigmoid')(x)
  
  # Make decoder
  decoder = Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()
  
  return encoder, decoder

# Reference: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
def loss_func(encoder_mu, encoder_log_variance):
  """A custom loss function for a variational autoencoder. The loss function 
  consists of a reconstruction loss, which brings about the efficiency of the 
  vae, and a kl loss term that makes the latent space regular

  Args:
      encoder_mu (tf.keras.layers.Dense): The mean layer of the encoder
      encoder_log_variance (tf.keras.layers.Dense): The log variance layer of 
      the encoder
  """
  def vae_reconstruction_loss(y_true, y_predict):
    reconstruction_loss_factor = 1000
    reconstruction_loss = K.mean(K.square(y_true-y_predict), axis=[1])
    return reconstruction_loss_factor * reconstruction_loss

  def vae_kl_loss(encoder_mu, encoder_log_variance):
    kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
    return kl_loss

  def vae_kl_loss_metric(y_true, y_predict):
    kl_loss = -0.5 * K.sum(1.0 + encoder_log_variance - K.square(encoder_mu) - K.exp(encoder_log_variance), axis=1)
    return kl_loss

  def vae_loss(y_true, y_predict):
    reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
    kl_loss = vae_kl_loss(y_true, y_predict)

    loss = reconstruction_loss + kl_loss
    return loss

  return vae_loss

if __name__ == '__main__':
  # Build autoencoder
  input_dim = 2048
  encoder, decoder = build_autoencoder_parts(input_dim)
  
  # Reference for these lines: https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/
  
  # Create a later representing the input to the vae
  vae_input = Input(shape=(input_dim,), name="vae_input")
  
  # The vae input layer is then connected to the encoder to encode the input 
  # and return the latent vector
  vae_encoder_output = encoder(vae_input)
  
  # The output of the encoder is connected to the decoder
  vae_decoder_output = decoder(vae_encoder_output)
  
  # Wrap in a model object
  vae = Model(vae_input, vae_decoder_output, name="vae")
  vae.summary()
  
  z_mean = encoder.get_layer("z_mean")
  z_log_var = encoder.get_layer("z_log_var")
  
  # Compile the VAE
  vae.compile(
    optimizer='adam', 
    loss=loss_func(z_mean, z_log_var)
  )

  # Generate training data from feature vectors
  features_path = '../outputs/HNE_features'
  training_data = load_csv_files(features_path)
  print(f"Training data shape: {training_data.shape}")
  print(f"Length of training data: {len(training_data)}")

  # Fit the autoencoder model to minimize reconstruction loss
  vae.fit(
    training_data,
    training_data,
    epochs=5,
    batch_size=32
  )