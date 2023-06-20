'''
  A simple variational autoencoder (not convolutional) meant to reduce
  dimension of a 1D feature vector as input.
  
  Author: Jackson Howe
  Date Created: June 20, 2023
  Last Updated: June 20, 2023
'''
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

latent_dim = 10 # Dimensionality of the latent space

# Encoder 
encoder_inputs = tf.keras.Input(shape=(2048,))
x = tf.keras.layers.Dense(512, activation='relu')(encoder_inputs)
x = tf.keras.layers.Dense(400, activation='relu')(x)
x = tf.keras.layers.Dense(240, activation='relu')(x)
z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)

# Reparameterization trick
def sampling(args):
  z_mean, z_log_var = args
  epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
  return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = tf.keras.layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = tf.keras.layers.Dense(240, activation='relu')(z)
decoder_inputs = tf.keras.layers.Dense(400, activation='relu')(z)
decoder_inputs = tf.keras.layers.Dense(512, activation='relu')(z)
decoder_outputs = tf.keras.layers.Dense(2048, activation='sigmoid')(decoder_inputs)

# Define VAE as a model
vae = tf.keras.Model(encoder_inputs, decoder_outputs)

# Define loss function for VAE
reconstruction_loss = tf.keras.losses.binary_crossentropy(encoder_inputs, decoder_outputs) * 2048
kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile the VAE
vae.compile(optimizer='adam')
vae.summary()