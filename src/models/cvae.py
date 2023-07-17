"""A class that implements a deep convolutional variational autoencoder

  Author: YouTube Channel - Valerio Velardo
  link: https://www.youtube.com/watch?v=TtyoFTyJuEY&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=4
  
  Date Created: June 7, 2023
  Last Updated: June 12, 2023
"""

import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

# Make sure tensorflow doesn't evaluate any operations before the whole model architecture is built
tf.compat.v1.disable_eager_execution()


def calculate_reconstruction_loss(y_target, y_predicted):
  """Calculate reconstruction term for VAE loss function"""
  
  error = y_target - y_predicted
  reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
  return reconstruction_loss


def calculate_kl_loss(model):
  """Calculate regularization term for VAE loss function"""
  
  # wrap '_calculate_kl_loss' such that it takes the model as an argument,
  # returns a function which can take arbitrary number of arguments
  # (for compatibility with 'metrics' and utility in loss function)
  # and returns the kl loss
  # Reference: https://stackoverflow.com/questions/73981914/tensorflow-attribute-error-method-object-has-no-attribute-from-serialized
  def _calculate_kl_loss(*args):
    # D_KL = 1/2 * \Sigma {1 + log_variance - \mu - \sigma}
    kl_loss = -0.5 * K.sum(1 + model.log_variance -
                           K.square(model.mu) - K.exp(model.log_variance), axis=1)
    return kl_loss
  return _calculate_kl_loss


class CVAE:
  """CVAE represents a deep convolutional variational autoencoder architecture"""

  def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
    self.input_shape = input_shape # [width, height, num_channels]
    self.conv_filters = conv_filters # [first, second, ..., last]
    self.conv_kernels = conv_kernels # []
    self.conv_strides = conv_strides # []
    self.latent_space_dim = latent_space_dim
    self.reconstruction_loss_weight = 1000

    self.encoder = None
    self.decoder = None
    self.model = None

    # Private variables
    self._num_conv_layers = len(conv_filters)
    self._shape_before_bottleneck = None
    self._model_input = None

    # Build model
    self._build()

  def summary(self):
    """Print architecture information to the console."""
    
    self.encoder.summary()
    self.decoder.summary()
    self.model.summary()
 
  def compile(self, learning_rate=0.0001):
    """Compile the model before use"""
    
    optimizer = Adam(learning_rate=learning_rate)
    # standardization + regularisation for the loss function
    self.model.compile(
      optimizer=optimizer, 
      loss=self._calculate_combined_loss,
      metrics=[calculate_reconstruction_loss, calculate_kl_loss(self)])

  def train(self, x_train, num_epochs, steps_per_epoch, validation_data, validation_steps, cp_path):
    """Train the model"""
    # Since an autoencoder wants to minimize reconstruction loss, the desired output (second argument) is the training data itself
    
    # Create an early stopping callback based on reconstruction loss.
    early_stopping = tf.keras.callbacks.EarlyStopping(
      monitor='val_calculate_reconstruction_loss',
      verbose=1,
      patience=10,
      mode='min',
      restore_best_weights=True
    )
    
    # Create a checkpoint to save weights after each epoch.
    self._create_folder(cp_path)
    cp_path = os.path.join(cp_path, 'weights-{epoch:02d}.h5')
    
    model_cp = tf.keras.callbacks.ModelCheckpoint(
      filepath=cp_path, 
      monitor='val_calculate_reconstruction_loss',
      verbose=1,
      save_freq='epoch',
      save_weights_only=True
    )
    
    # Save model initial weights
    self.model.save_weights(cp_path.format(epoch=0))
    
    # Train the model and return its history
    history = self.model.fit(
      x_train,
      epochs=num_epochs,
      steps_per_epoch=steps_per_epoch,
      callbacks=[early_stopping, model_cp],
      validation_data=validation_data,
      validation_steps=validation_steps,
      shuffle=True
    )
    
    return history
    
  def save(self, save_folder="."):
    """Save a VAE to the desired folder"""
    
    self._create_folder(save_folder)
    self._save_parameters(save_folder)
    self._save_weights(save_folder)

  def load_weights(self, weights_path):
    """Load trained weights for a model"""
    
    self.model.load_weights(weights_path)

  def reconstruct(self, images):
    """Reconstruct images using a trained VAE"""
    
    # Pipe the images into the encoder, then pipe the result of the encoder back
    # into the decoder
    latent_representations = self.encoder.predict(images)
    reconstructed_images = self.decoder.predict(latent_representations)
    return reconstructed_images, latent_representations

  @classmethod
  def load(cls, save_folder="."):
    """Load an autoencoder from a saved folder"""
    
    parameters_path = os.path.join(save_folder, "parameters.pkl")
    weights_path = os.path.join(save_folder, "weights.h5")
    with open(parameters_path, "rb") as f:
      parameters = pickle.load(f)
    vae = CVAE(*parameters)
    vae.load_weights(weights_path)
    return vae
  
  def _calculate_combined_loss(self, y_target, y_predicted):
    reconstruction_loss = calculate_reconstruction_loss(y_target, y_predicted)
    kl_loss = calculate_kl_loss(self)()
    combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
    return combined_loss
    
  def _create_folder(self, folder):
    if not os.path.exists(folder):
      os.makedirs(folder)

  def _save_parameters(self, folder):
    parameters = [
      self.input_shape,
      self.conv_filters,
      self.conv_kernels,
      self.conv_strides,
      self.latent_space_dim
    ]

    # Dump into a pickle file
    save_path = os.path.join(folder, "parameters.pkl")
    with open(save_path, "wb") as f:
      pickle.dump(parameters, f)

  def _save_weights(self, folder):
    save_path = os.path.join(folder, "weights.h5")
    self.model.save_weights(save_path)

  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_vae()

  def _build_vae(self):
    """Link together encoder and decoder (encoder is the input to the decoder)."""
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input, model_output, name="vae")

  def _build_decoder(self):
    # Input layer
    decoder_input = self._add_decoder_input()

    # Add dense layer with flattened shape before bottleneck
    dense_layer = self._add_dense_layer(decoder_input)

    # Reshape model to get encoder shape before bottleneck
    reshape_layer = self._add_reshape_layer(dense_layer)

    # Add all the conv transpose blocks in the reverse order of the encoder
    conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)

    # Add the last conv transpose layer and the output activation layer
    decoder_output = self._add_decoder_output(conv_transpose_layers)

    self.decoder = Model(decoder_input, decoder_output, name="decoder")

  def _add_decoder_input(self):
    return Input(shape=self.latent_space_dim, name="decoder_input")
  
  def _add_dense_layer(self, decoder_input):
    # 3D array [width, height, num_channels] -> 1D array
    num_neurons = np.prod(self._shape_before_bottleneck)
    dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
    return dense_layer
  
  def _add_reshape_layer(self, dense_layer):
    return Reshape(self._shape_before_bottleneck)(dense_layer)

  def _add_conv_transpose_layers(self, arch):
    """Adds convolutional transpose blocks"""
    
    # loop through all the conv layers in reverse order and stop at the first layer
    for layer_idx in reversed(range(1, self._num_conv_layers)):
      # [layer_A, layer_B, layer_C] -> [layer_C, layer_B]
      arch = self._add_conv_transpose_layer(arch, layer_idx)
    return arch
  
  def _add_conv_transpose_layer(self, arch, layer_idx):
    # The conv transpose layer number for the decoder is the reverse of the encoder
    layer_num = self._num_conv_layers - layer_idx

    # Add conv transpose layer
    conv_transpose_layer = Conv2DTranspose(
      filters=self.conv_filters[layer_idx],
      kernel_size=self.conv_kernels[layer_idx],
      strides=self.conv_strides[layer_idx],
      padding="same",
      name=f"decoder_conv_transpose_layer_{layer_num}"
    )
    arch = conv_transpose_layer(arch)

    # Add ReLU layer
    arch = ReLU(name=f"decoder_relu_{layer_num}")(arch)

    # Add Batch Normalization layer
    arch = BatchNormalization(name=f"decoder_bn_{layer_num}")(arch)
                                                              
    return arch

  def _add_decoder_output(self, arch):
    # Add last conv transpose layer
    conv_transpose_layer = Conv2DTranspose(
        # [width, height, num_channels] coincide with encoder input
        filters=1, 
        kernel_size=self.conv_kernels[0],
        strides=self.conv_strides[0],
        padding="same",
        name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
    )
    arch = conv_transpose_layer(arch)

    # Add last activation layer
    output_layer = Activation("sigmoid", name="sigmoid_layer")
    arch = output_layer(arch)
    return arch

  def _build_encoder(self):
    # Create a basic architecture consisting of just the input layer
    encoder_input = self._add_encoder_input() 
    self._model_input = encoder_input

    # Add the specified number of convolutional layers to the existing input layer
    conv_layers = self._add_conv_layers(encoder_input)

    # Add the final "output" of the encoder, the bottleneck
    bottleneck = self._add_bottleneck(conv_layers)

    self.encoder = Model(inputs=encoder_input, outputs=bottleneck, name="encoder")

  def _add_encoder_input(self):
    return Input(shape=self.input_shape, name="encoder_input")

  def _add_conv_layers(self, encoder_input):
    """Create convolutional layers for the encoder"""
    
    arch = encoder_input

    # Add one layer at a time to the existing architecture
    for layer_idx in range(self._num_conv_layers):
      arch = self._add_conv_layer(layer_idx, arch)
    return arch
  
  def _add_conv_layer(self, layer_index, arch):
    layer_num = layer_index + 1

    # Create a convolutional layer
    conv_layer = Conv2D(
      filters=self.conv_filters[layer_index],
      kernel_size=self.conv_kernels[layer_index],
      strides=self.conv_strides[layer_index],
      padding="same",
      name=f'encoder_conv_layer_{layer_num}'
    )

    # Apply the new layer to the existing architecture
    arch = conv_layer(arch)

    # Create a ReLU layer and add to existing block
    arch = ReLU(name=f'encoder_relu_{layer_num}')(arch)

    # Create a batch normalization and apply it to the block
    arch = BatchNormalization(name=f'encoder_bn_{layer_num}')(arch)

    # Return the architecture with an additional conv block
    return arch

  def _add_bottleneck(self, arch):
    """Flatten the data, add a bottleneck with Gaussian sampling to the architecture"""
    
    # Store the info about the flattened model for the decoder
    self._shape_before_bottleneck = K.int_shape(arch)[1:] # [batch size, width, height, num_channels]

    # Flatten model and create layers for mean and variance
    arch = Flatten()(arch)
    
    # NOTE: we no longer have a sequential model, the architecture branches out
    # into the mean layer and the log variance layer
    self.mu = Dense(self.latent_space_dim, name="mu")(arch)
    self.log_variance = Dense(self.latent_space_dim, name="log_variance")(arch)
    
    # Define the function for the lambda layer
    def sample_point_normal(args):
      mu, log_variance = args
      
      # Sample a point from a standard normal distribution
      epsilon = K.random_normal(shape=K.shape(self.mu), mean=0, stddev=1)
      
      # The equation for sampling a point is:
      #
      # z = \mu + \Sigma * \epsilon 
      #
      # where \Sigma = exp((log_variance) / 2)
      sampled_point = mu + K.exp(log_variance / 2) * epsilon
      return sampled_point
    
    # Connect mean and log variance again by sampling a point from a normal 
    # distribution
    arch = Lambda(
      sample_point_normal, 
      name="encoder_output")([self.mu, self.log_variance])
    
    return arch