'''
  A program that implements a deep convolutional variational autoencoder

  Author: YouTube Channel - Valerio Velardo
  link: https://www.youtube.com/watch?v=TtyoFTyJuEY&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=4
  
  Date Created: June 7, 2023
  Last Updated: June 12, 2023
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from tensorflow.keras import backend as K
from keras.optimizers import Adam
import numpy as np
import os
import pickle

import tensorflow as tf

# Make sure tensorflow doesn't evaluate any operations before the whole model architecture is built
tf.compat.v1.disable_eager_execution()

'''
  Calculate the reconstruction term for the VAE loss function
'''

def calculate_reconstruction_loss(y_target, y_predicted):
  error = y_target - y_predicted
  reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
  return reconstruction_loss

'''
  Calculate the standardisation term for the VAE loss function
  
'''
def calculate_kl_loss(model):
  # wrap '_calculate_kl_loss' such that it takes the model as an argument,
  # returnsa function which can take arbitrary number of arguments
  # (for compatibility with 'metrics' and utility in loss function)
  # and returns the kl loss
  # Reference: https://stackoverflow.com/questions/73981914/tensorflow-attribute-error-method-object-has-no-attribute-from-serialized
  def _calculate_kl_loss(*args):
    # D_KL = 1/2 * \Sigma {1 + log_variance - \mu - \sigma}
    kl_loss = -0.5 * K.sum(1 + model.log_variance -
                           K.square(model.mu) - K.exp(model.log_variance), axis=1)
    return kl_loss
  return _calculate_kl_loss

class VAE:
  '''
    VAE represents a Deep Convolutional variational autoencoder architecture
  '''

  '''
    Constructor
  '''
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

  '''
    Public method that prints information about the architecture of the model to the console
  '''
  def summary(self):
    self.encoder.summary()
    self.decoder.summary()
    self.model.summary()
 
  '''
    Compile model before use
  '''
  def compile(self, learning_rate=0.0001):
    optimizer = Adam(learning_rate=learning_rate)
    # standardization + regularisation for the loss function
    self.model.compile(
      optimizer=optimizer, 
      loss=self._calculate_combined_loss,
      metrics=[calculate_reconstruction_loss, calculate_kl_loss(self)])

  '''
    Training the model
  '''
  def train(self, x_train, batch_size, num_epochs):
    # Since an autoencoder wants to minimize reconstruction loss, the desired output (second argument) is the training data itself
    self.model.fit(x_train,
                   x_train,
                   batch_size=batch_size,
                   epochs=num_epochs,
                   shuffle=True)
    
  '''
    Save a vae in the desired folder. If no such folder exists, build the folder
  '''
  def save(self, save_folder="."):
    self._create_folder(save_folder)
    self._save_parameters(save_folder)
    self._save_weights(save_folder)

  '''
    Load trained weights from model
  '''
  def load_weights(self, weights_path):
    self.model.load_weights(weights_path)

  '''
    Reconstruct images using a trained vae
  '''
  def reconstruct(self, images):
    # Pipe the images into the encoder, then pipe the result of the encoder back
    # into the decoder
    latent_representations = self.encoder.predict(images)
    reconstructed_images = self.decoder.predict(latent_representations)
    return reconstructed_images, latent_representations

  @classmethod
  def load(cls, save_folder="."):
    '''
      A method that loads an autoencoder from a saved folder
    '''
    parameters_path = os.path.join(save_folder, "parameters.pkl")
    weights_path = os.path.join(save_folder, "weights.h5")
    with open(parameters_path, "rb") as f:
      parameters = pickle.load(f)
    vae = VAE(*parameters)
    vae.load_weights(weights_path)
    return vae
  
  def _calculate_combined_loss(self, y_target, y_predicted):
    reconstruction_loss = calculate_reconstruction_loss(y_target, y_predicted)
    kl_loss = calculate_kl_loss(self)()
    combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
    return combined_loss
    
  '''
    Make a directory if it doesn't exist
  '''
  def _create_folder(self, folder):
    if not os.path.exists(folder):
      os.makedirs(folder)

  '''
    Save parameters of the autoencoder model to the specified folder
  '''
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

  '''
    Save trained weights of model
  '''
  def _save_weights(self, folder):
    save_path = os.path.join(folder, "weights.h5")
    self.model.save_weights(save_path)

  ''' 
  Three-step model building 
  '''
  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_vae()

  '''
    Link together encoder and decoder (encoder is the input to the decoder)
  '''
  def _build_vae(self):
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input, model_output, name="vae")

  '''
    A function that calls the building blocks of building the decoder
  '''
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

  '''
    Return the input layer with the latent space dimension
  '''
  def _add_decoder_input(self):
    return Input(shape=self.latent_space_dim, name="decoder_input")
  
  def _add_dense_layer(self, decoder_input):
    # 3D array [width, height, num_channels] -> 1D array
    num_neurons = np.prod(self._shape_before_bottleneck)
    dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
    return dense_layer
  
  def _add_reshape_layer(self, dense_layer):
    return Reshape(self._shape_before_bottleneck)(dense_layer)

  '''
    Adds convolutional transpose blocks

    arch is the graph of nodes representing the built network so far
  '''
  def _add_conv_transpose_layers(self, arch):
    # loop through all the conv layers in reverse order and stop at the first layer
    for layer_idx in reversed(range(1, self._num_conv_layers)):
      # [layer_A, layer_B, layer_C] -> [layer_C, layer_B]
      arch = self._add_conv_transpose_layer(arch, layer_idx)
    return arch
  
  '''
    Add one conv transpose block to the graph

    One conv transpose block consists of conv transpose, relu layer, and batch norm layer
  '''
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

  '''
    A function that calls the building blocks of creating the encoder
  '''
  def _build_encoder(self):
    # Create a basic architecture consisting of just the input layer
    encoder_input = self._add_encoder_input() 
    self._model_input = encoder_input

    # Add the specified number of convolutional layers to the existing input layer
    conv_layers = self._add_conv_layers(encoder_input)

    # Add the final "output" of the encoder, the bottleneck
    bottleneck = self._add_bottleneck(conv_layers)

    self.encoder = Model(inputs=encoder_input, outputs=bottleneck, name="encoder")

  '''
    Get Input object with the specified input shape
  '''
  def _add_encoder_input(self):
    return Input(shape=self.input_shape, name="encoder_input")

  '''
    Create convolutional blocks as specified by input

    arch is the graph of nodes representing the built network so far
  '''
  def _add_conv_layers(self, encoder_input):
    arch = encoder_input

    # Add one layer at a time to the existing architecture
    for layer_idx in range(self._num_conv_layers):
      arch = self._add_conv_layer(layer_idx, arch)
    return arch

  '''
    Add a convolutional block to the existing architecture

    Each block consists of Conv2D, ReLU, and Batch Normalization layer
  '''
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

  '''
    Flatten the data, add a bottleneck with Guassion sampling (dense layer) to 
    the architecture
  '''
  def _add_bottleneck(self, arch):
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

if __name__ == '__main__':
  vae = VAE(
    input_shape=[229, 229, 3], 
    conv_filters=[32, 64, 64, 64], 
    conv_kernels=[3, 3, 3, 3], 
    conv_strides=[1, 2, 2, 1], 
    latent_space_dim=6)
  vae.summary()
  print(vae._shape_before_bottleneck)