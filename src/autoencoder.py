'''
  A program that implements a deep convolutional variational autoencoder

  Author: YouTube Channel - Valerio Velardo
  link: https://www.youtube.com/watch?v=TtyoFTyJuEY&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=4
  
  Date Created: June 7, 2023
  Last Updated: June 8, 2023
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

class Autoencoder:
  '''
    Autoencoder represents a Deep Convolutional autoencoder architecture
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
    loss = BinaryCrossentropy() # coudl also use mse
    self.model.compile(optimizer=optimizer, loss=loss)

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
  Three-step model building 
  '''
  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_autoencoder()

  '''
    Link together encoder and decoder (encoder is the input to the decoder)
  '''
  def _build_autoencoder(self):
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input, model_output, name="autoencoder")

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
    Flatten the data and add a bottleneck (dense layer) to the architecture
  '''
  def _add_bottleneck(self, arch):
    # Store the info about the flattened model for the decoder
    self._shape_before_bottleneck = K.int_shape(arch)[1:] # [batch size, width, height, num_channels]

    # Flatten model
    arch = Flatten()(arch)
    arch = Dense(self.latent_space_dim, name="encoder_bottleneck")(arch)
    return arch

if __name__ == '__main__':
  ae = Autoencoder(
    input_shape=[28, 28, 1], 
    conv_filters=[32, 64, 64, 64], 
    conv_kernels=[3, 3, 3, 3], 
    conv_strides=[1, 2, 2, 1], 
    latent_space_dim=2)
  ae.summary()
