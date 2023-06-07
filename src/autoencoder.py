'''
  A program that implements a deep convolutional variational autoencoder

  Author: https://www.youtube.com/watch?v=TtyoFTyJuEY&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=4
  (Youtube tutorial) with adjustments from Jackson Howe
  Date Created: June 7, 2023
  Last Updated: June 7, 2023
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K

class Autoencoder:
  '''
    Autoencoder represents a Deep Convolutional autoencoder architecture
  '''

  # Constructor
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

    # Build model
    self._build()

  # Public method that prints information about the architecture of the model to the console
  def summary(self):
    self.encoder.summary();

  # Three-step model building 
  def _build(self):
    self._build_encoder()
    # self._build_decoder()
    # self._build_autoencoder()

  # A function that calls the building blocks of creating the encoder
  def _build_encoder(self):
    # Create a basic architecture consisting of just the input layer
    encoder_input = self._add_encoder_input() 

    # Add the specified number of convolutional layers to the existing input layer
    conv_layers = self._add_conv_layers(encoder_input)

    # Add the final "output" of the encoder, the bottleneck
    bottleneck = self._add_bottleneck(conv_layers)

    self.encoder = Model(inputs=encoder_input, outputs=bottleneck, name="encoder")

  # Get Input object with the specified input shape
  def _add_encoder_input(self):
    return Input(shape=self.input_shape, name="encoder_input")

  # Create convolutional "blocks" as specified by input
  def _add_conv_layers(self, encoder_input):
    arch = encoder_input

    # Add one layer at a time to the existing architecture
    for layer_idx in range(self._num_conv_layers):
      arch = self._add_conv_layer(layer_idx, arch)
    return arch

  # Add a convolutional block to the existing architecture
  # 
  # Each block consists of Conv2D, ReLU, and Batch normalization layer
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

  # Flatten the data and add a bottleneck (dense layer) to the architecture
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
