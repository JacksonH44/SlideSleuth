'''
  A program that implements a deep convolutional variational autoencoder

  Author: https://www.youtube.com/watch?v=TtyoFTyJuEY&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=4
  (Youtube tutorial) with adjustments from Jackson Howe
  Date Created: June 7, 2023
  Last Updated: June 7, 2023
'''

class Autoencoder:
  '''
    Autoencoder represents a Deep Convolutional autoencoder architecture
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

    self._num_conv_layers = len(conv_filters)

    self._build()

  def _build(self):
    self._build_encoder()
    self._build_decoder()
    self._build_autoencoder()