'''
  A simple variational autoencoder (not convolutional) meant to reduce
  dimension of a 1D feature vector as input.
  
  Note: https://towardsdatascience.com/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-ii-24b9cca69bd6 could be interesting for optimizations
  
  Author: Jackson Howe
  Date Created: June 20, 2023
  Last Updated: July 5, 2023
'''
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from os import listdir
from os.path import join
import os
import pickle

tf.compat.v1.disable_eager_execution()

LATENT_SPACE_DIM = 2 # Dimension of the latent space
TRAIN_DIR = '../outputs/HNE_features/train'
TEST_DIR = '../outputs/HNE_features/test'

BATCH_SIZE = 256
NUM_EPOCHS = 10

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

def calculate_reconstruction_loss(y_target, y_predicted):
  """A function that calculates the reconstruction part of the loss function

  Args:
      y_target (np.array): An array of values that represents the ground truth 
      vector
      y_predicted (np.array): An array of values that represents the predicted 
      vector

  Returns:
      Integer: The reconstruction loss value
  """
  error = y_target - y_predicted
  reconstruction_loss = K.mean(K.square(error))
  return reconstruction_loss

def calculate_kl_loss(model):
  """Calculate the Kullback-Leibler divergence part of the loss function.

  Args:
      model (tf.keras.models.Model): The model you want to compute loss for

  Returns:
      <tf.Tensor>: A tensor representing the KL loss
  """
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

class VAE:
  """
    VAE represents a deep variational autoencoder architecture
  """
  def __init__(self, input_shape, latent_space_dim):
    """Constructor

    Args:
        input_shape (Tuple): A tuple representing the shape of the input to the 
        model
        latent_space_dim (Integer): The dimension of the bottleneck in the model
    """
    self.input_shape = input_shape
    self.latent_space_dim = latent_space_dim
    self.reconstruction_loss_weight = 1000
    
    self.encoder = None
    self.decoder = None
    self.model = None
    
    # Private variable
    self._model_input = None
    
    # Build model
    self._build()
    
  def summary(self):
    """A public method that prints information about the architecture of the 
    model to the console
    """
    self.encoder.summary()
    self.decoder.summary()
    self.model.summary()
    
  def compile(self, learning_rate=1e-3):
    """Compile model before use

    Args:
        learning_rate (Integer, optional): The model learning rate. Defaults to 
        1e-3.
    """
    optimizer = Adam(learning_rate=learning_rate)
    
    # Standardization + regularisation for the loss function
    self.model.compile(
      optimizer=optimizer,
      loss=self._calculate_combined_loss,
      metrics=[calculate_reconstruction_loss, calculate_kl_loss(self)]
    )
    
  def save(self, save_folder="."):
    """Save the weights of a model

    Args:
        save_folder (str, optional): The path to the folder in which the model 
        weights will be saved. Defaults to the current working directory.
    """
    self._create_folder(save_folder)
    self._save_weights(save_folder)
    self._save_parameters(save_folder)
    
  def train(self, X_train, batch_size, num_epochs):
    """A function that trains the model

    Args:
        X_train (np.array): The training set for the model
        batch_size (Integer): Batch size
        num_epochs (Integer): Number of full training epochs
    """
    # Create an early stopping callback based on reconstruction loss
    # early_stopping = tf.keras.callbacks.EarlyStopping(
    #   monitor=TODO: Find what to monitor!!,
    #   verbose=1,
    #   patience=10,
    #   mode='max',
    #   restore_best_weights=True
    # )
      
    # Fit the model
    self.model.fit(
      X_train,
      X_train, 
      batch_size=batch_size,
      epochs=num_epochs,
      # callbacks=[early_stopping],
      verbose=1 
    )
    
  def load_weights(self, weights_path):
    """Load trained weights from a model

    Args:
        weights_path (str): Path to the model weights
    """
    self.model.load_weights(weights_path)
    
  @classmethod
  def load(cls, save_folder="."):
    """A method that loads weights for a variational autoencoder

    Args:
        save_folder (str, optional): The path to the folder to load weights 
        from. Defaults to current working directory.
    """
    parameters_path = os.path.join(save_folder, "parameters.pkl")
    weights_path = os.path.join(save_folder, "weights.h5")
    with open(parameters_path, "rb") as f:
      parameters = pickle.load(f)
    
    # Build the VAE from the parameters
    vae = VAE(*parameters)
    vae.load_weights(weights_path)
    return vae
    
  def _create_folder(self, folder):
    """Create a folder if it doesn't exist

    Args:
        folder (String): Path to the folder
    """
    if not os.path.exists(folder):
      os.makedirs(folder)
      
  def _save_weights(self, folder):
    """Save trained weights of a model

    Args:
        folder (str): Path to folder to save weights to
    """
    save_path = os.path.join(folder, "weights.h5")
    self.model.save_weights(save_path)
    
  def _save_parameters(self, folder):
    """Save parameters for a model

    Args:
        folder (str): Path to parameter folder
    """
    parameters = [
      self.input_shape,
      LATENT_SPACE_DIM
    ]
    
    # Dump into a pickle file
    save_path = os.path.join(folder, "parameters.pkl")
    with open(save_path, "wb") as f:
      pickle.dump(parameters, f)
    
  def _calculate_combined_loss(self, y_target, y_predicted):
    """Custom VAE loss function consisting of the reconstruction loss and 
    KL-Divergence

    Args:
        y_target (np.array): Ground truth data in vector format
        y_predicted (np.array): Predicted data in vector format
    """
    reconstruction_loss = calculate_reconstruction_loss(y_target, y_predicted)
    kl_loss = calculate_kl_loss(self)()
    combined_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss
    return combined_loss
    
  def _build(self):
    """Three-step model building
    """
    self._build_encoder()
    self._build_decoder()
    self._build_vae()
    
  def _build_vae(self):
    """Link together encoder and decoder (encoder is the input to the decoder)
    """
    model_input = self._model_input
    model_output = self.decoder(self.encoder(model_input))
    self.model = Model(model_input, model_output, name="vae")
    
  def _build_decoder(self):
    """A function that builds the decoder layer by layer
    """
    decoder_input = self._add_decoder_input()
    
    # Add dense layers
    dense_layers = self._add_decoder_dense_layers(decoder_input)
    
    # Add decoder output classification layer
    decoder_output = self._add_decoder_output(dense_layers)
    
    # Create the decoder
    self.decoder = Model(
      decoder_input, 
      decoder_output, 
      name="decoder"
    )
    
  def _add_decoder_input(self):
    """Create an input layer with the latent space dimension

    Returns:
        tf.keras.layer.Input: The decoder input layer
    """
    return Input(shape=self.latent_space_dim, name="decoder_input")
  
  def _add_decoder_dense_layers(self, decoder_input):
    """Add the dense layers that make up the decoder

    Args:
        decoder_input (tf.keras.layer.Input): The decoder input layer from the 
        encoder bottleneck
    """
    x = decoder_input
    
    # First layer of decoder (corresponds to third layer of encoder)
    x = Dense(
      32,
      activation='relu',
      name='decoder_dense_3'
    )(x)
    x = BatchNormalization(
      name='encoder_batch_norm_3'
    )(x)
    
    # Second layer of decoder (corresponds to second layer of encoder)
    x = Dense(
      256,
      activation='relu',
      name='decoder_dense_2'
    )(x)
    x = BatchNormalization(
      name='encoder_batch_norm_2'
    )(x)
    
    # Third layer of decoder (corresponds to first layer of encoder)
    x = Dense(
      1024,
      activation='relu',
      name='decoder_dense_1'
    )(x)
    x = BatchNormalization(
      name='encoder_batch_norm_1'
    )(x)
    
    return x
  
  def _add_decoder_output(self, dense_layers):
    """Adds the outer classification layer for the decoder

    Args:
        dense_layers (tf.keras.layer.Dense): The decoder architecture without the outer classification layer
    """
    x = dense_layers
    
    # Get the input shape (assuming the shape is [features,])
    input_dim = self.input_shape[0]
    x = Dense(input_dim, activation='sigmoid')(x)
    return x
    
  def _build_encoder(self):
    """A function that builds the encoder part of the VAE layer by layer
    """
    # Save the input layer in a variable to use for final model building
    encoder_input = self._add_encoder_input()
    self._model_input = encoder_input
    
    # Can change/play around with this architecture
    dense_layers = self._add_encoder_dense_layers(encoder_input)
    
    # Add the bottleneck, can change dimension in constructor
    bottleneck = self._add_bottleneck(dense_layers)
    
    # Create the encoder
    self.encoder = Model(
      inputs=encoder_input, 
      outputs=bottleneck, 
      name="encoder"
    )
    
  def _add_encoder_input(self):
    """Create an input object with the specified input shape

    Returns:
        tf.keras.layer.Input: The input layer to the VAE
    """
    return Input(shape=self.input_shape, name="encoder_input")
  
  def _add_encoder_dense_layers(self, encoder_input):
    """Add the main body of the VAE using dense layers. Add batch norm layers 
    to speed up training

    Args:
        encoder_input (tf.keras.layers.Input): The encoder input

    Returns:
        tf.keras.layers.Dense: The architecture of the encoder without the 
        bottleneck
    """
    x = encoder_input
    
    # First dense layer
    x = Dense(
      1024,
      activation='relu',
      name='encoder_dense_1'
    )(x)
    x = BatchNormalization(
      name='encoder_batch_norm_1'
    )(x)
    
    # Second dense layer
    x = Dense(
      512,
      activation='relu',
      name='encoder_dense_2'
    )(x)
    x = BatchNormalization(
      name='encoder_batch_norm_2'
    )(x)
    
    # Third dense layer
    x = Dense(
      32,
      activation='relu',
      name='encoder_dense_3'
    )(x)
    x = BatchNormalization(
      name='encoder_batch_norm_3'
    )(x)
    
    return x
  
  def _add_bottleneck(self, dense_layers):
    """Add a bottleneck with Gaussian sampling (dense layers) to the 
    architecture.
    
    NOTE: We no longer have a sequential model, the architecture branches out into the mean layer and the log variance layer

    Args:
        dense_layers (tf.keras.layers.Dense): The encoder without the bottleneck

    Returns:
        tf.keras.layers.Lambda: The encoder layers including the bottleneck 
        sampling layer
    """
    x = dense_layers
    
    # Branching the model into mean and log variance layers
    self.mu = Dense(self.latent_space_dim, name="mu")(x)
    self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)
    
    # Define the function for the lambda layer
    def sample_point_normal(args):
      mu, log_variance = args
      
      # Sample a point from the standard normal distribution
      epsilon = K.random_normal(shape=K.shape(self.mu), mean=0, stddev=1)
      
      # The equation for sampling a point is:
      # z = \mu + Sigma * epislon
      # where \Sigma = exp((log_variance) / 2)
      sampled_point = mu + K.exp(log_variance / 2) * epsilon
      return sampled_point
    
    # Connect mean and log variance again by sampling a point from distribution
    x = Lambda(
      sample_point_normal,
      name="encoder_output"
    )([self.mu, self.log_variance])
    
    return x

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
  vae.train(
    X_train=train_data,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS
  )
  
  vae.save('../model/vae-2023-07-06')