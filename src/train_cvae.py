'''
  A program that trains a user-built autoencoder

  Author: YouTube Channel: Valerio Velardo, with edits by Jackson Howe
  link: https://www.youtube.com/watch?v=6fZdJKm-fSk&list=PL-wATfeyAMNpEyENTc-tVH5tfLGKtSWPp&index=6

  Date Created: June 8, 2023
  Last Updated: June 9, 2023
'''

from cvae import CVAE
import pickle

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100

'''
  A function that returns a trained variational autoencoder
'''
def train(x_train, learning_rate, batch_size, epochs):
  # YOU must specify the model parameters
  vae = CVAE(
      input_shape=(28, 28, 1),
      conv_filters=(32, 64, 64, 64),
      conv_kernels=(3, 3, 3, 3),
      conv_strides=(1, 2, 2, 1),
      latent_space_dim=2
    )
  
  # Train model
  vae.summary()
  vae.compile(learning_rate)
  vae.train(x_train, batch_size, epochs)
  return vae

if __name__ == '__main__':
  # Load stored Python object from desired location
  data_path = '../outputs/ae_data.pkl'
  data = None
  with open(data_path, 'rb') as file:
    data = pickle.load(file)
  vae = train((data[0])[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)

  # Save autoencoder model
  model_path = '../model'
  vae.save(model_path)
