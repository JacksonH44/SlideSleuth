'''
  A program that saves the MNIST dataset for testing purposes

  Author: Jackson Howe
  Date Created: June 8, 2023
  Last Updated: June 8, 2023
'''

from tensorflow.keras.datasets import mnist
import pickle

'''
  A function to load desired data to the model
'''
  
def load_mnist():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.astype("float32") / 255
  x_train = x_train.reshape(x_train.shape + (1,))

  x_test = x_test.astype("float32") / 255
  x_test = x_test.reshape(x_test.shape + (1,))

  return [x_train, y_train, x_test, y_test]

if __name__ == '__main__':
  # Store Python instance of Autoencoder object in desired location
  data = load_mnist()
  data_path = '../outputs/ae_data.pkl'
  with open(data_path, 'wb') as file:
    pickle.dump(data, file)
