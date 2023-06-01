'''
Description: A file that extracts a feature vector from a test image using transfer learning
Author: Jackson Howe
Last Updated: May 25, 2023
'''

# Imports
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from os.path import exists
import sys

'''
  A function that instatiates the transfer learning model
'''

def instantiate():
  # Extract a feature vector - don't need top classification layer, instead we'll need a pooling layer
  # NOTE: can change pooling to 'max' if you wish
  # See https://keras.io/api/applications/resnet/ for more (May 23, 2023)
  model = ResNet50(weights='imagenet',
                   include_top=False,
                   pooling='avg')
  return model

'''
  A function that extracts a feature vector from an image using transfer learning.
'''

def extract_features(model, img_path, output_path):

  # Visit https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image for more info
  # about preprocessing library (May 23, 2023)

  # Convert to PIL image
  img = image.load_img(img_path,
                       target_size=(229, 229))

  # Convert PIL image to numpy array
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  # Adequate image to format ResNet50 requires (caffe style)
  # Visit https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
  # or https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras for more info
  # (May 23, 2023)
  x = preprocess_input(x)

  # Extract feature vector
  # Current length is 2048
  features = model.predict(x)

  # Write vector to the csvfile specified by output_path
  if exists(output_path):
    mode = 'a'
  else:
    mode = 'w'

  with open(output_path, mode, newline='\n') as csvfile:
      np.savetxt(csvfile, features, delimiter=",", fmt="%1.5f")


if __name__ == '__main__':
  # This is a specific path on my own device with a test image, for other tests on different devices
  # this path must be changed
  # img_path = '/scratch/jhowe4/results/2023-05-23/1/001_files/5.0/10_20.jpeg'

  # This is a specific path on my own device with test data, for other tests on different devices
  # this path must be changed
  # data_path = '/scratch/jhowe4/data/feature_vector.csv'

  img_path = sys.argv[1]
  data_path = sys.argv[2]

  model = instantiate()
  extract_features(model, img_path, data_path)
