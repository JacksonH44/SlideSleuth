"""Feature extraction pipeline

    Date Created: May 25, 2023
    Last Updated: June 15, 2023
"""

__author__ = 'Jackson Howe'

from os.path import exists
import sys

import PIL
from PIL import ImageFile
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

def instantiate():
  """Creates a feature extraction model

  Returns:
      tf.keras.applications.resnet50.ResNet50: ResNet50 feature extraction model
  """
  
  # Extract a feature vector - don't need top classification layer, instead we'll need a pooling layer.
  # NOTE: can change pooling to 'max' if you wish
  # See https://keras.io/api/applications/resnet/ for more (May 23, 2023)
  model = ResNet50(weights='imagenet',
                   include_top=False,
                   pooling='avg')
  return model

def extract_features(model, img_path, output_path):
  """A function that extracts a feature vector from an image tile"""

  # Visit https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image for more info
  # about preprocessing library (May 23, 2023)

  try:
    # Convert to PIL image
    img = image.load_img(img_path,
                       target_size=(224, 224))

    # Convert PIL image to numpy array
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Adequate image to format ResNet50 requires (caffe style)
    # Visit https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input
    # or https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras for more info
    # (May 23, 2023)
    x = preprocess_input(x)
    print(x.shape)

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
  except PIL.UnidentifiedImageError:
    print(f"The image at {img_path} is unreadable")


if __name__ == '__main__':
  img_path = sys.argv[1]
  data_path = sys.argv[2]

  model = instantiate()
  extract_features(model, img_path, data_path)
