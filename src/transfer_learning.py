'''
  A program that builds a custom dataset loader for WSI image data that has
  been tiled by deepzoom_tile.py. It also tests the dataset loader by using
  transfer learning to measure the AUC of the ROC curve of the model trained on
  our custom dataset.
  
  Author: Jackson Howe
  Date Created: June 15, 2023
  Last Updated: June 15, 2023
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
import pandas as pd
import os
import PIL
from PIL import ImageFile

# For troubleshooting, try some ideas on this link:
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = 229
DIR_PATH = '/scratch/jhowe4/outputs/GDC/paad_example'
CSV_PATH = '/scratch/jhowe4/outputs/GDC/paad_example/labeled_paad_example_images.csv'
ERROR_LOG_FILE = '/scratch/jhowe4/outputs/GDC/paad_example/error_log.txt'

def process_image(img_path):
  """A function that converts an image to a numpy array and normalizes 
  it as well

  Args:
      img_path (string): Path to where the image is

  Returns:
      np.array: A numpy array representing the image
  """
  img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
  x = image.img_to_array(img)
  x = x / 255
  return x

def load_image_directory(dir_path):
  """Create a list of numpy arrays from all images in a directory

  Args:
      dir_path (string): path to the directory holding all images

  Returns:
      list(np.array): List of images in the directory represented as 
      numpy arrays
  """
  data = []
  for mag in os.listdir(dir_path): # Directory has a subdirectory that is the magnification
    for img in os.listdir(os.path.join(dir_path, mag)):
      try:
        data.append(process_image(os.path.join(dir_path, mag, img)))
        # There are some corrupted images, TODO: Fix corrupted images error
      except PIL.UnidentifiedImageError:
        with open('/scratch/jhowe4/outputs/GDC/paad_example/corrupted_images2.txt', 'a') as f:
          f.write(f'{os.path.join(dir_path, mag, img)}\n')
  return data

def load_data(dir_path, csv_path):
  """Generate a dataset consisting of all images in a directory structure

  Args:
      dir_path (string): Path to the highest level directory
      csv_path (string): Path to the csv file recording the labels for each image

  Returns:
      tuple(list(np.array), list(integer)): a grouping of the observation and 
      the label each observation has
  """
  df = pd.read_csv(csv_path)
  data_directories = []
  for file in os.listdir(dir_path):
    if os.path.isdir(os.path.join(dir_path, file)):
      data_directories.append(file)
  
  for dir in data_directories:
    image_data = []
    label_data = []
    image_basename = os.path.basename(dir)
    image_name = f"{image_basename.split('_')[0]}.svs"
    try:
      label = df[df['file'] == image_name]['class'].values[0]
      data = load_image_directory(os.path.join(dir_path, dir))
      labels = [label] * len(data)
      image_data = image_data + data
      label_data = label_data + labels
    except IndexError:
      print(f"The directory {dir} has no corresponding label")
  
  return image_data, label_data

def build_model():
  """A function that builds the transfer learning model

  Returns:
      keras.Model: a ResNet50 CNN with pooling and Dense layer on top
  """
  base = ResNet50(
      weights='imagenet', # Load weights pre-trained on ImageNet
      input_shape=(IMG_SIZE, IMG_SIZE, 3),
      include_top=False # Do not include the ImageNet classifier at the top
  )

  # Freeze the base model
  base.trainable = False
  inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
  
  # Make sure the base model is running in inference mode,
  # by passing 'training=False'. This is important for fine-tuning
  x = base(inputs, training=False)
  # Convert features of shape 'base.output_shape[1:]' to vectors
  x = keras.layers.GlobalAveragePooling2D()(x)

  # A Dense classifier with a single unit (binary classification)
  outputs = keras.layers.Dense(1)(x)
  model = keras.Model(inputs, outputs)
  return model

if __name__ == '__main__':
  # Clean the log file if it exists from a previous program
  if (os.path.exists(ERROR_LOG_FILE)):
    os.remove(ERROR_LOG_FILE)
  
  # Use custom data loader to load in image data
  data, labels = load_data(DIR_PATH, CSV_PATH)
  
  print(f"Finished loading data. There are {len(data)} images overall")
  
  # Split data into training and testing. 
  # NOTE: If you have multiple cases for some image, they must all be
  # in the same group (i.e. all part of the testing set, all part of the
  # training set)
  X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

  model = build_model()
  
  # Train the model on the new data, keep track of ROC AUC
  model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()])
  
  print("...model compiled!")
  
  model.fit(
    x=np.asarray(X_train),
    y=np.asarray(y_train),
    batch_size=32,
    epochs=20,
    validation_split=0.1,
    shuffle = True
  )
  
  print("...model fitted")
  
  # Evaluate the model
  model.evaluate(
    x=np.asarray(X_test),
    y=np.asarray(y_test),
    batch_size=32
  )

  print("...model evaluated")