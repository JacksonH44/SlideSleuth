'''
  A program that performs a classification task using transfer learning. Right
  now, the program is coded to classify PAAD WSIs from TCGA as either primary
  tumour or solid tissue normal. 
  
  Author: Jackson Howe
  References: https://github.com/Akshina07/DeepLearning-TumorHypoxia/blob/master/Training_Evaluation/resnet/train_resnet.py
  Date Created: June 19, 2023
  Last Updated: June 20, 2023
'''

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import MeanSquaredError
import sys
from os.path import abspath, join, isdir
from os import listdir
import math
import matplotlib.pyplot as plt
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABELS = 'labels.csv'
IMG_DIR = '/scratch/jhowe4/inputs/GDC/paad_example'
IMAGE_RESIZE = 224
BATCH_SIZE = 32
NUM_TRAINING_TILES = 32670
NUM_VALID_TILES = 3885
NUM_TEST_TILES = 3068
CHANNELS = 3
DENSE_LAYER_ACTIVATION = 'softmax'
POOLING = 'avg'
OBJECTIVE_FUNCTION = 'binary_crossentropy'
LOSS_METRICS = ['accuracy']

# Training global variables
NUM_EPOCHS = 4
BATCH_SIZE_TRAINING = 200
BATCH_SIZE_VALIDATION = 200
STEPS_PER_EPOCH_TRAINING = math.ceil(NUM_TRAINING_TILES/BATCH_SIZE_TRAINING)
STEPS_PER_EPOCH_VALIDATION = math.ceil(NUM_VALID_TILES/BATCH_SIZE_VALIDATION)

def create_df(dir_path):
  """Create the dataframes for the training, validation, and testing sets of data to be used to generate the Tensorflow data generator. This function assumes dir_path is the root directory, with three subdirectories: train, test, valid, and a labels.csv file in the root directory as well.

  Args:
      dir_path (string): path to the root directory

  Returns:
      train, validation, and test dataframes: dataframes for the train, test, and validation data to be used in the model
  """
  # Get absolute path of directory
  dir_path = abspath(dir_path)
  
  # Get the name of the label file and read into a dataframe
  label_path = join(dir_path, LABELS)
  label_df = pd.read_csv(label_path)
  
  # Create train, validation, and test dataframes
  train_dict = {'x_col': [], 'y_col': []}
  validation_dict = {'x_col': [], 'y_col': []}
  test_dict = {'x_col': [], 'y_col': []}
  
  # There should be three subdirectories: train, valid, and test
  # Training
  for dir in listdir(join(dir_path, 'train')):
    if (isdir(join(dir_path, 'train', dir))):
      # Extract corresponding label value
      for _, row in label_df.iterrows():
        if row[0].rsplit(".", 1)[0] == (dir.split("_")[0]):
          label = row[1]
      for mag in listdir(join(dir_path, 'train', dir)):
        for img_file in listdir(join(dir_path, 'train', dir, mag)):
          # Add the image path label pair to the training dictionary
          img_path = join(dir_path, 'train', dir, mag, img_file)
          train_dict['x_col'].append(img_path)
          train_dict['y_col'].append(label)
  
  # Create dataframe
  train_df = pd.DataFrame.from_dict(train_dict)
  train_df['y_col'] = train_df['y_col'].astype(str)
  
  # Validation
  for dir in listdir(join(dir_path, 'valid')):
    if (isdir(join(dir_path, 'valid', dir))):
      # Extract corresponding label value
      for _, row in label_df.iterrows():
        if row[0].rsplit(".", 1)[0] == (dir.split("_")[0]):
          label = row[1]
      for mag in listdir(join(dir_path, 'valid', dir)):
        for img_file in listdir(join(dir_path, 'valid', dir, mag)):
          # Add the image path label pair to the validation dictionary
          img_path = join(dir_path, 'valid', dir, mag, img_file)
          validation_dict['x_col'].append(img_path)
          validation_dict['y_col'].append(label)
  
  # Create dataframe
  validation_df = pd.DataFrame.from_dict(validation_dict)
  validation_df['y_col'] = validation_df['y_col'].astype(str)

    
  # Test
  for dir in listdir(join(dir_path, 'test')):
    if (isdir(join(dir_path, 'test', dir))):
      # Extract corresponding label value
      for _, row in label_df.iterrows():
        if row[0].rsplit(".", 1)[0] == (dir.split("_")[0]):
          label = row[1]
      for mag in listdir(join(dir_path, 'test', dir)):
        for img_file in listdir(join(dir_path, 'test', dir, mag)):
          # Add the image path label pair to the test dictionary
          img_path = join(dir_path, 'test', dir, mag, img_file)
          test_dict['x_col'].append(img_path)
          test_dict['y_col'].append(label)
  
  # Create dataframe
  test_df = pd.DataFrame.from_dict(test_dict)
  test_df['y_col'] = test_df['y_col'].astype(str)

  
  return train_df, validation_df, test_df

def build_model():
  """Build the test model, from the ResNet50 foundation

  Returns:
      tf.keras.Model: the model to be used in transfer learning
  """
  model = Sequential()
  # 1st layer is the resnet base model (transfer learning)
  model.add(ResNet50(
    include_top=False, 
    pooling=POOLING, 
    weights='imagenet'))
  
  # 2nd layer as Dense for binary classification task
  model.add(Dense(1, activation = DENSE_LAYER_ACTIVATION))
  
  # Not training the resnet on the new data set. Using the pre-trained weigths
  model.layers[0].trainable = False 
  
  # Compile the model 
  opt = optimizers.Adam(
    learning_rate=0.005,
    name='Adam_transfer_learning',
    epsilon=1e-6
  )
  model.compile(
    optimizer = opt, 
    loss = OBJECTIVE_FUNCTION, 
    metrics = LOSS_METRICS) 
  
  return model

if __name__ == '__main__':
  # The argument to the script is the directory path to the data
  dir_path = sys.argv[1]
  train_df, validation_df, test_df = create_df(dir_path)
  print(train_df)
  print(validation_df)
  print(test_df)
  
  # data initialization and preprocessing
  data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1/255)
  
  # Train data
  train_generator = data_generator.flow_from_dataframe(
    train_df,
    directory=IMG_DIR,
    x_col='x_col',
    y_col='y_col',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
  )
  
  # Validation data
  validation_generator = data_generator.flow_from_dataframe(
    validation_df,
    directory=IMG_DIR,
    x_col='x_col',
    y_col='y_col',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
  )
  
  # Test data
  test_generator = data_generator.flow_from_dataframe(
    test_df,
    directory=IMG_DIR,
    x_col='x_col',
    y_col='y_col',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode='rgb',
    class_mode='binary',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
  )
  
  print("...All generators made!")
  
  model = build_model()
  model.summary()
  
  # Train the model
  try:
    fit_history = model.fit(
      train_generator,
      steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
      epochs = NUM_EPOCHS,
      validation_data=validation_generator,validation_steps=STEPS_PER_EPOCH_VALIDATION)
  except PIL.UnidentifiedImageError:
    print("Unidentified image")
  
  print("..trained model")