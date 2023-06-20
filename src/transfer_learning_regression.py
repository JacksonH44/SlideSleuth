'''
  A program that uses a transfer learning model to perform an 'image regression
  task', in which training data consists of an image and a real-valued score in
  [0, 1] with values closer to 1 indicating that the tumour is invasive, and 
  values closer to 0 indicating that the tumour is noninvasive.
  
  NOTE: (June 20, 2023) This file does not work! Working on multiple bug fixes 
  for this file. This note will be removed upon completion.
  
  Author: Jackson Howe
  Reference: https://github.com/Akshina07/DeepLearning-TumorHypoxia/blob/master/Training_Evaluation/resnet/train_resnet.py
  Date Created: June 16, 2023
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

LABELS = 'labels.csv'
IMAGE_RESIZE = 224
BATCH_SIZE = 32
NUM_TRAINING_TILES = 115963
NUM_VALID_TILES = 13942
NUM_TEST_TILES = 15195
CHANNELS = 3
DENSE_LAYER_ACTIVATION = 'softmax'
POOLING = 'avg'
OBJECTIVE_FUNCTION = 'mean_squared_error'
LOSS_METRICS = ['mean_squared_error']

# Training global variables
NUM_EPOCHS = 4
BATCH_SIZE_TRAINING = 200
BATCH_SIZE_VALIDATION = 200
STEPS_PER_EPOCH_TRAINING = math.ceil(NUM_TRAINING_TILES/BATCH_SIZE_TRAINING)
STEPS_PER_EPOCH_VALIDATION = math.ceil(NUM_VALID_TILES/BATCH_SIZE_VALIDATION)

def create_df(dir_path):
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
      label = label_df[label_df['case'] == (dir.split("_")[0]).lstrip('0')]['score'].values[0]
      for mag in listdir(join(dir_path, 'train', dir)):
        for img_file in listdir(join(dir_path, 'train', dir, mag)):
          # Add the image path label pair to the training dictionary
          img_path = join(dir_path, 'train', dir, mag, img_file)
          train_dict['x_col'].append(img_path)
          train_dict['y_col'].append(label)
  
  # Create dataframe
  train_df = pd.DataFrame.from_dict(train_dict)
  
  # Validation
  for dir in listdir(join(dir_path, 'valid')):
    if (isdir(join(dir_path, 'valid', dir))):
      # Extract corresponding label value
      label = label_df[label_df['case'] == (dir.split("_")[0]).lstrip('0')]['score'].values[0]
      for mag in listdir(join(dir_path, 'valid', dir)):
        for img_file in listdir(join(dir_path, 'valid', dir, mag)):
          # Add the image path label pair to the validation dictionary
          img_path = join(dir_path, 'valid', dir, mag, img_file)
          validation_dict['x_col'].append(img_path)
          validation_dict['y_col'].append(label)
  
  # Create dataframe
  validation_df = pd.DataFrame.from_dict(validation_dict)
    
  # Test
  for dir in listdir(join(dir_path, 'test')):
    if (isdir(join(dir_path, 'test', dir))):
      # Extract corresponding label value
      label = label_df[label_df['case'] == (dir.split("_")[0]).lstrip('0')]['score'].values[0]
      for mag in listdir(join(dir_path, 'test', dir)):
        for img_file in listdir(join(dir_path, 'test', dir, mag)):
          # Add the image path label pair to the test dictionary
          img_path = join(dir_path, 'test', dir, mag, img_file)
          test_dict['x_col'].append(img_path)
          test_dict['y_col'].append(label)
  
  # Create dataframe
  test_df = pd.DataFrame.from_dict(test_dict)
  
  return train_df, validation_df, test_df

def build_model():
  model = Sequential()
  # 1st layer is the resnet base model (transfer learning)
  model.add(ResNet50(
    include_top=False, 
    pooling=POOLING, 
    weights='imagenet'))
  
  # 2nd layer as Dense for regression task
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
  # data initialization and preprocessing
  data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1/255)
  
  # Train data
  train_generator = data_generator.flow_from_dataframe(
    train_df,
    x_col='x_col',
    y_col='y_col',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode='rgb',
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
  )
  
  # Validation data
  validation_generator = data_generator.flow_from_dataframe(
    validation_df,
    x_col='x_col',
    y_col='y_col',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode='rgb',
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
  )
  
  # Test data
  test_generator = data_generator.flow_from_dataframe(
    test_df,
    x_col='x_col',
    y_col='y_col',
    target_size=(IMAGE_RESIZE, IMAGE_RESIZE),
    color_mode='rgb',
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validate_filenames=True
  )
  
  model = build_model()
  
  # Train the model
  try:
    fit_history = model.fit_generator(
      train_generator,
      steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
      epochs = NUM_EPOCHS,
      validation_data=validation_generator,validation_steps=STEPS_PER_EPOCH_VALIDATION)
  except PIL.UnidentifiedImageError:
    pass
  
  print("..trained model")

  model_json = model.to_json()
  with open("../model/resnet5x.json", "w") as json_file:
    json_file.write(model_json)

  # serialize weights to HDF5
  model.save_weights("../model/resnet5x_weights.h5")
  print("Saved model to disk")
  
  # PLOTS THE MODEL STATISTICS ( TRAINING VS VALIDATION LOSS AND ACCURACY )
  plt.figure(1, figsize = (15,8)) 
  plt.subplot(221)  
  plt.plot(fit_history.history['acc'])  
  plt.plot(fit_history.history['val_acc'])  
  plt.title('model accuracy')  
  plt.ylabel('accuracy')  
  plt.xlabel('epoch')  
  plt.legend(['train', 'valid']) 
    
  plt.subplot(222)  
  plt.plot(fit_history.history['loss'])  
  plt.plot(fit_history.history['val_loss'])  
  plt.title('model loss')  
  plt.ylabel('loss')  
  plt.xlabel('epoch')  
  plt.legend(['train', 'valid']) 

  plt.show()
  plt.savefig("../model/resnet5x_june192023.jpg")