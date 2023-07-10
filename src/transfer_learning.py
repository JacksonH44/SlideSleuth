'''
  A program that augments the pipeline specific to TCGA data and trains a 
  transfer learning model on that data generated from the pipeline.
  
  Author: Jackson Howe
  Date Created: June 24, 2023
  Last Updated: July 7, 2023
  '''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
import numpy as np
from os import listdir, makedirs, remove
from os.path import join, isdir, exists
from shutil import move, rmtree
from PIL import Image
from PIL import ImageFile
import PIL
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import math
import os
import tempfile
from datetime import datetime
import pickle
from optparse import OptionParser
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True
# tf.compat.v1.disable_eager_execution()

LABEL = '/scratch/jhowe4/outputs/GDC/brca_example_10x/labels.csv'
ERR_FILE = '/scratch/jhowe4/outputs/GDC/brca_example_10x/corrupt_images_log.txt'
WEIGHTS_PATH = '../model/weights'

EPOCHS = 5
BATCH_SIZE = 2048
NUM_TRAINING_TILES = 203110
NUM_POS_TRAINING = 175410
NUM_NEG_TRAINING = 27700
NUM_VALID_TILES = 15855
STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / BATCH_SIZE)

METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.AUC(name='prc', curve='PR') # Precision-recall curve
]

def is_opt_provided(parser, dest):
  """Determines if an argument to the option parser is provided

  Args:
      parser (OptionParser): Parser
      dest (str): Option destination of interest
  """
  
  if any (opt.dest == dest and (opt._long_opts[0] in sys.argv[1:] or opt._short_opts[0] in sys.argv[1:]) for opt in parser._get_all_options()):
    return True
  return False  

def save_model(model, save_dir):
  """A function that makes the directory in which the weights will be stored in 
  and writes it to a unique file

  Args:
      model (tf.keras.models.Model): The model you wish to save
      weights_dir (str): The path to the directory in which the weights will be 
      saved
  """
  
  # Make the folder if it doesn't exist
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
  # Get a unique file name
  now = datetime.now()
  date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
  
  # Save model weights
  model.save_weights(os.path.join(save_dir, f'weights_{date_time}.h5'))
  
  # Save model parameters
  parameters = [
      BATCH_SIZE,
      EPOCHS
    ]
    
  # Dump into a pickle file
  save_path = os.path.join(save_dir, f"parameters_{date_time}.pkl")
  with open(save_path, "wb") as f:
    pickle.dump(parameters, f)
    
def organize_dir(dir_path):
  """the result of tile_[type].sh is the following directory structure:
     root/
      train/
        [case_1]_files/
          5.0/
            [img_1]
            ...
            [img_m]
          20.0/
          ...
        ...
        [case_n]_files/
          ...
      test/
      valid/
      
      This function changes the directory structure from:
      root/
        [case_1]_files/
          5.0/
            [img_1]
            ...
            [img_m]
        [case_n]_files/
          ...
      
      to:
      root/
        [case_1]-[img_1]
        ...
        [case_n]-[img_m]

  Args:
      dir_path (String): Path to the root folder of the dataset directory
  """
  
  # Read in the labels and create a new subdirectory under the root for each 
  # class
  label_df = pd.read_csv(LABEL)
  classes = label_df['class'].unique()
  for c in classes:
    if not exists(join(dir_path, str(c))):
      makedirs(join(dir_path, str(c)))
    
  for file in listdir(dir_path):
    if isdir(join(dir_path, file)):
      try:
        # Get the label associate with each whole slide image
        label = label_df[label_df['file'] == f"{((file.split('_'))[0])}.svs"]['class'].values[0]
        for mag in listdir(join(dir_path, file)):
          for img_file in listdir(join(dir_path, file, mag)):
            # Move the file from the deep subdirectory structure outlined above 
            # to either the train, test, or valid folder, identify which WSI it 
            # came from by prepending the name of the WSI before the image name
            move(join(dir_path, file, mag, img_file), join(dir_path, str(label), f"{file.split('_')[0]}-{mag}-{img_file}"))
      except IndexError:
        # TODO: in the TCGA-PAAD dataset, there is one file that when I call 
        # splitData.R to assign it a label, it ends up in the dataframe, but 
        # apparently it is not in the actual folder. For now, we just don't 
        # include it in the data 
        print(f"{file} has no corresponding label")
        
  # Delete and clean up the excess files and now empty directories
  for file in listdir(dir_path):
    file_is_class = False
    for c in classes:
      if str(c) == file:
        file_is_class = True
        
    if not file_is_class:
      if isdir(join(dir_path, file)):
        rmtree(join(dir_path, file))
      else:
        try:
          remove(join(dir_path, file))
        except FileNotFoundError as f:
          print(f"Failed to delete {join(dir_path, file)}: {str(f)}")

def clean_datasets(dir_path):
  """A function that verifies all images are not corrupted, and removes corrupt 
  images

  Args:
      dir_path (String): path to directory that needs to be cleaned
  """
  
  for label in listdir(dir_path):
    for img_file in listdir(join(dir_path, label)):
      try:
        _ = Image.open(join(dir_path, label, img_file))
      except PIL.UnidentifiedImageError:
        remove(join(dir_path, label, img_file))
        with open(ERR_FILE, 'a') as err_file:
          err_file.write(f"Removed fie {join(dir_path, label, img_file)} because it was unreadable")
          
def make_model(metrics=METRICS, output_bias=None):
  """A functiont that makes a transfer learning model

  Args:
      metrics (list<tf.keras.metrics>, optional): A list of metrics for 
      tensorflow to track during training. Defaults to METRICS.
      output_bias (np.array, optional): Output bias to put on the last layer (1 
      neuron dense layer). Defaults to None.

  Returns:
      tf.keras.models.Model: The built tensorflow model
  """
  
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = tf.keras.Sequential()
    
  # Add ResNet50 model
  model.add(tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights ='imagenet'))
    
  # Add dense layers
  model.add(tf.keras.layers.Dense(1024, activation='relu'))
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dense(32, activation='relu'))
  
  # Add outer classification layer to model
  model.add(tf.keras.layers.Dense(1, activation = 'sigmoid', bias_initializer=output_bias))
  
  # Not training the resnet on the new data set. Using the pre-trained weigths
  model.layers[0].trainable = False 
    
  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=metrics
  )
    
  return model
          
def extract_labels(ds):
  """A function that gets the labels for each observation in the dataset. Use 
  this function when you get need the labels of observations that are in a 
  dataset returned from the flow_from_directory method of the 
  ImageDataGenerator class

  Args:
      ds (tf.data.Dataset): The dataset you want to extract labels from

  Returns:
      np.array: Vector of true dataset labels
  """
  
  labels = []
  num_batches = len(ds)
  for batch_idx in range(num_batches):
    _, batch_labels = ds[batch_idx]
    labels.extend(batch_labels)
  
  labels = np.array(labels)
  return labels 

def plot_roc(name, labels, predictions, **kwargs):
  """Plots ROC curve based off false and true positive frequencies. From https://www.tensorflow.org/tutorials/structured_data/imbalanced_data.
  
  Args:
      name (str): Label name
      labels (np.array): Ground truth 
      predictions (np.array): Trained model predictions
  """
  
  fp, tp, _ = roc_curve(labels, predictions)
    
  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5, 100.5])
  plt.ylim([-0.5, 100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

def plot_metrics(history):
  """Plot a variety of model metrics. From https://www.tensorflow.org/tutorials/
  structured_data/imbalanced_data.

  Args:
      history (tf.keras.History): Trained model history
  """
  
  metrics = ['loss', 'prc', 'precision', 'recall']
  
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    
    plt.subplot(2,2,n+1)
    plt.plot(
      history.epoch, 
      history.history[metric], 
      color='lightcoral', 
      label='Train'
    )
    plt.plot(
      history.epoch, 
      history.history['val_'+metric],
      color='lightcoral', 
      linestyle="--", 
      label='Val'
    )
    
    plt.xlabel('Epoch')
    plt.ylabel(name)
    
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.6,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    plt.tight_layout()

if __name__ == '__main__':
  # User can specify if they want to preprocess their data, or if its already processed into train/, test/, valid/, and 0/, 1/ within each
  parser = OptionParser(usage='Usage: %prog options')
  parser.add_option(
    '-P',
    '--preprocess',
    dest='preprocessing directive',
    default=False,
    help='preprocess image data before running transfer learning'
  )
  
  # Make sure the error file is clean
  if exists(ERR_FILE):
    remove(ERR_FILE)
    
  dir_path = '/scratch/jhowe4/outputs/GDC/brca_example_10x'
  train_path = join(dir_path, 'train')
  test_path = join(dir_path, 'test')
  valid_path = join(dir_path, 'valid')
  
  if is_opt_provided(parser, 'preprocessing directive'):
    # Organize the data directory
    organize_dir(train_path)
    organize_dir(test_path)
    organize_dir(valid_path)
  
    # Clean each of its subfolders
    clean_datasets(train_path)
    clean_datasets(test_path)
    clean_datasets(valid_path)

  # Create data generator
  datagen = ImageDataGenerator(
    rescale=1/255,
    preprocessing_function=preprocess_input
  )
  
  # Training dataset
  train_ds = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=BATCH_SIZE
  )
  
  # Testing dataset
  test_ds = datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=BATCH_SIZE
  )
  
  # Validation dataset
  valid_ds = datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=BATCH_SIZE
  )
  
  # Total number of positive and negative samples in the training dataset
  pos = NUM_POS_TRAINING
  neg = NUM_NEG_TRAINING
  
  total = pos + neg
  print(f"Number of Positive Samples: {pos}\nNumber of Negative Samples: {neg}")
  weight_for_0 = (1 / neg) * (total / 2.0)
  weight_for_1 = (1 / pos) * (total / 2.0)

  class_weight = {0: weight_for_0, 1: weight_for_1}
  print('Weight for class 0: {:.2f}' .format(weight_for_0))
  print('Weight for class 1: {:.2f}' .format(weight_for_1))

  initial_bias = np.log([pos / neg])

  # Checkpoint the initial weights
  model = make_model(output_bias=initial_bias)
  initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
  model.save_weights(initial_weights)
  
  # Create a callback to stop when the model converges
  early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
  )
  
  # Fit the model
  history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stopping],
    class_weight=class_weight
  )
  
  # Save the model weights
  save_model(model, WEIGHTS_PATH)
  
  # Evaluate model
  eval_result = model.evaluate(
    test_ds,
    steps=len(test_ds)
  )
  
  # Plot the ROC curve
  train_predictions_baseline = model.predict(train_ds, batch_size=BATCH_SIZE)
  test_predictions_baseline = model.predict(test_ds, batch_size=BATCH_SIZE)
  
  train_labels = extract_labels(train_ds)
  test_labels = extract_labels(test_ds)
    
  plot_roc(
    "Train Baseline", 
    train_labels, 
    train_predictions_baseline, 
    color='deepskyblue'
  )
  plot_roc(
    "Test Baseline", 
    test_labels, 
    test_predictions_baseline, 
    color='deepskyblue', 
    linestyle='--'
  )
  plt.legend(loc='lower right')
  plt.savefig('../img/roc_curve.png')
  plt.close()
  
  # Plot other model metrics
  plot_metrics(history)
  plt.savefig('../img/plot_metrics.pdf')
  plt.close()