"""A program that trains a ResNet50 fine-tuned network on uhn with binary labels
  
  Date Created: August 1, 2023
  Last Updated: August 1, 2023
"""

__author__ = "Jackson Howe"

import math
import os
from os.path import join
import tempfile
from datetime import datetime
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
import numpy as np
from PIL import ImageFile
from sklearn.metrics import roc_curve, precision_recall_curve, auc

ImageFile.LOAD_TRUNCATED_IMAGES = True

unique_id = datetime.now().strftime("%Y%m%d-%H%M%S")

DIR_PATH = "/scratch/jhowe4/outputs/uhn/CK7_images"
WEIGHTS_PATH = f'../../models/uhn_binary_classifier-{unique_id}'
FIG_PATH = f'../../reports/figures/uhn_binary_classifier-{unique_id}'

EPOCHS = 15
BATCH_SIZE = 64

strategy = tf.distribute.MirroredStrategy()
print(f"Number of available GPUs for distributed processing: {strategy.num_replicas_in_sync}")
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

NUM_TRAINING_TILES = 112081
NUM_POS_TRAINING = 37913
NUM_NEG_TRAINING = 74168
NUM_VALID_TILES = 27052
STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / GLOBAL_BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / GLOBAL_BATCH_SIZE)

with strategy.scope():
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

def save_model(save_dir):
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
  
  # Save model parameters
  parameters = [
      BATCH_SIZE,
      EPOCHS
    ]
    
  # Dump into a pickle file
  save_path = os.path.join(save_dir, f"parameters.pkl")
  with open(save_path, "wb") as f:
    pickle.dump(parameters, f)
          
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
    
  # Use distributed computing across multiple GPUs.
  with strategy.scope():
    model = tf.keras.Sequential()
    
    # Create a ResNet50 layer with the last 15 layers unfrozen
    resnet = tf.keras.applications.ResNet50(
      include_top=False,
      pooling='avg',
      weights='imagenet'
    )
    # If model training is too long, change the number of layers to train to a smaller number
    for layer in resnet.layers[:-15]:
      layer.trainable = False
  
    model.add(resnet)
  
    # Add an activation layer meant for binary classification with a kernel 
    # regularizer to prevent overfitting.
    model.add(tf.keras.layers.Dense(
      1, 
      activation = 'sigmoid', 
      bias_initializer=output_bias,
      kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    
    # Add a decaying learning rate over time to reduce overfitting.
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.00005,
      decay_steps=STEPS_PER_EPOCH*1000,
      decay_rate=1,
      staircase=False
    )
    
    # Compile the model.
    model.compile(
      optimizer=tf.keras.optimizers.Adam(lr_schedule),
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

def plot_roc(_, labels, predictions, **kwargs):
  """Plots ROC curve based off false and true positive frequencies. From https://www.tensorflow.org/tutorials/structured_data/imbalanced_data.
  
  Args:
      name (str): Label name
      labels (np.array): Ground truth 
      predictions (np.array): Trained model predictions
  """
  
  fp, tp, _ = roc_curve(labels, predictions)
  
  # Calculate AUC using scikit-learn
  auc_score = auc(fp, tp)
    
  plt.plot(100*fp, 100*tp, label="ROC curve (area = %0.2f)" % auc_score, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5, 100.5])
  plt.ylim([-0.5, 100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')
  
def plot_prc(labels, predictions, **kwargs):
  precision, recall, _ = precision_recall_curve(labels, predictions)
  
  plt.plot(precision, recall, linewidth=2, **kwargs)
  plt.xlabel('Precision')
  plt.xlabel('Recall')
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')


if __name__ == '__main__':
  dir_path = DIR_PATH
  train_path = join(dir_path, 'train')
  test_path = join(dir_path, 'test')
  valid_path = join(dir_path, 'valid')

  # Create data generator
  datagen = ImageDataGenerator(
    rescale=1/255,
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    height_shift_range=0.2,
    shear_range=20
  )
  
  # Training dataset
  train_ds = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=GLOBAL_BATCH_SIZE
  )
  
  # Testing dataset
  test_ds = datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=GLOBAL_BATCH_SIZE
  )
  
  # Validation dataset
  valid_ds = datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=GLOBAL_BATCH_SIZE
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
    monitor='val_loss',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
  )
  
  # Create a checkpoint callback after each epoch trained
  checkpoint_path = os.path.join(WEIGHTS_PATH, 'weights-{epoch:02d}.h5')
    
  model_cp = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor='val_loss',
    verbose=1,
    save_freq='epoch',
    save_weights_only=True,
    save_best_only=True
  )
  
  model.summary()

  # Fit the model
  history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stopping, model_cp],
    class_weight=class_weight
  )
  
  # Save the model weights
  save_model(WEIGHTS_PATH)
  
  # Save the model history to be used for plotting later, etc.
  with open(join(WEIGHTS_PATH, "history.pkl"), "wb") as file:
    pickle.dump(history.history, file)
  
  # Evaluate model
  eval_result = model.evaluate(
    test_ds,
    steps=len(test_ds)
  )
  
  # Plot the ROC and PRC curves.
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
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'train_roc_curve.svg'), format='svg')
  plt.close()
  plot_roc(
    "Test Baseline", 
    test_labels, 
    test_predictions_baseline, 
    color='deepskyblue', 
    linestyle='--'
  )
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'test_roc_curve.svg'), format='svg')
  plt.close()
  
  plot_prc(
    train_labels,
    train_predictions_baseline,
    color='deepskyblue'
  )
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'train_prc_curve.svg'), format='svg')
  plt.close()
  plot_prc(
    test_labels, 
    test_predictions_baseline, 
    color='deepskyblue', 
    linestyle='--'
  )
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'test_prc_curve.svg'), format='svg')
  plt.close()
  