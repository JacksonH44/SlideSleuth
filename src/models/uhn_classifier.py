"""A program that trains and tests a resnet classifier for uhn data
  
  Date Created: July 20, 2023
  Last Updated: July 22, 2023
"""

__author__ = "Jackson Howe"

import math
import os
from os.path import join
from datetime import datetime
import pickle
import sys

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import ImageFile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from resnet_classifier_test import extract_labels

ImageFile.LOAD_TRUNCATED_IMAGES = True

DIR_PATH = "../../data/interim/HNE_images"
WEIGHTS_PATH = f'../../models/HNE_classifier-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

EPOCHS = 20
BATCH_SIZE = 64

strategy = tf.distribute.MirroredStrategy()
print(f"Number of available GPUs for distributed processing: {strategy.num_replicas_in_sync}")
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

NUM_TRAINING_TILES = 118955
NUM_INVASIVE_TRAINING = 53866
NUM_NONINVASIVE_TRAINING = 31949
NUM_UNDEFINED_TRAINING = 33140
NUM_VALID_TILES = 16165
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
    for layer in resnet.layers[:-15]:
      layer.trainable = False
  
    model.add(resnet)
  
    # Add an activation layer meant for binary classification with a kernel 
    # regularizer to prevent overfitting.
    model.add(tf.keras.layers.Dense(
      3, 
      activation = 'sigmoid', 
      bias_initializer=output_bias,
      kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    
    # Add a decaying learning rate over time to reduce overfitting.
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
      0.001,
      decay_steps=STEPS_PER_EPOCH*1000,
      decay_rate=1,
      staircase=False
    )
    
    # Compile the model.
    model.compile(
      optimizer=tf.keras.optimizers.Adam(lr_schedule),
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=metrics
    )
    
  return model

def plot_cm(labels, predictions, p=0.5):
  
  cm = confusion_matrix(labels.argmax(axis=1), predictions.argmax(axis=1) > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')


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
    brightness_range=[0.5, 1.5]
  )
  
  # Training dataset
  train_ds = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=GLOBAL_BATCH_SIZE
  )
  
  # Testing dataset
  test_ds = datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=GLOBAL_BATCH_SIZE
  )
  
  # Validation dataset
  valid_ds = datagen.flow_from_directory(
    valid_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=GLOBAL_BATCH_SIZE
  )
  
  # Calculate the class weights for the loss function of the classifier.
  invasive = NUM_INVASIVE_TRAINING
  noninvasive = NUM_NONINVASIVE_TRAINING
  undefined = NUM_UNDEFINED_TRAINING
  
  total = invasive + noninvasive + undefined
  weight_for_invasive = (1 / invasive) * (total / 3.0)
  weight_for_noninvasive = (1 / noninvasive) * (total / 3.0)
  weight_for_undefined = (1 / undefined) * (total / 3.0)

  class_weight = {0: weight_for_invasive, 1: weight_for_noninvasive, 2: weight_for_undefined}
  print('Weight for class invasive: {:.2f}' .format(weight_for_invasive))
  print('Weight for class noninvasive: {:.2f}' .format(weight_for_noninvasive))
  print('Weight for class undefined: {:.2f}' .format(weight_for_undefined))
  
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
  
  # Create the classifier
  model = make_model()
  
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
  save_model(model, WEIGHTS_PATH)
  
  # Save the model history to be used for plotting later, etc.
  with open(join(WEIGHTS_PATH, "history.pkl"), "wb") as file:
    pickle.dump(history.history, file)
  
  # Evaluate model
  eval_result = model.evaluate(
    test_ds,
    steps=len(test_ds)
  )
  
  # Plot the confusion matrix
  train_predictions_baseline = model.predict(train_ds, batch_size=BATCH_SIZE)
  test_predictions_baseline = model.predict(test_ds, batch_size=BATCH_SIZE)
  
  train_labels = extract_labels(train_ds)
  test_labels = extract_labels(test_ds)
  
  plot_cm(train_labels, train_predictions_baseline)
  plt.savefig('../../reports/figures/uhn_cm_train.png')
  plt.close()
  
  plot_cm(test_labels, test_predictions_baseline)
  plt.savefig('../../reports/figures/uhn_cm_test.png')
  plt.close()  