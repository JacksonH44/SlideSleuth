"""A script that visualzes the performance of a binary classifier

Date Created: July 27, 2023
Last Updated: July 27, 2023
"""
  
__author__ = "Jackson Howe"

from datetime import datetime
import sys
from os import makedirs
from os.path import exists, join

sys.path.insert(0, 'home/jhowe4/projects/def-sushant/jhowe4/SlideSleuth/src/models')

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import numpy as np

from tcga_classifier import make_model

unique_id = datetime.now().strftime("%Y%m%d-%H%M%S")

DIR_PATH = "/scratch/jhowe4/outputs/GDC/paad_example2"
WEIGHTS_PATH = '../../models/tcga-paad_classifier-20230720-050820/weights-48.h5'
FIG_PATH = f'../../reports/figures/tcga_classifier-20230720-050820'

if not exists(FIG_PATH):
  makedirs(FIG_PATH)

BATCH_SIZE = 64

strategy = tf.distribute.MirroredStrategy()
print(f"Number of available GPUs for distributed processing: {strategy.num_replicas_in_sync}")
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

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
  
  # Define the model architecture.
  model = make_model()
  
  # Load the previous best weights from training into the model.
  model.load_weights(WEIGHTS_PATH)

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
  plt.savefig(join(FIG_PATH, 'train_roc_curve.png'))
  plt.close()
  plot_roc(
    "Test Baseline", 
    test_labels, 
    test_predictions_baseline, 
    color='deepskyblue', 
    linestyle='--'
  )
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'test_roc_curve.png'))
  plt.close()
  
  plot_prc(
    train_labels,
    train_predictions_baseline,
    color='deepskyblue'
  )
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'train_prc_curve.png'))
  plt.close()
  plot_prc(
    test_labels, 
    test_predictions_baseline, 
    color='deepskyblue', 
    linestyle='--'
  )
  plt.legend(loc='lower right')
  plt.savefig(join(FIG_PATH, 'test_roc_curve.png'))
  plt.close()