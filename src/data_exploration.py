import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import ImageFile
import math

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
EPOCHS = 25
BATCH_SIZE = 256
NUM_TRAINING_TILES = 32434
NUM_VALID_TILES = 3880
STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / BATCH_SIZE)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Check for GPU
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
colors = ['violet', 'mediumspringgreen']

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = tf.keras.Sequential()
    
    # Add ResNet50 model
    model.add(tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights ='imagenet'))
  
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
  labels = []
  num_batches = len(ds)
  for batch_idx in range(num_batches):
    _, batch_labels = ds[batch_idx]
    labels.extend(batch_labels)
  
  labels = np.array(labels)
  return labels 

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    plt.savefig('../img/plot_metrics.pdf')
    plt.close()
    
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
  
  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5, 20])
  plt.ylim([80, 100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

# Create data generator
datagen = ImageDataGenerator(
  rescale=1/255,
  preprocessing_function=preprocess_input
)
  
# Training dataset
train_ds = datagen.flow_from_directory(
  '/scratch/jhowe4/outputs/GDC/paad_example2/train',
  target_size=(224, 224),
  class_mode='binary',
  batch_size=32
)
  
# Testing dataset
test_ds = datagen.flow_from_directory(
  '/scratch/jhowe4/outputs/GDC/paad_example2/test',
  target_size=(224, 224),
  class_mode='binary',
  batch_size=32
)
  
# Validation dataset
valid_ds = datagen.flow_from_directory(
  '/scratch/jhowe4/outputs/GDC/paad_example2/valid',
  target_size=(224, 224),
  class_mode='binary',
  batch_size=32
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

model = make_model()
model.summary()

# Find the prevalence of each class in the test dataset
pos = 0
neg = 0
num_batches = len(train_ds)
for batch_idx in range(num_batches):
    _, batch_labels = train_ds[batch_idx]
    for label in batch_labels:
        if label == 1:
            pos = pos + 1
        elif label == 0:
            neg = neg + 1

total = pos + neg
print(f"Number of Positive Samples: {pos}\nNumber of Negative Samples: {neg}")
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}
print('Weight for class 0: {:.2f}' .format(weight_for_0))
print('Weight for class 1: {:.2f}' .format(weight_for_1))

initial_bias = np.log([pos / neg])
initial_bias

# Checkpoint the initial weights
model = make_model(output_bias=initial_bias)
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

# Make the weighted model
weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
  train_ds,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  callbacks=[early_stopping],
  validation_data=valid_ds,
  class_weight=class_weight
)

weighted_model.save_weights('../model/weights/tf-2023-07-05_weights.h5')
    
plot_metrics(weighted_history)

train_predictions_baseline = weighted_model.predict(
  train_ds, 
  batch_size=BATCH_SIZE
)
test_predictions_baseline = weighted_model.predict(
  test_ds, 
  batch_size=BATCH_SIZE
)

train_labels = extract_labels(train_ds)
test_labels = extract_labels(test_ds)

plot_roc(
  "train_baseline", 
  train_labels, 
  train_predictions_baseline, 
  color=colors[0]
)
plot_roc(
  "test_baseline", 
  test_labels, 
  test_predictions_baseline, 
  color=colors[0],
  linestyle="--"
)
plt.legend(loc="lower right")
plt.savefig('../img/roc_curve.pdf')
plt.close()