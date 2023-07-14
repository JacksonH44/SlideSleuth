#!/usr/bin/env python
# coding: utf-8

# Following the tutorial on imbalanced data by Tensorflow here: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

# In[1]:
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

import sklearn
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# In[3]:
# Create data generator
datagen = ImageDataGenerator(
  rescale=1/255,
  preprocessing_function=preprocess_input,
  rotation_range=40,
  horizontal_flip=True
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


# ### Define the model and the metrics
# Define a function that creates a simple transfer learning model for binary classification

# In[4]:
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

def make_model(metrics=METRICS, output_bias=None):
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
  
# ### Baseline Model
# Make and train the model using the function define above

# In[5]:
import math
EPOCHS = 50
BATCH_SIZE = 64
NUM_TRAINING_TILES = 32434
NUM_VALID_TILES = 3880
STEPS_PER_EPOCH = math.ceil(NUM_TRAINING_TILES / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(NUM_VALID_TILES / BATCH_SIZE)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True
)

model = make_model()
model.summary()

# Test run the model
# Set the correct initial bias
# In[6]:
pos = 28883
neg = 3551

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
# Confirm that the bias helps

# In[ ]:
# # Zero bias model
# model = make_model()
# model.load_weights(initial_weights)
# model.layers[-1].bias.assign([0.0])
# zero_bias_history = model.fit(
#     train_ds,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_data=valid_ds,
#     validation_steps=VALIDATION_STEPS,
#     verbose=1
# )

# # Bias model
# model = make_model()
# model.load_weights(initial_weights)
# careful_bias_history = model.fit(
#     train_ds,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_data=valid_ds,
#     validation_steps=VALIDATION_STEPS,
#     verbose=1
# )

colors = ['violet', 'mediumspringgreen']

def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values
    plt.semilogy(history.epoch, history.history['loss'], color=colors[n], label=f"Train {label}")
    plt.semilogy(history.epoch, history.history['val_loss'], color=colors[n], label=f"Val {label}", linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
# plot_loss(zero_bias_history, 'Zero Bias', 0)
# plot_loss(careful_bias_history, 'Careful Bias', 1)
# plt.close()


# ### Train the Model

# In[ ]:
model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[early_stopping],
    validation_data=valid_ds,
    validation_steps=VALIDATION_STEPS,
    class_weight=class_weight
)
model.save_weights('../model/weights/tl-2023-07-06_weights.h5')
# ### Check Training History
# Make a plot of the model's accuracy and loss on the training and validation set

# In[14]:


colors = ['violet', 'mediumspringgreen']
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
    plt.tight_layout()
    
plot_metrics(baseline_history)
plt.savefig('../img/plot_metrics.pdf')
plt.close()


# In[ ]:
# Evaluate metrics
train_predictions_baseline = model.predict(train_ds, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_ds, batch_size=BATCH_SIZE)


# In[ ]:
# Get training and testing labels
def extract_labels(ds):
  labels = []
  num_batches = len(ds)
  for batch_idx in range(num_batches):
    _, batch_labels = ds[batch_idx]
    labels.extend(batch_labels)
  
  labels = np.array(labels)
  return labels 

train_labels = extract_labels(train_ds)
test_labels = extract_labels(test_ds)

# In[ ]:
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100.5])
    plt.ylim([-0.5, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
plt.savefig('../img/roc_curve.png')
plt.close()