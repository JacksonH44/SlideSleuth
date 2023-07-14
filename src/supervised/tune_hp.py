'''
  A script that is part of the data pipeline to find optimal hyperparameters.
  
  Accuracy: 
  
    Hyperparameter update: 
      Learning Rate: 0.005
      Best # of Epochs: 1
    
    So far best accuracy: 0.7865004
    So far best AUC: 0.7295095
  
  Author: Jackson Howe
  Date Created: June 27, 2023
  Last Updated: June 27, 2023
'''

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras
from os import makedirs
from os.path import isdir, dirname
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Have to install keras with a subprocess because for some reason I get a 
# ModuleNotFoundError otherwise (June 27, 2023)
import sys
import subprocess

# implement pip as a subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras-tuner'])
import keras_tuner as kt
print(f"keras-tuner version: {kt.__version__}")

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
def model_builder(hp):
  # Build model
  model = tf.keras.Sequential()

  # Transfer learning
  model.add(tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights ='imagenet'))
  
  # 2nd layer as Dense for 2-class classification
  model.add(tf.keras.layers.Dense(2, 'sigmoid'))
  
  # Not training the resnet on the new data set. Using the pre-trained weigths
  model.layers[0].trainable = False
  
  # Tune the learning rate for the optimizer
  # Choose and optimal value from 0.01, 0.001, 0.005, 0.0001, 0.0005
  hp_learning_rate = hp.Choice(
    'learning_rate', 
    values=[1e-2, 1e-3, 5e-3, 1e-4, 5e-4])  
  
  
  # compile the model
  model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), 
    loss='binary_crossentropy', 
    metrics = [tf.keras.metrics.AUC(), 'accuracy'])

  return model

if __name__ == '__main__':
  # Create data generator
  datagen = ImageDataGenerator(
    rescale=1/255,
    preprocessing_function=preprocess_input
  )
  
  # Training dataset
  train_ds = datagen.flow_from_directory(
    '/scratch/jhowe4/outputs/GDC/paad_example2/train',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
  )
  
  # Testing dataset
  test_ds = datagen.flow_from_directory(
    '/scratch/jhowe4/outputs/GDC/paad_example2/test',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
  )
  
  # Validation dataset
  valid_ds = datagen.flow_from_directory(
    '/scratch/jhowe4/outputs/GDC/paad_example2/valid',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
  )
  
  # Instanstiate the tuner
  tuner = kt.Hyperband(
    model_builder,
    objective=kt.Objective('val_auc', direction='max'),
    max_epochs=10,
    factor=3,
    directory='../outputs/hp',
    project_name='2023-06-29')
  
  # Create a callback to stop training early after reaching a certain value for 
  # the validatoin loss
  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
  
  # Run the hyperparameter search. The arguments for the search method are the 
  # same as those used in model.fit
  tuner.search(
    train_ds,
    steps_per_epoch=4,
    validation_data=valid_ds,
    validation_steps=4,
    epochs=3
  )
  
  # Get the optimal hyperparameters
  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
  
  print(f"""
  The hyperparameter seach is complete. The optimal learning rate for the 
  optimizer is {best_hps.get('learning_rate')}.
        """)
  
  # Build the model
  model = tuner.hypermodel.build(best_hps)
  
  # Fit the model to a max number of epochs
  fit_history = model.fit(
    train_ds,
    steps_per_epoch=4,
    validation_data=valid_ds,
    validation_steps=4,
    epochs=3
  )
  
  # Find the optimal number of epochs to train the model with the 
  # hyperparameters obtained from the search
  val_acc_per_epoch = fit_history.history['val_auc']
  best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
  print('Best epoch: %d' % (best_epoch,))
  
  hypermodel = tuner.hypermodel.build(best_hps)
  
  # Retrain the model
  hypermodel_fit_history = hypermodel.fit(
    train_ds,
    steps_per_epoch=4,
    validation_data=valid_ds,
    validation_steps=4,
    epochs=best_epoch
  )
  
  # Save the model weights
  weights_file = '../model/tf-2023-06-29/weights.h5'
  weights_dir = dirname(weights_file)
  
  if not isdir(weights_dir):
    makedirs(weights_dir)
  
  hypermodel.save_weights(weights_file)
  
  # Evaluate model performance
  eval_result = hypermodel.evaluate(
    test_ds,
    steps=len(test_ds)
  )
  print("[test loss, test AUC, test accuracy]:", eval_result)
  
  # Get predicted probabilities for the positve class
  y_pred_prob = hypermodel.predict(test_ds)
  
  # Get true labels
  y_true = np.concatenate([test_ds[i][1] for i in range(len(test_ds))])
  y_true_binary = label_binarize(y_true, classes=[0, 1])
  
  # Compute the false positive rate (fpr), true positive rate (tpr), and thresholds
  fpr, tpr, thresholds = roc_curve(y_true_binary.ravel(), y_pred_prob[:, 1].ravel())

  # Compute the Area Under the Curve (AUC)
  roc_auc = auc(fpr, tpr)

  # Plot the ROC curve
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic')
  plt.legend(loc="lower right")
  plt.savefig('../roc.pdf')
  
  # # Plot accuracy and loss
  # acc = hypermodel_fit_history.history['accuracy']
  # val_acc = hypermodel_fit_history.history['val_accuracy']
  # loss = hypermodel_fit_history.history['loss']
  # val_loss = hypermodel_fit_history.history['val_loss']
  
  # epochs = range(1, len(acc) + 1)

  # plt.figure()
  # plt.plot(epochs, acc, 'b', label='Training acc')
  # plt.plot(epochs, val_acc, 'bo', label='Validation acc')
  # plt.title('Training and validation accuracy')
  # plt.legend()

  # plt.savefig('../img/accuracy.pdf')

  # plt.figure()
  # plt.plot(epochs, loss, 'b', label='Training loss')
  # plt.plot(epochs, val_loss, 'bo', label='Validation loss')
  # plt.title('Training and validation loss')
  # plt.legend()

  # plt.savefig('../img/loss.pdf')