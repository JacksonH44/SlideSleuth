'''
  A program that augments the pipeline specific to TCGA data and trains a 
  transfer learning model on that data generated from the pipeline.
  
  Author: Jackson Howe
  Date Created: June 24, 2023
  Last Updated: June 27, 2023
  '''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
from os import listdir, makedirs, remove
from os.path import join, isdir, exists
from shutil import move, rmtree
from PIL import Image
from PIL import ImageFile
import PIL

ImageFile.LOAD_TRUNCATED_IMAGES = True
# tf.compat.v1.disable_eager_execution()

LABEL = '/scratch/jhowe4/outputs/GDC/paad_example2/labels.csv'
    
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
      
      This function changes the directory structure to:
      root/
        train/
          [case_1]-[img_1]
          ...
          [case_n]-[img_m]
        valid/
          ...
        test/
          ...

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
            move(join(dir_path, file, mag, img_file), join(dir_path, str(label), f"{file.split('_')[0]}-{img_file}"))
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
        img = Image.open(join(dir_path, label, img_file))
      except PIL.UnidentifiedImageError:
        remove(join(dir_path, label, img_file))
        print(f"Removed file {join(dir_path, label, img_file)}")
      

if __name__ == '__main__':
  # Clean the datasets of corrupt tiled images
  # clean_datasets('/scratch/jhowe4/outputs/GDC/paad_example2/train')
  # clean_datasets('/scratch/jhowe4/outputs/GDC/paad_example2/test')
  # clean_datasets('/scratch/jhowe4/outputs/GDC/paad_example2/valid')

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
  
  # Build model
  model = tf.keras.Sequential()

  # Transfer learning
  model.add(tf.keras.applications.ResNet50(include_top = False, pooling = 'avg', weights ='imagenet'))
  
  # 2nd layer as Dense for 2-class classification
  model.add(tf.keras.layers.Dense(2, activation = 'sigmoid'))
  
  # Not training the resnet on the new data set. Using the pre-trained weigths
  model.layers[0].trainable = False  
  
  # compile the model
  model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = [tf.keras.metrics.AUC()])
  
  # Fit the model
  fit_history = model.fit(
    train_ds,
    steps_per_epoch=4,
    validation_data=valid_ds,
    validation_steps=4,
    epochs=3,
  )
  
  # Save the model weights
  model.save_weights('../model/tf-2023-06-28/weights.h5')
  
  # Evaluate model
  eval_result = model.evaluate(
    test_ds,
    steps=len(test_ds)
  )
  print("[test loss, test accuracy]:", eval_result)