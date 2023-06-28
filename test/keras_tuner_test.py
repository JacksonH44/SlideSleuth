'''
  For some reason I kept getting a ModuleNotFoundError with the keras-tuner 
  library, so this is just a method of getting keras-tuner in the required 
  environment.
  
  Author: Jackson Howe
  Date Created: June 27, 2023
  Last Updated: June 28, 2023
'''

import tensorflow as tf
from tensorflow import keras
import sys
import subprocess

# implement pip as a subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'keras-tuner'])

import keras_tuner as kt
print(kt.__version__)