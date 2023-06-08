'''
  Function that allows a user to test if they have a GPU available

  Author: Jackson Howe
  Date Created: June 8, 2023
  Last Updated: June 8, 2023
'''

import tensorflow as tf
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")