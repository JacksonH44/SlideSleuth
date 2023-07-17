'''
  A function that generates feature vectors for all image tiles corresponding to each image in a dataset.

  Usage: python generate_features.py /path/to/root/of/all/image/tiles /path/to/desired/output/file
  Example: python generate_features.py /scratch/jhowe4/outputs/pipeline_test/2023-06-05/1 ../outputs/tmp_out

  The program deepzoom_tile.py writes its tiled slide image path in the following way:

  output_folder
    |
    +-- 001_files
    |     |
    |     + 5.0
    |       |
    |       +-- 10_20.jpeg
    |       ...
    |       +-- 30_20.jpeg
    ....
    +-- 098_files
    |     |
    |     + 5.0
    |       |
    |       +-- 10_20.jpeg
    |       ...
    |       +-- 30_20.jpeg

  So this program assumes such an output folder when searching for tiled images.

  Author: Jackson Howe
  Date Created: June 1, 2023
  Last Updated: June 2, 2023
'''

# imports
import sys
from os.path import exists, isdir
from os import listdir, makedirs
import extract_features
from shutil import rmtree

'''
  A function that takes in a root directory and a destination directory for all image slide tiles and writes
  each of them to a .csv file. For each whole slide image, one csv file will be created, with each row of the
  csv file representing a feature vector extracted from a tile of the whole slide image.
'''

def write_csv(src_dir, dest_dir):
  # Instantiate the model, take care to do this only once, as it is much faster than through each time in a
  model = extract_features.instantiate()

  for file in listdir(src_dir):
    cur_path = f"{src_dir}/{file}"
    if isdir(cur_path):
      # Strip away the '_files' part at the end and add 'features'
      basename = file.split('_')[0]
      basename = basename + '_features'
      for mag in listdir(cur_path):
        cur_path = f"{cur_path}/{mag}"

        dir_to_make = f'{dest_dir}/{basename}/{mag}'
        makedirs(dir_to_make)

        output_folder = f'{dir_to_make}/{basename}.csv'
        
        # Extract features from each image tile
        for img_tile in listdir(cur_path):
          extract_features.extract_features(model, f'{cur_path}/{img_tile}', output_folder)

if __name__ == '__main__':
  # Verify source directory
  try:
    src_dir = sys.argv[1]
    if not (exists(src_dir) and isdir(src_dir)):
      raise FileNotFoundError
  except IndexError:
    print("Missing source directory")
  except FileNotFoundError:
    print(f"The directory {src_dir} doesn't exist")

  # Verify destination directory
  try:
    dest_dir = sys.argv[2]
  except IndexError:
    print("Missing destination directory")

  # Extract feature vectors and write to a csv
  try:
    write_csv(src_dir, dest_dir)
  except KeyboardInterrupt:
    print('The program has been terminated.')
