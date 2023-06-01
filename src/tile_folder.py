'''
  A program that calls deepzoom_tile.py on each image in a data set.

  Usage: python tile_folder.py /path/to/source /path/to/destination
  Example: python tile_folder.py ~/projects/def-sushant/jhowe4/DeepTumour/inputs/HNE

  Author: Jackson Howe
  Last Updated: June 1, 2023
'''

from optparse import OptionParse
import sys

'''
  A function that tiles all the whole slide images (WSIs) in a folder
'''

if __name__ == '__main__':
  # parser is to read command line arguments and also ssetup arguments or options for a command line script
  parser = OptionParser(usage='Usage: %prog [options] <slide>')

  parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
                      default=True, action='store_false', help='display entire scan area')
  parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
                      type='int', default=1, help='overlap of adjacent tiles [1]')
  parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
                      default='jpeg', help='image format for tiles [jpeg]')
  parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
                      type='int', default=4, help='number of worker processes to start [4]')
  parser.add_option('-o', '--output', metavar='NAME',
                      dest='basename', help='base name of output file')
  parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
                      type='int', default=90, help='JPEG compression quality [90]')
  parser.add_option('-r', '--viewer', dest='with_viewer',
                      action='store_true', help='generate directory tree with HTML viewer')
  parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
                      type='int', default=254, help='tile size [254]')
  parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg', type='float',
                      default=50, help='Max background threshold [50]; percentager of background allowed')
  parser.add_option('-x', '--xmlfile', metavar='NAME',
                      dest='xmlfile', help='xml file if needed')
  parser.add_option('-m', '--mask_type', metavar='COUNT', dest='mask_type', type='int',
                      default=1, help='if xml file is used, keep tile within the ROI (1) or outside of it (0)')
  parser.add_option('-R', '--ROIpc', metavar='PIXELS', dest='ROIpc', type='float', default=50,
                      help='To be used with xml file - minimum percentage of tile covered by ROI')

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
