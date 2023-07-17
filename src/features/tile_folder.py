'''
  A program that calls deepzoom_tile.py on each image in a data set.

  Usage: python tile_folder.py opts /path/to/source
  Example: python tile_folder.py -s 229 -e 0 -j 32 -B 50 --output=/path/to/output ~/projects/def-sushant/jhowe4/TissueTango/inputs/HNE

  Author: Jackson Howe
  Last Updated: June 1, 2023
'''

from optparse import OptionParser
from glob import glob
from os.path import basename, splitext, join
from os import listdir
import deepzoom_tile

'''
  A function that tiles all the whole slide images (WSIs) in a folder
'''

if __name__ == '__main__':
  # Build parse to read options and also take in any command line arguments
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

  # Get the path to the slides directory
  (opts, args) = parser.parse_args()
  try:
    slidepath = args[0]
  except IndexError:
    parser.error('Missing slide argument')

  # Fill in rest of required file data
  if opts.xmlfile is None:
    opts.xmlfile = ''

  for file in listdir(slidepath):
    if (file[0] != "."):
      files = glob(slidepath)
      for imgNb in range(len(files)):
        filename = f"{files[imgNb]}/{file}"
        opts.basenameJPG = splitext(basename(filename))[0]
        print("processing: " + opts.basenameJPG)

        # appends the image file name to the output folder path
        output = join(opts.basename, opts.basenameJPG)
        print(f"Processing {filename} that is going to {output}")

        deepzoom_tile.DeepZoomStaticTiler(filename, 
                        output, 
                        opts.format, 
                        opts.tile_size, 
                        opts.overlap, 
                        opts.limit_bounds, 
                        opts.quality,
                        opts.workers, 
                        opts.with_viewer, 
                        opts.Bkg, 
                        opts.basenameJPG, 
                        opts.xmlfile, 
                        opts.mask_type, 
                        opts.ROIpc).run()
