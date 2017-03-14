
import os.path
from collections import namedtuple
import cv2
import numpy as np
import shutil

CVRecord = namedtuple('CVRecord', ['name', 'frame', 'figures', 'imtype'])

class LoggerCV:
  def __init__(self, enabled, log_rate):
    self.__enabled = enabled
    self.log_rate = log_rate
    self.records = []

  def enabled(self, frame_count):
    return self.__enabled and frame_count % self.log_rate == 0


  def log_rate(self):
    return self.log_rate

  def log(self, record):
    self.records.append(record)


  def write_records(self, output_dir):
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for record in self.records:
      basename = output_dir + '/' + record.name + 'f' + str(record.frame)
      for image in record.figures:
        filename = basename + '.png'
        if ( record.imtype == 'figure'):
          image.savefig(filename)
        elif (record.imtype == 'opencv'):
          if ( len(image.shape) == 3):
            if not os.path.exists(filename):
              cv2.imwrite(filename, cv2.cvtColor(image.astype(np.float32)*255, cv2.COLOR_RGB2BGR))
            else:
              count = 0
              while os.path.exists(filename):
                filename = basename + 'c' + str(count) + '.jpg'
                count += 1
              cv2.imwrite(filename, cv2.cvtColor(image.astype(np.float32)*255, cv2.COLOR_RGB2BGR))
          else:
            if not os.path.exists(filename):
              cv2.imwrite(filename, image)
            else:
              count = 0
              while os.path.exists(filename):
                filename = basename + 'c' + str(count) + '.jpg'
                count += 1
              cv2.imwrite(filename, image)








