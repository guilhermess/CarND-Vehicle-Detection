import numpy as np
import os
import pickle
from VehicleClassifier import VehicleClassifier
from scipy.ndimage.measurements import label
from LoggerCV import LoggerCV
from LoggerCV import CVRecord

import cv2

class VehicleDetectionPipeline:
  '''
  The VehicleDetectionPipeline class implements the framework for processing video images, detecting lane lines and
  marking them on the road image.
  '''

  def __init__(self,
               car_images, noncar_images,
               classifier,
               threshold,
               scale1,
               scale2,
               scale3,
               step1,
               step2,
               step3,
               logger):
    self.logger = logger
    self.car_classifier = VehicleClassifier(car_images, noncar_images, classifier, logger)
    self.previous_heatmaps = []
    self.weights = [10, 9, 8, 6, 4]
    self.scale1 = scale1
    self.scale2 = scale2
    self.scale3 = scale3
    self.step1 = step1
    self.step2 = step2
    self.step3 = step3
    self.threshold = threshold
    self.frame = 0


  def process_frame(self, image):
    '''
    This method receives a video frame (image) as a parameter and returns an image with cars marked with boxes
    :param image: image to detect cars
    :return: new image with cars marked
    '''
    windows = self.car_classifier.find_cars(image, 330, 650, self.scale1, self.step1, self.frame)
    if self.scale2 != 0 and self.scale2 != self.scale1:
      windows += self.car_classifier.find_cars(image, 330, 650, self.scale2, self.step2, self.frame)

    if self.scale3 != 0 and self.scale3 != self.scale1 and self.scale3 != self.scale2:
      windows += self.car_classifier.find_cars(image, 330, 650, self.scale3, self.step3, self.frame)

    new_heatmap = self.heatmap(image.shape[0:2], windows)
    new_heatmap[new_heatmap <= self.threshold] = 0
    self.previous_heatmaps.append(new_heatmap)
    hlength = min(len(self.previous_heatmaps), len(self.weights))
    heatmaps = np.dstack(self.previous_heatmaps[-1:-hlength-1:-1])
    weights = self.weights[0:hlength]
    heatmap = np.average(heatmaps, weights=weights, axis=2)


    self.debug_frame(new_heatmap, heatmap, image, windows)

    if self.logger.enabled(self.frame):
      self.logger.log(CVRecord("heatmap",
                               self.frame, [heatmap], 'opencv'))

    heatmap[heatmap <= self.threshold] = 0

    if self.logger.enabled(self.frame):
      self.logger.log(CVRecord("heatmap_threshold",
                               self.frame, [heatmap], 'opencv'))

    labels = label(heatmap)
    boxes = self.prune_labels(labels)
    result = self.draw_bboxes(image, boxes)

    self.frame +=1
    return result


  def heatmap(self, shape, windows):
    '''
    Create a heatmap with a specific shape, using the bounding boxes defined in windows to define hot regions
    :param shape: shape of the heatmap
    :param windows: bounding boxes used for computing the heatmap
    :return: heatmap array
    '''
    heatmap = np.zeros(shape, dtype=np.float32)
    for window in windows:
      xmin = window[0][0]
      ymin = window[0][1]
      xmax = window[1][0]
      ymax = window[1][1]
      heatmap[ymin:ymax, xmin:xmax] += 1
    return heatmap


  def prune_labels(self, labels):
    boxes = []
    for car_number in range(1, labels[1] + 1):
      # Find pixels with each car_number label value
      nonzero = (labels[0] == car_number).nonzero()
      # Identify x and y values of those pixels
      nonzeroy = np.array(nonzero[0])
      nonzerox = np.array(nonzero[1])
      # Define a bounding box based on min/max x and y
      bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
      area = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
      ymax = bbox[1][1]

      threshold_area = 40 * ymax - 15000
      if area > threshold_area:
        boxes.append(bbox)
    return boxes


  def draw_bboxes(self, img, boxes):
    for bbox in boxes:
      # Draw the box on the image
      cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255, 0.4), 6)
    # Return the image
    return img

