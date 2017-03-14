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
               spatial, hoc, hog,
               threshold_low, threshold_high,
               debug_video,
               logger):
    self.logger = logger
    self.car_classifier = VehicleClassifier(car_images, noncar_images, classifier, logger,
                                            features_spatial=spatial, features_histogram_of_colors=hoc,
                                            features_histogram_of_gradients=hog)
    self.previous_heatmaps = []
    self.weights = [10, 9, 8, 6, 4]
    self.threshold_low = threshold_low
    self.threshold_high = threshold_high
    self.frame = 0
    self.debug_video = debug_video


  def process_frame(self, image):
    '''
    This method receives a video frame (image) as a parameter and returns an image with cars marked with boxes
    :param image: image to detect cars
    :return: new image with cars marked
    '''

    windows = self.car_classifier.find_cars(image, 370, 480, 1.1, 1, self.frame)
    windows = self.car_classifier.find_cars(image, 400, 500, 1.2, 1, self.frame)
    windows += self.car_classifier.find_cars(image, 400, 520, 1.3, 1, self.frame)
    windows += self.car_classifier.find_cars(image, 420, 650, 1.4, 1, self.frame)


    heatmap = self.heatmap(image.shape[0:2], windows)

    self.previous_heatmaps.append(heatmap)
    hlength = min(len(self.previous_heatmaps), len(self.weights))
    heatmaps = np.dstack(self.previous_heatmaps[-1:-hlength - 1:-1]).astype(np.float32)
    weights = self.weights[0:hlength]
    weighted_heatmap = np.average(heatmaps, weights=weights, axis=2)

    weighted_heatmap[weighted_heatmap <= self.threshold_low] = 0

    pre_threshold_labels = label(weighted_heatmap)

    weighted_heatmap[weighted_heatmap <= self.threshold_high] = 0

    threshold_labels = label(weighted_heatmap)

    if self.logger.enabled(self.frame):
      self.logger.log(CVRecord("heatmap_threshold", self.frame, [heatmap], 'opencv'))

    boxes = self.prune_labels(pre_threshold_labels, threshold_labels)
    result = self.draw_bboxes(image, boxes)

    self.frame += 1

    if self.debug_video:
      boxes_and_result = self.concatenate_horizontal_images(self.draw_bboxes(np.copy(image), windows), result)
      heatmaps = self.concatenate_horizontal_images(cv2.merge((weighted_heatmap*50, weighted_heatmap*50, weighted_heatmap*50)),
                                                    cv2.merge((heatmap*50, heatmap*50, heatmap*50)))
      return self.concatenate_vertical_images(boxes_and_result, heatmaps)
    else:
      return result


  def concatenate_horizontal_images(self, img11, img12):
    height, width= img11.shape[:2]
    total_width = 2 * width
    new_img = np.zeros(shape=(height, total_width, 3), dtype=img11.dtype)
    new_img[:height, :width] = img11
    new_img[:height, width:width + width] = img12
    return new_img


  def concatenate_vertical_images(self, img11, img21):
    height, width= img11.shape[:2]
    total_height = 2 * height
    new_img = np.zeros(shape=(total_height, width, 3), dtype=img11.dtype)
    new_img[:height, :width] = img11
    new_img[height:height + height, :width] = img21
    return new_img


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


  def prune_labels(self, pre_threshold_labels, threshold_labels):
    boxes = []
    for car_number in range(1, threshold_labels[1] + 1):
      # Find pixels with each car_number label value
      tnonzero = (threshold_labels[0] == car_number).nonzero()
      # Identify x and y values of those pixels
      tnonzeroy = np.array(tnonzero[0])
      tnonzerox = np.array(tnonzero[1])
      # Define a bounding box based on min/max x and y
      threshold_bbox = ((np.min(tnonzerox), np.min(tnonzeroy)), (np.max(tnonzerox), np.max(tnonzeroy)))

      for pre_threshold_car_number in range(1, pre_threshold_labels[1] + 1):
        ptnonzero = (pre_threshold_labels[0] == pre_threshold_car_number).nonzero()
        # Identify x and y values of those pixels
        ptnonzeroy = np.array(ptnonzero[0])
        ptnonzerox = np.array(ptnonzero[1])
        # Define a bounding box based on min/max x and y
        pre_threshold_bbox = ((np.min(ptnonzerox), np.min(ptnonzeroy)), (np.max(ptnonzerox), np.max(ptnonzeroy)))

        if pre_threshold_bbox[0][0] <= threshold_bbox[0][0] and threshold_bbox[1][0] <= pre_threshold_bbox[1][0] and \
                pre_threshold_bbox[0][1] <= threshold_bbox[0][1] and threshold_bbox[1][1] <= pre_threshold_bbox[1][1]:
          area = (pre_threshold_bbox[1][0] - pre_threshold_bbox[0][0]) * (pre_threshold_bbox[1][1] - pre_threshold_bbox[0][1])
          ymax = pre_threshold_bbox[1][1]
          threshold_area = 40 * ymax - 15000
          if area > threshold_area:
            boxes.append(pre_threshold_bbox)
    return boxes


  def draw_bboxes(self, img, boxes):
    for bbox in boxes:
      # Draw the box on the image
      cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255, 0.4), 6)
    # Return the image
    return img

