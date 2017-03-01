import matplotlib.pyplot as plt
import camera_calibration as cc
import frame_processing as fp
import numpy as np

import cv2

class VehicleDetectionPipeline:
  '''
  The VehicleDetectionPipeline class implements the framework for processing video images, detecting lane lines and
  marking them on the road image.
  '''

  def __init__(self, calibration_images,
               calibration_nx, calibration_ny,
               logger):
    self.camera_matrix, self.distortion_coeff = cc.calibrate(calibration_images,
                                                             calibration_nx,
                                                             calibration_ny)
    self.logger = logger
    self.windows = None
    self.window_sizes = [(32,32), (64,64), (96,96), (128,128), (256,256)]
    self.relative_y_start_stops = [(0.3, 0.7), (0.4, 0.8), (0.4, 0.8), (0.4, 0.9), (0.5, 1.0)]

  def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
      # Draw a rectangle given bbox coordinates
      cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


  def process_frame(self, image):
    undistorted_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)

    windows = self.slide_windows(image.shape,[None,None], self.y_start_stops(image.shape[0]),
                                 self.window_sizes)

    box_img = self.draw_boxes(undistorted_image, windows)

    return box_img


  def detect_cars(self, image):
    print("Hello!")


  def y_start_stops(self, height):
    absolute_y_start_stops = []
    for (ymin, ymax) in self.relative_y_start_stops:
      absolute_y_start_stops.append([ int(ymin * height), int(ymax * height)])
    return absolute_y_start_stops


  def slide_windows(self, shape,
                    x_start_stop=[None, None],
                    y_start_stops=([None, None]),
                    xy_windows=[(64, 64)],
                    xy_overlap=(0.5, 0.5)):
    if self.windows is not None:
      return self.windows
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
      x_start_stop[0] = 0
    if x_start_stop[1] == None:
      x_start_stop[1] = shape[1]

    window_list = []
    for i in range(len(xy_windows)):
      xy_window = xy_windows[i]
      y_start_stop = y_start_stops[i]
      if y_start_stop[0] == None:
        y_start_stop[0] = 0
      if y_start_stop[1] == None:
        y_start_stop[1] = shape[0]

      # Compute the span of the region to be searched
      xspan = x_start_stop[1] - x_start_stop[0]
      yspan = y_start_stop[1] - y_start_stop[0]
      # Compute the number of pixels per step in x/y
      nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
      ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
      # Compute the number of windows in x/y
      nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
      ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
      nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
      ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
      # Initialize a list to append window positions to

      # Loop through finding x and y window positions
      # Note: you could vectorize this step, but in practice
      # you'll be considering windows one by one with your
      # classifier, so looping makes sense
      for ys in range(ny_windows):
        for xs in range(nx_windows):
          # Calculate window position
          startx = xs * nx_pix_per_step + x_start_stop[0]
          endx = startx + xy_window[0]
          starty = ys * ny_pix_per_step + y_start_stop[0]
          endy = starty + xy_window[1]
          # Append window position to list
          window_list.append(((startx, starty), (endx, endy)))
      # Return the list of windows
    return window_list


