import frame_processing as fp
import matplotlib.image as mpimg
import numpy as np
import pickle
import cv2
import os

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from LoggerCV import LoggerCV
from LoggerCV import CVRecord
import matplotlib.pyplot as plt
import glob

class VehicleClassifier:
  def __init__(self,
               car_files_dir, noncar_files_dir,
               classifier,
               logger,
               color_space='YUV',
               spatial_size=32,
               histogram_of_colors_bins=64,
               histogram_of_colors_range=(0, 256),
               orient=9,
               pix_per_cell=8,
               cell_per_block=2,
               hog_channels=(0, 1, 2),
               features_spatial=True,
               features_histogram_of_colors=True,
               features_histogram_of_gradients=True,
               ):

    self.logger = logger
    self.color_space = color_space

    ## Color: Spatial Features Parameters
    self.features_spatial = features_spatial
    self.spatial_size = (spatial_size, spatial_size)

    ## Color: Histogram of Colors Parameters
    self.features_histogram_of_colors = features_histogram_of_colors
    self.histogram_of_colors_bins = histogram_of_colors_bins
    self.histogram_of_colors_range = histogram_of_colors_range

    # Gradient: Histogram of Gradients Parameters
    self.features_histogram_of_gradients = features_histogram_of_gradients
    self.orient = orient
    self.pix_per_cell = pix_per_cell
    self.cell_per_block = cell_per_block
    self.hog_channels = hog_channels

    if (os.path.exists(classifier)):
      with open(classifier, 'rb') as input:
        self.car_classifier, self.scaler = pickle.load(input)
    else:
      car_files = self.get_files(car_files_dir, '.png')
      noncar_files = self.get_files(noncar_files_dir, '.png')
      self.car_classifier, self.scaler = self.train(car_files, noncar_files)
      with open(classifier, 'wb') as output:
        pickle.dump((self.car_classifier, self.scaler), output, pickle.HIGHEST_PROTOCOL)


  def train(self, car_files, noncar_files):
    car_features = []
    noncar_features = []
    for car_file in car_files:
      car_image = mpimg.imread(car_file)
      car_features.append(self.extract_features(car_image))

    for noncar_file in noncar_files:
      noncar_image = mpimg.imread(noncar_file)
      noncar_features.append(self.extract_features(noncar_image))

    x = np.vstack((car_features, noncar_features)).astype(np.float64)
    x_scaler = StandardScaler().fit(x)
    scaled_x = x_scaler.transform(x)

    y = np.hstack((np.ones(len(car_features)),
                   np.zeros(len(noncar_features))))

    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.2, random_state=rand_state)
    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(x_train, y_train)

    print("Classifier Test Accuracy: {}".format(svc.score(x_test, y_test)))

    return svc, x_scaler

  def get_files(self, dir, extension ):
    filelist = []
    for root, dirs, files in os.walk(dir):
      for file in files:
        _, fextension = os.path.splitext(file)
        if fextension == extension:
          filelist.append(root + '/' + file )
    return filelist

  def extract_features(self, image):
    color_image = fp.convert_color(image, 'RGB', self.color_space).astype(np.float64)
    features = []
    if self.features_spatial:
      color_spatial_features = fp.spatial_binning(color_image, self.spatial_size)
      features.append(color_spatial_features)
    if self.features_histogram_of_colors:
      color_histogram_features = fp.color_histogram(color_image,
                                                    self.histogram_of_colors_bins,
                                                    self.histogram_of_colors_range)
      features.append(color_histogram_features)
    if self.features_histogram_of_gradients:
      hog_features = []
      for channel in self.hog_channels:
        hog_channel_features = self.get_hog_features(color_image[:,:,channel], self.orient,
                                                     self.pix_per_cell, self.cell_per_block, False, False)
        hog_features.append(hog_channel_features.ravel())
      histogram_of_gradients_features = np.hstack(hog_features)
      features.append(histogram_of_gradients_features)

    return np.concatenate(features)


  def find_cars(self, image, ystart, ystop, scale, cells_per_step, frame_number):
    if self.logger.enabled(frame_number):
      self.logger.log(CVRecord("find_cars_original", frame_number, [image.astype(np.float32)/255], 'opencv'))

    img = fp.convert_color(image, 'RGB', self.color_space).astype(np.float64)/255

    ctrans_tosearch = img[ystart:ystop, :, :]
    if scale != 1:
      imshape = ctrans_tosearch.shape
      ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // self.pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // self.pix_per_cell) - 1
    nfeat_per_block = self.orient * self.cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // self.pix_per_cell) - 1
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = self.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
    hog2 = self.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
    hog3 = self.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)

    if self.logger.enabled(frame_number):
      debug_boxes = []

    candidate_windows = []
    for xb in range(nxsteps):
      for yb in range(nysteps):
        ypos = yb * cells_per_step
        xpos = xb * cells_per_step
        # Extract HOG for this patch
        hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
        hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

        xleft = xpos * self.pix_per_cell
        ytop = ypos * self.pix_per_cell

        # Extract the image patch
        subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

        #if self.logger.enabled(frame_number):
        #  self.logger.log( CVRecord("find_cars_subimg_" + str(xleft) + "_" + str(ytop),
        #                            frame_number, [fp.convert_color(subimg*255, self.color_space, 'RGB')], 'opencv'))

        # Get color features
        spatial_features = fp.spatial_binning(subimg, size=self.spatial_size)
        hist_features = fp.color_histogram(subimg, nbins=self.histogram_of_colors_bins)

        # Scale features and make a prediction
        features = self.scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
        prediction = self.car_classifier.predict(features)

        if self.logger.enabled(frame_number):
          xbox_left = np.int(xleft * scale)
          ytop_draw = np.int(ytop * scale)
          win_draw = np.int(window * scale)
          debug_boxes.append(((xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        if prediction == 1:
          #if ( self.logger.enabled(frame_number)):
          #  print("found car {}_{}".format(xleft, ytop))
          xbox_left = np.int(xleft * scale)
          ytop_draw = np.int(ytop * scale)
          win_draw = np.int(window * scale)
          candidate_windows.append(((xbox_left, ytop_draw + ystart),
                                    (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    if self.logger.enabled(frame_number):
      box_img = self.draw_boxes(image, candidate_windows)
      self.logger.log(CVRecord("car_boxes_scale_" + str(scale),
                               frame_number, [box_img.astype(np.float32)/255], 'opencv'))

      debug_boxes = self.draw_boxes(image, debug_boxes)
      self.logger.log(CVRecord("debug_boxes_scale_" + str(scale),
                               frame_number, [debug_boxes.astype(np.float32) / 255], 'opencv'))

    return candidate_windows

  def draw_boxes(self, img, boxes):
    copy_img = np.copy(img)
    for box in boxes:
      r = np.random.randint(0, 255)
      g = np.random.randint(0, 255)
      b = np.random.randint(0, 255)
      cv2.rectangle(copy_img, box[0], box[1], (r,g,b), 6)
    return copy_img


  def get_hog_features(self, img, orient,
                       pix_per_cell, cell_per_block,
                       vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
      features, hog_image = hog(img, orientations=orient,
                                pixels_per_cell=(pix_per_cell, pix_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block),
                                transform_sqrt=False,
                                visualise=vis, feature_vector=feature_vec)
      return features, hog_image
    # Otherwise call with one output
    else:
      features = hog(img, orientations=orient,
                     pixels_per_cell=(pix_per_cell, pix_per_cell),
                     cells_per_block=(cell_per_block, cell_per_block),
                     transform_sqrt=False,
                     visualise=vis, feature_vector=feature_vec)
      return features


if __name__ == "__main__":
  images = glob.glob('test_images/*.jpg')
  logger = LoggerCV(False,0)
  car_classifier = VehicleClassifier('./vehicles', './non-vehicles/', 'classifier.p', logger)
  count = 0
  for file in images:
    img = mpimg.imread(file)
    boxes = car_classifier.find_cars(img, 400, 720, 2.0, 0)
    box_img = car_classifier.draw_boxes(img, boxes)
    mpimg.imsave("img_" + str(count) + ".jpg", box_img*255)
    count += 1

