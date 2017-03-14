
import numpy as np
import cv2

def convert_color(img, src, tgt):
  '''
  Converts from color space src to target
  :param img: image to convert
  :param src: source color space string.
  :param tgt: target color space string.
  :return: image converted to color space
  '''
  if src == tgt:
    return np.copy(img)
  id = src + '2' + tgt
  conv_map = {
    'RGB2YCrCb' : cv2.COLOR_RGB2YCrCb,
    'RGB2YUV'   : cv2.COLOR_RGB2YUV,
    'RGB2LUV'   : cv2.COLOR_RGB2LUV,
    'RGB2HLS'   : cv2.COLOR_RGB2HLS,
    'RGB2HSV'   : cv2.COLOR_RGB2HSV,
    'RGB2BGR'   : cv2.COLOR_RGB2BGR,
    'BGR2RGB': cv2.COLOR_BGR2RGB,
    'BGR2YCrCb': cv2.COLOR_BGR2YCrCb,
    'BGR2YUV': cv2.COLOR_BGR2YUV,
    'BGR2LUV': cv2.COLOR_BGR2LUV,
    'BGR2HSV': cv2.COLOR_BGR2HSV,
    'YUV2BGR': cv2.COLOR_YUV2BGR,
    'YUV2RGB': cv2.COLOR_YUV2RGB,
    'YCrCb2RGB' : cv2.COLOR_YCrCb2RGB,
    'YCrCb2BGR': cv2.COLOR_YCrCb2BGR,
  }
  return cv2.cvtColor(img, conv_map[id])


def spatial_binning(img, size=(32, 32)):
  '''
  Get spatial binning features from image
  :param img: image to extract features from
  :param size: target image size
  :return: spatial binning features
  '''
  color1 = cv2.resize(img[:, :, 0], size).ravel()
  color2 = cv2.resize(img[:, :, 1], size).ravel()
  color3 = cv2.resize(img[:, :, 2], size).ravel()
  return np.hstack((color1, color2, color3))


def color_histogram(img, nbins=32, bins_range=(0, 256)):
  '''
  Get histogram of colors features
  :param img: image to extract features from
  :param nbins: number of bins in histogram
  :param bins_range: histogram range
  :return: color histogram features
  '''
  # Compute the histogram of the color channels separately
  channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
  channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
  channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
  # Concatenate the histograms into a single feature vector
  hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
  # Return the individual histograms, bin_centers and feature vector
  return hist_features

