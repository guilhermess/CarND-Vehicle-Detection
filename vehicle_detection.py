
from VehicleDetectionPipeline import VehicleDetectionPipeline
from moviepy.editor import VideoFileClip
from LoggerCV import LoggerCV
import numpy as np
import argparse
import glob
import cv2


def video_processing(input_video_file, start, end, output_video_file, vehicle_detection_pipeline):
  input_video = VideoFileClip(input_video_file).subclip(start, end)
  output_video = input_video.fl_image(vehicle_detection_pipeline.process_frame)
  output_video.write_videofile(output_video_file, audio=False)


def vehicle_detection(input_video_file,
                      start,
                      end,
                      output_video_file,
                      car_images,
                      noncar_images,
                      classifier,
                      spatial, hoc, hog,
                      threshold_low,
                      threshold_high,
                      debug_video,
                      logger):
  vehicle_pipeline = VehicleDetectionPipeline(car_images, noncar_images, classifier, spatial, hoc, hog,
                                              threshold_low, threshold_high,
                                              debug_video, logger)
  video_processing(input_video_file, start, end, output_video_file, vehicle_pipeline)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Self-Driving Car Vehicle Detection')
  parser.add_argument('-input_video', help='Path to video file to process', action='store')
  parser.add_argument('-start', help='Process subclip of input video, start in seconds', action='store', default=0/25.0)
  parser.add_argument('-end', help='Process subclip of input video, end in seconds', action='store', default=None)
  parser.add_argument('-output_video', help='Output video file', action='store', default='output.mp4')
  parser.add_argument('-car_images', help='Car Images for Classifier', action='store', default='./vehicles/')
  parser.add_argument('-noncar_images', help='Non-Car Images for Classifier', action='store', default='./non-vehicles/')
  parser.add_argument('-classifier', help='Classifier pickle database, reuse instead of training again.', action='store', default='./classifier.p')
  parser.add_argument('-threshold_low', help='Min Threshold for heatmap to draw bounding box.', action='store', default=2)
  parser.add_argument('-threshold_high', help='Threshold for heatmap to consider as a car box.', action='store', default=4)
  parser.add_argument('-spatial', help='Enable/Disable spatial binned features', action='store', default="False")
  parser.add_argument('-hoc', help='Enable/Disable histogram of colors features', action='store', default="False")
  parser.add_argument('-hog', help='Enable/Disable histogram of gradients features', action='store', default="True")

  parser.add_argument('-log_enabled', help='Path to file to store the log', action='store', default=True)
  parser.add_argument('-log_dir', help='Path to file to store the log', action='store', default='./log')
  parser.add_argument('-log_rate', help='Every % frames store image in log. Valid only if log is enabled', action='store', default=25)
  parser.add_argument('-debug_video', help='Write a debug video with heatmap and candidate windows.', action='store', default="False")
  args = parser.parse_args()

  logger = LoggerCV(args.log_enabled, args.log_rate)

  end_time = args.end
  if end_time is not None:
    end_time = float(end_time)

  spatial = True
  if args.spatial != "True":
    spatial = False

  hoc = True
  if args.hoc != "True":
    hoc = False

  hog = True
  if args.hog != "True":
    hog = False

  debug_video = True
  if args.debug_video != "True":
    debug_video = False

  vehicle_detection(args.input_video, float(args.start), end_time,
                    args.output_video,
                    args.car_images, args.noncar_images,
                    args.classifier,
                    spatial,
                    hoc,
                    hog,
                    float(args.threshold_low),
                    float(args.threshold_high),
                    debug_video,
                    logger)

  if ( args.log_enabled == True ):
    logger.write_records(args.log_dir)
