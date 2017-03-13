
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
                      calibration_images,
                      calibration_nx,
                      calibration_ny,
                      car_images,
                      noncar_images,
                      classifier,
                      threshold,
                      scale1, scale2, scale3,
                      step1, step2, step3,
                      logger):
  vehicle_pipeline = VehicleDetectionPipeline(calibration_images, calibration_nx, calibration_ny,
                                                car_images, noncar_images, classifier, threshold,
                                                scale1, scale2, scale3, step1, step2, step3,
                                                logger)
  video_processing(input_video_file, start, end, output_video_file, vehicle_pipeline)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Self-Driving Car Vehicle Detection')
  parser.add_argument('-input_video', help='Path to video file to process', action='store')
  parser.add_argument('-start', help='Process subclip of input video, start in seconds', action='store', default=0/25.0)
  parser.add_argument('-end', help='Process subclip of input video, end in seconds', action='store', default=None)
  parser.add_argument('-output_video', help='Output video file', action='store', default='output.mp4')
  parser.add_argument('-calibration_images', help='Directory containing camera calibration images', action='store', default='./camera_cal/*.jpg')
  parser.add_argument('-calibration_nx', help='Number of x vertices in calibration images chess board', action='store', default=9)
  parser.add_argument('-calibration_ny', help='Number of y vertices in calibration images chess board', action='store', default=6)
  parser.add_argument('-car_images', help='Number of y vertices in calibration images chess board', action='store', default='./vehicles/')
  parser.add_argument('-noncar_images', help='Number of y vertices in calibration images chess board', action='store', default='./non-vehicles/')
  parser.add_argument('-classifier', help='Classifier pickle database, reuse instead of training again.', action='store', default='./classifier.p')
  parser.add_argument('-threshold', help='Threshold for heatmap to set box to 0.', action='store', default=6)
  parser.add_argument('-log_enabled', help='Path to file to store the log', action='store', default=True)
  parser.add_argument('-log_dir', help='Path to file to store the log', action='store', default='./log')
  parser.add_argument('-log_rate', help='Every % frames store image in log. Valid only if log is enabled', action='store', default=25)

  parser.add_argument('-scale1', help='', action='store', default=1.0)
  parser.add_argument('-scale2', help='', action='store', default=0.0)
  parser.add_argument('-scale3', help='', action='store', default=0.0)
  parser.add_argument('-step1', help='', action='store', default=2)
  parser.add_argument('-step2', help='', action='store', default=2)
  parser.add_argument('-step3', help='', action='store', default=2)
  args = parser.parse_args()

  print("start: {} end: {}".format(args.start, args.end))

  logger = LoggerCV(args.log_enabled, args.log_rate)

  calibration_images = glob.glob(args.calibration_images)

  end_time = args.end
  if end_time is not None:
    end_time = float(end_time)

  vehicle_detection(args.input_video, float(args.start), end_time,
                    args.output_video, calibration_images,
                    args.calibration_nx, args.calibration_ny,
                    args.car_images, args.noncar_images,
                    args.classifier,
                    float(args.threshold),
                    float(args.scale1),
                    float(args.scale2),
                    float(args.scale3),
                    int(args.step1),
                    int(args.step2),
                    int(args.step3),
                    logger)

  if ( args.log_enabled == True ):
    logger.write_records(args.log_dir)
