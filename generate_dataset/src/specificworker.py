#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 by YOUR NAME HERE
#
#    This file is part of RoboComp
#
#    RoboComp is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    RoboComp is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
#

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Libraries needed
import pyrealsense2 as rs2                            # Camera RGBD  or ".bag" files
import numpy as np                                          # Arrays managment
import cv2 as cv                                                  # Images and graphic interface

import os
import sys


class SpecificWorker(GenericWorker):
      
      period = 35             # Period
      
      # Video
      pathVideo = "/media/robocomp/externalDisk/rightCameraVideo3.bag"        # The path of the video we will be reading
      pipeline = None         # Pipeline to connect with the video and request
      configuration = None    # Configuration of the pipeline
      
      # Dataset
      pathDatasetFolder = "/media/robocomp/externalDisk/dataset"              # The folder where the dataset will be saved
      extensionImages = ".jpeg"           # Indicates the extension of the image ("png", "jpeg", "jpg", etc.)
      counterFrames = None


      # Class constructor. Set initial values to global variables and Period
      def __init__(self, proxy_map, startup_check=False):
            super(SpecificWorker, self).__init__(proxy_map)
            
            # First of all we need to start the link with the video if you want to request frames.
            self.initialize_video ()
            self.initialize_variables ()
            
            # Activates the clock and call method compute when it gets the time that the variable period indicates
            self.Period = self.period
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

            return

      # Destructor
      def __del__(self):
            cv.destroyAllWindows()
            self.pipeline.stop ()
            print ("Number of frames saved:", str (self.counterFrames - 1))
            
            return
      
      def setParams(self, params):

            return True
  
      # Compute
      @QtCore.Slot()
      def compute(self):
            # We receive a frame and flag that indicates if that´s a frame or nothing
            isImageAvailable, frame = self.pipeline.try_wait_for_frames ()

            if isImageAvailable:
                  rotatedFrame = self.rotate_frame (frame)
                  cv.imshow ("Frame", rotatedFrame)
                  letter = cv.waitKey (1)
                  if letter == 27:
                        sys.exit (-443)
                  
                  self.save_frame (rotatedFrame)
                  
                  # Increment the counter (Each image is called image_<number>, and it mustn´t be repeated)
            else:
                  sys.exit (0)

            return True


      

      # Initialize preload video
      def initialize_video (self):
            # First of all it generates an object of this classes and save in the global variable each one empty.
            self.pipeline = rs2.pipeline ()
            self.configuration = rs2.config()
            
            # Indicates the file you want to read without playback
            self.configuration.enable_device_from_file (self.pathVideo, False)
            
            # Activate color stream (30 FPS)
            self.configuration.enable_stream (rs2.stream.color, rs2.format.bgr8, 30)
            
            # Starts the pipeline
            self.pipeline.start (self.configuration)
            
            return

      def initialize_variables (self):
            self.counterFrames = 1
            
            return

      # Rotate image ()
      def rotate_frame(self, frame):
            colorFrame = frame.get_color_frame ()           # Gets the color frame
            colorFrameNumpy = np.asarray (colorFrame.get_data ())       # Convert frame into array (We need it to rotate frame)
            frameRotated = np.rot90 (colorFrameNumpy, 3)          # Rotate frame
            
            return frameRotated
      
      def save_frame (self, finalFrame):
            # Path to save the frame
            pathFinalFrame = self.pathDatasetFolder + "/image_" + str (self.counterFrames) + self.extensionImages  
            
            print ("path final frame:", pathFinalFrame )      
            
            # Saving image with opencv
            cv.imwrite (pathFinalFrame, finalFrame)

            self.counterFrames += 1

            return
            
            
            
            









