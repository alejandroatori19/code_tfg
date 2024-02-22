# Libraries
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# ... (other imports)
import pyrealsense2 as rs2                            # Camera RGBD  or ".bag" files
import numpy as np                                          # Arrays managment
import cv2                                                  # Images and graphic interface

import os
import sys
import time


class SpecificWorker(GenericWorker):
      """
      SpecificWorker class handles tasks specific to video processing with RealSense camera or recorded video file.
      Inherits from GenericWorker class.

      Attributes:
      - period: Period for QTimer
      - averageTimePerFrame: Average time taken per frame
      - startTime: Start time for frame processing
      - pathVideo: Path to the video file or RealSense bag file
      - pipeline: RealSense pipeline object
      - configuration: Configuration object for the RealSense pipeline
      - pathDatasetFolderColorFrames: Folder to save color frames in the dataset
      - pathDatasetFolderDepthFrames: Folder to save depth frames in the dataset
      - counterFrames: Counter for the processed frames
      - isPipelineStarted: Flag indicating whether the RealSense pipeline is started
      
      """
      period = 35             # Period for QTimer
      averageTimePerFrame = None
      startTime = None

      # Video
      pathVideo = "/home/robocomp/Downloads/videosOficiale/video1.bag"        # The path of the video we will be reading
      pipeline = None         # Pipeline to connect with the video and request
      configuration = None    # Configuration of the pipeline
      
      # Dataset
      pathDatasetFolderColorFrames = "/home/robocomp/dataset/colorImages/"              # The folder where the dataset will be saved
      pathDatasetFolderDepthFrames = "/home/robocomp/dataset/depthImages/"
      counterFrames = None

      isPipelineStarted = False

      # -------------------------------------------------

      def __init__(self, proxy_map, startup_check=False):
            """
            Constructor for SpecificWorker class.

            Parameters:
            - proxy_map: Proxy map for communication
            - startup_check: Flag indicating whether to perform startup checks
            
            """
            
            super(SpecificWorker, self).__init__(proxy_map)
            
            # Set counter frames to 1 (First frame)
            self.counterFrames = 1

            # Prepare the timers
            self.startTime = time.time ()
            self.averageTimePerFrame = 0

            # Assigning the period to the global variable
            self.Period = self.period
            
            # Check if all conditions are meeting
            #self.check_conditions ()

            # Finally init the video preload
            self.initialize_video ()
      
            # Activates the clock and call method compute when it gets the time that the variable period indicates
            self.timer.timeout.connect(self.compute)
            self.timer.start(self.Period)

            return

      # ----------------

      def __del__(self):
            """
            Destructor method to clean up resources.
            
            """
            
            self.data_to_user ()

            # Stop getting frames from camera
            if self.isPipelineStarted:
                  self.pipeline.stop ()

            return True
      
      # -------------------------

      def setParams(self, params):
            """
            Set parameters if needed.

            Parameters:
            - params: Parameters to set
            
            """

            return True
      
      # ----------------

      @QtCore.Slot()
      def compute(self):
            """
            Method to perform computations at regular intervals.
            
            """

            initTime = time.time ()

            # We receive a frame and flag that indicates if that´s a frame or nothing
            isFramevailable, frame = self.pipeline.try_wait_for_frames ()

            # If frame is available
            if isFramevailable:
                  # Increment averageTimePerFrame
                  self.averageTimePerFrame += (time.time() - initTime)
                  
                  # First of all it rotate the frame. Only if it´s needed
                  rotatedColorFrame = self.rotate_frame (frame.get_color_frame ().get_data (), False)
                  rotatedDepthFrame = self.rotate_frame (frame.get_depth_frame ().get_data (), False)

                  # It will show to user both image
                  self.user_interface (rotatedColorFrame, rotatedDepthFrame, False) 

                  # Saving both frames into disk
                  #self.save_frames_to_disk (rotatedColorFrame, rotatedDepthFrame)
                  
                  # Increment the counter (Each image is called image_<number>, and it mustn´t be repeated)
                  self.counterFrames += 1
            else:
                  sys.exit (0)

            return True


      # ------------------------------------------------
      # ------------------ INITIALIZE ------------------
      # ------------------------------------------------

      def check_conditions (self):
            """
            Method to check conditions for proper initialization.
            
            """

            # Check if counterFrames is 1 already. At the start of the code.      
            if self.counterFrames != 1:
                  sys.exit ("FAILURE (0): counterFrames isn´t initialize at \"1\". Check the method check_conditions (self)")    

            # The path of dataset has to be a folder and it must be empty. If it´s not it will just stop.
            if not (os.path.isdir (self.pathDatasetFolderColorFrames) and (not any (os.listdir(self.pathDatasetFolderColorFrames)))):
                  sys.exit ("FAILURE (1): pathDatasetFolderColorFrames isn´t correct. Check the method check_conditions (self)")

            # The path of dataset has to be a folder and it must be empty. If it´s not it will just stop.
            if not (os.path.isdir (self.pathDatasetFolderDepthFrames) and (not any (os.listdir(self.pathDatasetFolderDepthFrames)))):
                  sys.exit ("FAILURE (1): pathDatasetFolderDepthFrames isn´t correct. Check the method check_conditions (self)")

            # Get the extension of video file
            extensionVideoFile = os.path.basename (self.pathVideo)            # Convert path/videoFile.bag into videoFile.bag
            extensionVideoFile = extensionVideoFile [(extensionVideoFile.find ('.') + 1):]     # Convert videoFile.bag into bag

            # The file has to be a file and must need to have a ".bag" extension
            if not (os.path.isfile (self.pathVideo) and extensionVideoFile == "bag"):
                  sys.exit ("FAILURE (2): pathVideo isn´t correct. Check the method check_conditions (self)")


            return True

      # -------------------------------------------------
      # -------------------- COMPUTE --------------------
      # -------------------------------------------------

      def save_frames_to_disk (self, colorFrame, depthFrame):
            """
            Method to save color and depth frames to disk.

            Parameters:
            - colorFrame: Color frame to save
            - depthFrame: Depth frame to save
            
            """
            
            # Path to save the frames
            pathColorrame = self.pathDatasetFolderColorFrames + "/image_" + str (self.counterFrames) + ".jpeg" 
            pathDepthFrame = self.pathDatasetFolderDepthFrames + "/image_" + str (self.counterFrames) + ".jpeg" 
                        
            # Saving image with opencv
            cv2.imwrite (pathColorrame, colorFrame)
            cv2.imwrite (pathDepthFrame, depthFrame)

            return True

      # ------------------------------------------------
      # -------------------- FINISH --------------------
      # ------------------------------------------------

      def data_to_user (self):
            """
            Method to print and display data to the user.
            
            """

            # Calculating data
            elapsedTime = time.time() - self.startTime
            self.averageTimePerFrame /= (self.counterFrames - 1)

            # Print data
            print ("\n\n--------------- INFORMATION ---------------")
            print (f"Time taken: {elapsedTime} seconds")        
            print (f"Average time per frame: {self.averageTimePerFrame} seconds")
            print (f"Images processed: {self.counterFrames - 1}")
            print ("------------- END INFORMATION -------------\n\n")
            print ("Number of frames saved:", str (self.counterFrames - 1))
            

            return True
      
            
      # -----------------------------------------------
      # -------------------- EXTRA --------------------
      # -----------------------------------------------
      
      def rotate_frame(self, frame, isDepthFrame = False):
            """
            Method to rotate a frame.

            Parameters:
            - frame: Frame to rotate
            - isDepthFrame: Flag indicating whether the frame is a depth frame

            Returns:
            - Rotated frame

            """
            
            frameNumpy = np.asarray (frame)       # Convert frame into array (We need it to rotate frame)
            frameRotated = np.rot90 (frameNumpy, 3)          # Rotate frame
            
            if isDepthFrame:
                  frameRotated = cv2.applyColorMap(cv2.convertScaleAbs(frameRotated, alpha=0.03), cv2.COLORMAP_JET)

            return frameRotated
      
      # -----------------------------------------------

      def user_interface (self, colorFrame, depthFrame, imagesConcatenated = True):
            """
            Method to display frames to the user.

            Parameters:
            - colorFrame: Color frame
            - depthFrame: Depth frame
            - imagesConcatenated: Flag indicating whether to concatenate color and depth frames

            """

            if imagesConcatenated:
                  
                  if not (colorFrame.shape == depthFrame.shape):
                        sys.exit ("FAILURE (4): You are trying to concatenate depth and RGB frames but they haven´t same shape")

                  # First of all concatenate both images
                  framesConcatenated = np.concatenate((colorFrame, depthFrame), axis=1) 
            
                  # It will be shown in a cv.namedWindow () which name and ID is "frames"
                  cv2.imshow ("frames", framesConcatenated)

                  # Finally it waits just 1 ms or a key pressed (If the key pressed is ESC then it finish)
                  if cv2.waitKey (1) == 27:
                        sys.exit ("It finish by pressing ESC. Recommending just use it for test or emergencies.")

            else:
                  # It will be shown in a cv.namedWindow () which name and ID is "frames"
                  cv2.imshow ("color frame", colorFrame)
                  cv2.imshow ("depth frame", depthFrame)

                  # Finally it waits just 1 ms or a key pressed (If the key pressed is ESC then it finish)
                  if cv2.waitKey (1) == 27:
                        sys.exit ("It finish by pressing ESC. Recommending just use it for test or emergencies.")

            return True      

      # -------------------------

      def initialize_video (self):
            """
            Method to initialize the video pipeline.
            
            """

            # First of all it generates an object of this classes and save in the global variable each one empty.
            self.pipeline = rs2.pipeline ()
            self.configuration = rs2.config()
            
            # Indicates the file you want to read without playback
            self.configuration.enable_device_from_file (self.pathVideo, False)
            
            # Activate color stream (30 FPS)
            #self.configuration.enable_stream (rs2.stream.color, rs2.format.bgr8, 30)
            #self.configuration.enable_stream (rs2.stream.depth, rs2.format.bgr8, 30)

            self.configuration.enable_all_streams ()

            # Starts the pipeline
            self.pipeline.start (self.configuration)
            
            self.isPipelineStarted = True

            return True
