# Libraries
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Libraries needed
import pyrealsense2 as rs2
import cv2 as cv
import time
import numpy as np

# Class
class SpecificWorker(GenericWorker):
    
    period = 33         # Clock´s period             (30 FPS = 33,333 period
    
    
    # Video
    pipeline = None
    configuration = None
    recorder = None
    
    # Extra data
    counterFrames = None
    startTime = None
    averageTimePerFrame = None

    # Paths & string data
    serialNumberCamera = "146222252950"                     # Other chance 146222252950
    pathVideoBag = "/home/robocomp/video.bag"


    METHOD_RECORDING = 2                # 1 per time, 2 per frames

    # Limits per recording
    LIMIT_TIME_RECORDING = 100          # That´s indicated in seconds
    LIMIT_FRAMES_RECORDING = 100        # That´s indicated in frames
    SHAPE_FRAMES = (640, 480)           # Width, height of frames
    FPS_RECORDING = 30                  # Frames per second of the recording

    def __init__(self, proxy_map, startup_check=False):
        
    
        super(SpecificWorker, self).__init__(proxy_map)
    
        self.Period = self.period          # Assing period
                
        self.startTime = time.time ()   # Set the starter time into variable
        self.counterFrames = 0          # Set the counters to 0
        self.averageTimePerFrame = 0    # Set to 0 because it´s an acumulator so it will be increased by sum

        self.check_conditions ()
    
        self.init_camera_and_record ()

        # Timer activation
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        return

    def __del__(self):
        # First of all it must show to user the data. It has information about times & number of frames that has been processed
        if self.counterFrames > 0:
            self.data_to_user ()
        
        # Stop recording and release resources
        self.pipeline.stop()

        return 

    def setParams(self, params):

        return True


    @QtCore.Slot()
    def compute(self):
        
        startTimeFrame = time.time ()

        # Request a frame
        isFramevailable, frames = self.pipeline.try_wait_for_frames ()
        
        self.averageTimePerFrame += (time.time () - startTimeFrame)

        # If there is a frame available it show to the user and save it into disk
        if isFramevailable:
            # Split frame into depth and RGB frame
            colorFrame = frames.get_color_frame().get_data ()
            depthFrame = frames.get_color_frame().get_data ()

            self.user_interface (colorFrame, depthFrame)
            

            self.counterFrames += 1

        else:
            self.timer.stop ()
            sys.exit ("The programme has finished because there aren´t frames available")

        if self.METHOD_RECORDING == 1:
            # Check if the code has reached the limit
            if (time.time () - self.startTime) > self.LIMIT_TIME_RECORDING:
                self.timer.stop ()
                sys.exit ("The programme has finished because it reaches the time limit")

        elif self.METHOD_RECORDING == 2:
            # Check if the code has reached the limit
            if self.counterFrames > self.LIMIT_FRAMES_RECORDING:
                self.timer.stop ()
                sys.exit ("The programme has finished because it reaches the time limit")
                
        return


    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------

    def check_conditions (self):
        """
        Method to check conditions for proper initialization.
        
        """    
        

        # Check if counterFrames is 1 already. At the start of the code.      
        if self.counterFrames != 0:
            sys.exit ("FAILURE (0): counterFrames isn´t initialize at \"0\". Check the method check_conditions (self)")    

        if not (self.METHOD_RECORDING == 1 or self.METHOD_RECORDING == 2):
            sys.exit ("FAILURE (1): METHOD_RECORDING isn´t initialize with correct value. Check the method check_conditions (self)")    

        # Get the extension of video file
        folderVideoFile = os.path.dirname (self.pathVideoBag)
        extensionVideoFile = os.path.basename (self.pathVideoBag)            # Convert path/videoFile.bag into videoFile.bag
        extensionVideoFile = extensionVideoFile [(extensionVideoFile.find ('.') + 1):]     # Convert videoFile.bag into bag

        # The file has to be a file and must need to have a ".bag" extension
        if not (os.path.isdir (folderVideoFile) and extensionVideoFile == "bag"):
            sys.exit ("FAILURE (2): pathVideoBag isn´t correct. Check the method check_conditions (self)")

        
        return True

    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------

    def user_interface (self, colorFrame, depthFrame):
        # It´s necessary to show it with opencv
        colorFrameArray = np.asanyarray (colorFrame)
        depthFrameArray = np.asanyarray (depthFrame)

        # Show it in 2 windows
        cv.imshow('RGB Frame', colorFrameArray)
        cv.imshow('Depth Frame', depthFrameArray)
    
        # Detect a key press, if not just skip the next checks of conditions
        letterPressed = cv.waitKey (1)

        # If users press the ESC key then the program will be closed
        if letterPressed == 27:
            self.timer.stop ()
            sys.exit ("The programme has finished while it was excuting because user press ESC.")

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
        self.averageTimePerFrame /= self.counterFrames

        # print data
        print (f"\n\n--------------- INFORMATION ---------------")
        print (f"Time taken: {elapsedTime} seconds")        
        print (f"Average time per frame: {self.averageTimePerFrame} seconds")
        print (f"Frames processed: {self.counterFrames - 1}")
        print (f"------------- END INFORMATION -------------\n\n")
            

        return True
            
    # -----------------------------------------------
    # -------------------- EXTRA --------------------
    # -----------------------------------------------

    def init_camera_and_record (self):
        self.pipeline = rs2.pipeline ()
        self.configuration = rs2.config ()

        self.configuration.enable_device (self.serialNumberCamera)

        # Prepare the configuration of the vieo
        self.configuration.enable_stream (rs2.stream.color, self.SHAPE_FRAMES[0], self.SHAPE_FRAMES[1], rs2.format.bgr8, self.FPS_RECORDING)
        self.configuration.enable_stream (rs2.stream.depth, self.SHAPE_FRAMES[0], self.SHAPE_FRAMES[1], rs2.format.z16, self.FPS_RECORDING)

        # Set the fate of images to the file indicated by the path
        self.configuration.enable_record_to_file (self.pathVideoBag)        


        # Start the pipeline with the video offered by the camera which serial number is the ones that saves the variable.
        pR = self.pipeline.start (self.configuration)

        # Recorder
        dR = pR.get_device ()
        self.recorder = dR.as_recorder ()

        return True
    
