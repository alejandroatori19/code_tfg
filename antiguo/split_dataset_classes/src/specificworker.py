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


# If RoboComp was compiled with Python bindings you can use InnerModel in Python
# import librobocomp_qmat
# import librobocomp_osgviewer
# import librobocomp_innermodel


class SpecificWorker(GenericWorker):
    """
    Class implementing a specific worker for processing image frames with YOLO neural network.

    Attributes:
    - period (int): Clock's period in seconds.
    - startTime (float): Time when the program starts.
    - averageTimePerFrame (float): Average time per frame (Detection + save data + save image).

    - neuralNetModel: YOLO model for neural network.
    - confidenceThreshold (float): Minimum confidence level for detections.
    - classList (list): List of classes corresponding to YOLO class indices.

    - detectionData (dict): Dictionary to store detection data.
    - classesInterested (list): List of classes for which detections are of interest.
    - counterFrames (int): Counter for processed frames.

    - listNameFiles (list): List of file names in the dataset folder.
    - pathDataset (str): Path to the dataset folder.
    - extensionsAvailable (list): List of valid file extensions.

    - pathFolderRois (str): Path to the folder where detection images are saved.
    - pathJsonFile (str): Path to the output JSON file for detection data.
    """

    period = 300     # Clock´s period
    startTime = None        # Time when program starts
    averageTimePerFrame = 0     # Average time per frame (Detection + save data + save image)

    # Neural net
    neuralNetModel = None       # Nueral net
    confidenceThreshold = 0.5           # Minimum confidence

    # List with classes & id of each class
    classList = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                 "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                 "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                 "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                 "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                 "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                 "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                 "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                 "teddy bear", "hair drier"
                 ]
    
    # Detections
    detectionData = {}              # Dictionary that saves the data from detections
    classesInterested = [0]          # List that saves the classes we are interested in (At this moment only people - 0)
    counterFrames = None            # Counter of frames

    # Dataset
    listNameFiles = []
    pathDataset = "/media/robocomp/externalDisk/dataset/originalImages/"
    extensionsAvailable = ["jpeg", "jpg", "png"]

    
    # Paths
    pathFolderRois = "/media/robocomp/externalDisk/dataset/detectionImages/"
    pathJsonFile = "/home/robocomp/data.json"


    pathTestImage = "/media/robocomp/externalDisk/dataset/originalImages/image_219.jpeg"

    

    def __init__(self, proxy_map, startup_check=False):
        """
        Constructor for SpecificWorker class.

        Parameters:
        - proxy_map: Proxy map for communication.
        - startup_check (bool): Flag for startup checks.

        """
        
        
        super(SpecificWorker, self).__init__(proxy_map)
        

        # Setting period
        self.Period = self.period
        self.startTime = time.time ()
        
        # Load neural net
        self.neuralNetModel = YOLO ("yolov8s.pt")
        
        # Before start the clock it checks the conditions
        self.check_conditions ()

        # Timer starts
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        return 


    def __del__(self):
        """
        Destructor for SpecificWorker class.

        """

        return    


    def setParams(self, params):
        """
        Set parameters for the worker.

        Parameters:
        - params: Parameters to set.

        Returns:
        - bool: True if successful.

        """

        self.counterFrames = 0
        return True


    @QtCore.Slot()
    def compute(self):
        """
        Main computation method triggered by the timer.
        Processes each frame, performs YOLO detection, and updates the UI (Testing).

        Returns:
        - bool: True if successful.

        """
        # Check if there are more frames available        
        if self.counterFrames < len (self.listNameFiles):

            # Pre define the path of image (Merging pathDataset + currentImage)
            pathImage = self.pathDataset + self.listNameFiles[self.counterFrames]

            # Then it reads the image and cross it throught the neural net
            frame = cv.imread (pathImage)

            # time when it start crossing
            startTimeDetection = time.time ()
            
            # Crossing neural net
            results = self.neuralNetModel (frame)
            
            # Add time to average time
            self.averageTimePerFrame += (time.time () - startTimeDetection)

            # Split the reuslts per detections and take the ones that interest most. 
            self.work_with_detection_results (frame, results)

            # Show it to the user. Just for testing.
            self.user_interface (frame, results)

            # Increment counter frames because it will get the next frame.
            self.counterFrames += 1
        
        else:
            # If there aren´t images available then just type the data into json file.
            self.data_to_user ()

            # Finally, finish the code
            sys.exit ("The programme has ended correctly.")


        return True

    

    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------

    def check_conditions (self):
        """
        Check various conditions before starting the computation.
        Validates paths, confidence threshold, and other parameters.

        Raises:
        - SystemExit: If conditions are not met.

        Returns:
        - bool: True if successful.

        """
        # Check if pathDataset saves a path of a folder and if it exists.
        if not (os.path.exists (self.pathDataset) and os.path.isdir (self.pathDataset)):
            sys.exit ("FAILURE (0): pathDataset doesn´t exist. Check the method check_conditions (self)")
        
        # Check if pathFolderRois saves a path of a folder and if it exists.
        if not (os.path.exists (self.pathFolderRois) and os.path.isdir (self.pathFolderRois)):
            sys.exit ("FAILURE (1): pathFolderRois doesn´t exist. Check the method check_conditions (self)")

        # Checks if the value is between (0, 1) both include
        if not (0 <= self.confidenceThreshold <= 1):
            sys.exit ("FAILURE (2): confidenceThreshold isn´t between (0, 1). Check the method check_conditions (self)")

        # Check if the list that contains the classes which we are interested in is empty.
        if len (self.classesInterested) <= 0:
            sys.exit ("FAILURE (3): classesInterested is empty. Check the method check_conditions (self)")

        # Check if the folder where it gonna save the json file exists.
        if not os.path.exists (os.path.dirname (self.pathJsonFile)):
            sys.exit ("FAILURE (4): pathJsonFile doesn´t exist. Check the method check_conditions (self)")

        # Supose in folder there are only files. Image (.jpeg, .jpg, .png)
        self.listNameFiles = [nameFile for nameFile in os.listdir(self.pathDataset)]

        indexFile = 0

        while indexFile < len (self.listNameFiles):
            
            # First of all, it gets the data that will need to check if it surpass or not the conditions
            nameFile = self.listNameFiles[indexFile]
            extensionFile = nameFile [nameFile.find ('.') + 1:]
            
            # If it´s not a directoy and it´s a file with extension which is available then pass to the next image
            if (not os.path.isdir(nameFile)) and extensionFile in self.extensionsAvailable:
                indexFile += 1

            # If one of this conditions are not meeting then it has to erase the position of this file/directory.
            else:
                print ("Deleting from list:", self.listNameFiles[indexFile])
                self.listNameFiles.pop (indexFile)

        return True


    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------


    def work_with_detection_results(self, frame, results):
        """
        Process YOLO detection results, filter based on conditions, and save relevant data.

        Parameters:
        - frame: Image frame for which detection is performed.
        - results: YOLO detection results.

        Returns:
        - bool: True if successful.

        """


        counterDetections = 1
        print ("\n\n")

        # Assuming 'results' is a list
        for detection in results[0].boxes:  
            # For each detection (how many objects are there in the image).
            obejctLabel = int (detection.cls.to('cpu')[0])              # Class
            confidence = float (detection.conf.to('cpu')[0])                   # Confindence
            boundingBox = detection.xyxy.to('cpu')[0]                   # Bounding box
            
            # Check if the detectino meets the conditions
            if (obejctLabel in self.classesInterested) & (confidence > self.confidenceThreshold):
                # Add information to json file
                self.save_detection__to_json (obejctLabel, confidence, boundingBox, counterDetections)

                # Shows to user the results of detection (Which meet the conditions)
                print ("Class:", obejctLabel)
                print ("Confidence:", confidence)
                print ("Bounding box:", boundingBox)
                print ("\n------------------------------\n")
                
                self.save_roi_into_disk (frame, boundingBox.numpy (), counterDetections)

                # Increment counter of detections.
                counterDetections += 1
        
        return True
    

    # ------------------------------------------------
    # -------------------- FINISH --------------------
    # ------------------------------------------------

    def data_to_user (self):
        """
        Display and save information about the processed data.

        Returns:
        - bool: True if successful.
        
        """

        # Calculating data
        elapsedTime = time.time() - self.startTime
        self.averageTimePerFrame /= len (self.listNameFiles)

        # Print data
        print ("\n\n--------------- INFORMATION ---------------")
        print (f"Time taken: {elapsedTime} seconds")        
        print (f"Average time per frame: {self.averageTimePerFrame} seconds")
        print (f"Images processed: {self.counterFrames}")
        print ("------------- END INFORMATION -------------\n\n")
        
        # Save json file
        with open (self.pathJsonFile, 'w') as json_file:
            json.dump (self.detectionData, json_file, indent = 4)
        
        return True


    # -----------------------------------------------
    # -------------------- EXTRA --------------------
    # -----------------------------------------------

    def user_interface (self, originalFrame, results):
        """
        Display the user interface with original frame and YOLO detection results.

        Parameters:
        - originalFrame: Original image frame.
        - results: YOLO detection results.

        Returns:
        - bool: True if successful.

        """

        # First of all we create an image with detections
        frameWithDetections = results[0].plot ()
        
        # Then concatenate the images (To be together)
        imagesConcatenated = np.concatenate((originalFrame, frameWithDetections), axis=1) 
        
        # Finally it shows both images
        cv.imshow ("Detections", imagesConcatenated)
        
        # It waits just for 1 sec. If user press ESC then it will finish. For check some errors or testing code
        if cv.waitKey (1) == 27:
            self.data_to_user ()

            sys.exit ("Finalizado con ESC.")
            
        return True

    # ---------------------------------------------------------------------------

    def save_detection__to_json (self, label, confidence, boundingBox, counterDetections):
        """
        Convert detection information to JSON format and update detectionData.

        Parameters:
        - label (int): Class label of the detection.
        - confidence (float): Confidence level of the detection.
        - boundingBox: Bounding box coordinates of the detection.
        - counterDetections (int): Counter for detections.

        Returns:
        - bool: True if successful.

        """

        # First of all we got a key (It could exist already)
        image_name = self.listNameFiles [self.counterFrames][:self.listNameFiles[self.counterFrames].find ('.')]

        print ("image_name:", image_name)

        # Check if it exists, if not just create
        if image_name not in self.detectionData.keys():
            self.detectionData[image_name] = []

        # Get the name of original file
        name_original_image = self.listNameFiles[self.counterFrames]
        name_roi = image_name + "_" + str (counterDetections)

        # convert to numpy to make it easy
        boundingBoxN = boundingBox.numpy()
        
        boundingBoxes = {"x1" : str (boundingBoxN[0]),
                       "y1" : str (boundingBoxN[1]),
                       "x2" : str (boundingBoxN[2]),
                       "y2" : str (boundingBoxN[3])
                       }

        # Generate a dictionary with new data
        newData = {"id_detection" : str (counterDetections),
                   "name_image_original" : name_original_image,
                   "name_roi" : name_roi,
                   "label_detection" : self.classList[label],
                   "confidence:" : str (confidence),
                   "boundingBox" : boundingBoxes
                   }

        # Add the new data of detection into list.
        self.detectionData[image_name].append (newData)

        return True

    def save_roi_into_disk (self, originalImage, boundingBox, counterDetections):
        # First it get the name of roi
        nameRoi = self.listNameFiles[self.counterFrames]   # Copy to variable. Ex. image_101.jpeg
        nameRoi = nameRoi [:nameRoi.find ('.')]      # Name without extension. Ex. image_101
        nameRoi += "_" + str (counterDetections)       # Adding into name the number of detection. Ex. image_101_1
        pathRoi = self.pathFolderRois + nameRoi + ".jpeg"

        # Then it gets the roi from original image with bounding boxes parameters
        roiImage = originalImage [int (boundingBox[1]) : int (boundingBox[3]), 
                                   int (boundingBox[0]) : int (boundingBox[2])
                                   ]

        # Save into disk
        cv.imwrite (pathRoi, roiImage)

        return True


















"""
print ("Counter frames:", self.counterFrames)

print ("Class:", obejctLabel)
print ("Confidence:", confidence)
print ("Bounding box:", boundingBox)

print ("extensionFile:", extensionFile)

# Testing
pathImageTest = "/media/robocomp/externalDisk/dataset/originalImages/image_508.jpeg"

            if os.path.isdir (nameFile):
                print ("is directory")
                print ("name_file:", nameFile)
                print ("extension file:", extensionFile)

            if extensionFile in self.extensionsAvailable:
                print ("no good extension")
                print ("name_file:", nameFile)
                print ("extension file:", extensionFile)
                print ("2extensions:", self.extensionsAvailable)

            #print ("definitivo:", self.listNameFiles)

            
        print ("bounding box [0]:", boundingBox[0])
        print ("bounding box [1]:", boundingBox[1])
        print ("bounding box [2]:", boundingBox[2])
        print ("bounding box [3]:", boundingBox[3])

"""

