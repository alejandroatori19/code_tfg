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

# Libraries imported
from ultralytics import YOLO            # Neural net managment
import cv2 as cv            # UI & save image
import numpy as np          # Matrix managment
import json         # Json managment
import time              # Time information (Duration)

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Video for understand all about results from neural net
# https://www.youtube.com/watch?v=QtsI0TnwDZs&ab_channel=Ultralytics


class SpecificWorker(GenericWorker):
    # Global variables
    period = 30     # Clock´s period
    startTime = None
    averageTimePerFrame = 0

    # Neural net
    neuralNetModel = None
    classList = []
    confidenceThreshold = 0.5
    
    
    
    # Detections
    detectionData = {}              # Dictionary that saves the data from detections
    classesInterested = []          # List that saves the classes we are interested in
    counterFrames = None            # Counter of frames

    # Dataset
    listNameFiles = []
    pathDataset = "/media/robocomp/externalDisk/dataset/originalImages/"

    
    # Paths
    pathFileClasses = "/home/robocomp/robocomp/code_tfg/detection_people_frames/classes.txt"
    pathFolderRois = "/media/robocomp/externalDisk/dataset/detectionImages/"
    pathJsonFile = "/home/robocomp/data.json"


    pathImageTest = "/media/robocomp/externalDisk/dataset/originalImages/image_508.jpeg"


    # Detections
    extensionRois = ".jpeg"

    # Dataset
    

    
    # Constructor
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        # Setting period
        self.Period = self.period
        self.startTime = time.time ()
        
        # Load neural net
        self.neuralNetModel = YOLO ("yolov8s.pt")
        
        # Read all classes from file with code. (Number that identifies them)
        with open (self.pathFileClasses, 'r') as file:
            self.classList = [line.strip() for line in file]

        # Here you can read a file that contains which clases it will read
        self.classesInterested.append (0)

        # Supose in folder there are only files. Image (.pneg)
        self.listNameFiles = [nameFile for nameFile in os.listdir(self.pathDataset)]

        # Timer starts
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)


    # Destructor
    def __del__(self):

        return    

    # Set parameters
    def setParams(self, params):
        self.counterFrames = 0
        return True


    @QtCore.Slot()
    def compute(self):
        #print('SpecificWorker.compute...')
        
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
            self.applying_detections (frame, results)

            # Show it to the user. Just for testing.
            self.user_interface (frame, results)

            self.counterFrames += 1
        
        else:
            # If there aren´t images available then just type the data into json file.
            self.data_to_user ()

            time.sleep (10)

            # Finally, finish the code
            sys.exit ("Se ha finalizado correctamente")


        return True

    # -------------------------------------------------

    # User interface
    def user_interface (self, originalFrame, results):
        # First of all we create an image with detections
        frameWithDetections = results[0].plot ()
        
        # Then concatenate the images (To be together)
        imagesConcatenated = np.concatenate((originalFrame, frameWithDetections), axis=1) 
        
        # Finally it shows both images
        cv.imshow ("Detections", imagesConcatenated)
        
        # It waits for a key pressed. If it´s ESC it finishes. Only for testing
        if cv.waitKey (1) == 27:
            self.data_to_user ()

            time.sleep (10)

            sys.exit (0)
            
        return

    # Save the information of detection in json
    def detectionToJson (self, label, confidence, boundingBox, counterDetections):
        # First of all we got a key (It could exist already)
        image_name = self.listNameFiles [self.counterFrames][:self.listNameFiles[self.counterFrames].find ('.')]

        print ("image_name:", image_name)

        # Check if it exists, if not just create
        if image_name not in self.detectionData.keys():
            self.detectionData[image_name] = []

        # Get the name of original file
        name_original_image = self.listNameFiles[self.counterFrames]
        name_roi = image_name + "_" + str (counterDetections)

        boundingBoxes = {"x1" : str (boundingBox.numpy()[0]),
                       "y1" : str (boundingBox.numpy()[1]),
                       "x2" : str (boundingBox.numpy()[2]),
                       "y2" : str (boundingBox.numpy()[3])
                       }

        # Generate a dictionary with new data
        newData = {"id_detection" : str (counterDetections),
                   "name_image_original" : name_original_image,
                   "name_roi" : name_roi,
                   "label_detection" : self.classList[label],
                   "confidence:" : str (confidence),
                   "boundingBox" : boundingBoxes
                   }

        self.detectionData[image_name].append (newData)



        return


    # Method that keeps the output data and save it in json file
    def applying_detections(self, frame, results):
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
                self.detectionToJson (obejctLabel, confidence, boundingBox, counterDetections)

                print ("Class:", obejctLabel)
                print ("Confidence:", confidence)
                print ("Bounding box:", boundingBox)
                print ("\n------------------------------\n")
                
                counterDetections += 1

            
            

        
        return
    
       

    def data_to_user (self):
        
        # Generate data
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
            return
        
        

        return







"""
print ("Counter frames:", self.counterFrames)

print ("Class:", obejctLabel)
print ("Confidence:", confidence)
print ("Bounding box:", boundingBox)

"""