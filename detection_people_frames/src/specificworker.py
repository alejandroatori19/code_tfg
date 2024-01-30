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

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Video for understand all about results from neural net
# https://www.youtube.com/watch?v=QtsI0TnwDZs&ab_channel=Ultralytics


class SpecificWorker(GenericWorker):
    # Global variables
    period = 2000     # Clock´s period
    
    # Neural net
    neuralNetModel = None
    labelsListInterest = []
    confidenceThreshold = 0.8
    
    # Detections
    detectionData = {}              # Dictionary (Json) which saves the data of detections
    folderRois = "/media/robocomp/externalDisk/dataset/detectionImages/"
    extensionRois = ".jpeg"
        
    # Testing
    pathImageTest = "/media/robocomp/externalDisk/dataset/originalImages/image_508.jpeg"

    
    # Constructor
    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        # Setting period
        self.Period = self.period
        
        # Timer starts
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        # Uploading neural net from file. It must be in the main folder of this component
        self.upload_neural_net_and_resources ()
        

    # Destructor
    def __del__(self):
        
        return    

    # Set parameters
    def setParams(self, params):

        return True


    @QtCore.Slot()
    def compute(self):
        #print('SpecificWorker.compute...')
        frame = cv.imread (self.pathImageTest)
        results = self.neuralNetModel (frame)

        self.applying_detections (frame, results)
                
        self.user_interface (frame, results)

        return True

    # Method that preload the neural net from a file
    def upload_neural_net_and_resources (self):
        # Load neural net
        self.neuralNetModel = YOLO ("yolov8s.pt")
        
        # Here you can read a file that contains which clases it will read
        self.labelsListInterest.append (0)
        
        return

    # User interface
    def user_interface (self, originalFrame, results):
        # First of all we create an image with detections
        frameWithDetections = results[0].plot ()
        
        # Then concatenate the images (To be together)
        imagesConcatenated = np.concatenate((originalFrame, frameWithDetections), axis=1) 
        
        # Finally it shows both images
        cv.imshow ("Detections", imagesConcatenated)
        
        # It waits for a key pressed. If it´s ESC it finishes. Only for testing
        if cv.waitKey (0) == 27:
            sys.exit (0)
            
        return

    # Method that saves the ROIs of people and save data into json file.
    def applying_detections2 (self, frame, results):

        predictions = results.xyxy[0].cpu ().numpy ()

        for det in predictions:
            object_label = int(det[5])  # Assuming class label is at index 5
            confidence = float(det[4])  # Assuming confidence score is at index 4
            bounding_box = det[:4]  # Assuming bounding box coordinates are at indices 0 to 3

            # Your logic for applying detections here
            print(f"Object Label: {object_label}, Confidence: {confidence}, Bounding Box: {bounding_box}")


        """
        # It visualize all the detection per neural net        
        for dataDetection in results[0].boxes:
            # First of all it gets the class label (0 - person), confidencePerDetection and bounding boxes
            obejctLabel = int (dataDetection.cls)               # Class
            confidence = float (dataDetection.conf.numpy ()[0])                    # Confindence
            boundingBox = dataDetection.xyxy.numpy ()                   # Bounding box
    
            print ("objectLabel:", obejctLabel)
            print ("confidence:", confidence)
            print ("boundingbox:", boundingBox)
            
            # We are only interested in people and detections that pass the requirements
            if objectLabel in self.labelsListInterest & confidence > self.confidenceThreshold:
                
                a =0
        """
        
        return

    def applying_detections(self, frame, results):

        # Assuming 'results' is a list
        for detection in results[0].boxes:  
            # For each detection (how many objects are there in the image).
            obejctLabel = int (detection.cls.to('cpu')[0])              # Class
            confidence = float (detection.conf.to('cpu')[0])                   # Confindence
            boundingBox = detection.xyxy.to('cpu')[0]                   # Bounding box
            
            print ("Class:", obejctLabel)
            print ("Confidence:", confidence)
            print ("Bounding box:", boundingBox)
            
            print ("\n------------------------------\n")

        
        return