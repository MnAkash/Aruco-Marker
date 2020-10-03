# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 23:18:01 2020

@author: akash
"""


import cv2
import numpy as np

#Load the dictionary that was used to generate the markers.
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters_create()
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    grey= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the markers in the image
    corners, ids, rejectedCandidates = cv2.aruco.detectMarkers(grey, dictionary, parameters=parameters)
    frame = cv2.aruco.drawDetectedMarkers(frame, corners,ids)
    
    cv2.imshow('frame',frame)
    #print(ids)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

