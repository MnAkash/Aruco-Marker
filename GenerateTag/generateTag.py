# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:15:45 2020

@author: akash
"""


import cv2 as cv
import numpy as np

size = 600
ID = 30
# Load the predefined dictionary
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_1000)

# Generate the marker
markerImage = np.zeros((size, size), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, ID, size, markerImage, 1);

cv.imwrite("marker"+str(ID)+".png", markerImage);


'''uncomment following block to draw and show the board'''
#board = cv.aruco.GridBoard_create(4, 5, 3.75, 0.5, dictionary)
#img = board.draw((864,1080))
#cv.imshow("aruco", img)
#cv.waitKey()