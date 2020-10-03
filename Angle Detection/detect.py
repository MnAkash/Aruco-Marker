# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:18:23 2020

@author: akash
"""

import cv2
from custom_aruco import *




cap = cv2.VideoCapture(0)


det_aruco_list = {}

while (True):
	ret,frame = cap.read()
	det_aruco_list = detect_Aruco(frame)
	if(det_aruco_list):
		img = mark_Aruco(frame,det_aruco_list)
		#robot_state = calculate_Robot_State(img,det_aruco_list)
		
	cv2.imshow('image',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()