# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 20:38:36 2020

@author: akash
"""

import cv2
import cv2.aruco as aruco
import math
import yaml

'''
functions in this file:
* angle_calculate(pt1,pt2, trigger = 0) - function to return angle between two points
* detect_Aruco(img) - returns the detected aruco list dictionary with id: corners
* mark_Aruco(img, aruco_list) - function to mark the centre and display the id
'''
def angle_calculate(pt1,pt2, trigger = 0):  # function which returns rotation angle in the range of 0-359
    angle_list_1 = list(range(359,0,-1))
    #angle_list_1 = angle_list_1[90:] + angle_list_1[:90]
    angle_list_2 = list(range(359,0,-1))
    angle_list_2 = angle_list_2[-90:] + angle_list_2[:-90]
    x=pt2[0]-pt1[0] # unpacking tuple
    y=pt2[1]-pt1[1]
    angle=int(math.degrees(math.atan2(y,x))) #takes 2 points nad give angle with respect to horizontal axis in range(-180,180)
    if trigger == 0:
        angle = angle_list_2[angle]
    else:
        angle = angle_list_1[angle]
    return int(angle)

def detect_Aruco(img):  #returns the detected aruco list dictionary with id: corners
    aruco_list = {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)   #creating aruco_dict with 6x6 bits with max 250 ids..so ids ranges from 0-249
    parameters = aruco.DetectorParameters_create()  #refer opencv page for clarification
    #lists of ids and the corners beloning to each id
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
    #corners is the list of corners(numpy array) of the detected markers. For each marker, its four corners are returned in their original order (which is clockwise starting with top left). So, the first corner is the top left corner, followed by the top right, bottom right and bottom left.
    
    if len(corners):    #returns no of arucos
        #print (len(corners))
        #print (len(ids))
        #print(type(corners))
        #print (corners[0][0])
        for k in range(len(corners)):
            temp_1 = corners[k]
            temp_1 = temp_1[0]
            temp_2 = ids[k]
            temp_2 = temp_2[0]
            aruco_list[temp_2] = temp_1
        return aruco_list

def mark_Aruco(img, aruco_list):    #function to mark the centre and display the id
    key_list = aruco_list.keys()
    font = cv2.FONT_HERSHEY_SIMPLEX
    robot_state = {}
    for key in key_list:
        dict_entry = aruco_list[key]    #dict_entry is a numpy array with shape (4,2)
        centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]#so being numpy array, addition is not list addition
        centre[:] = [int(x / 4) for x in centre]    #finding the centre

        orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)  # mid point of 1st and 2nd point of rectangle
        
        line_dist = math.sqrt((centre[1]-orient_centre[1])**2 + (centre[0]-orient_centre[0])**2)
        xAxis_end = (int(centre[0]+line_dist), int(centre[1]))  # mid point of 2nd and 3rd point of rectangle for drawing x axis

        pt1 , pt2 = tuple(dict_entry[0]) , tuple(dict_entry[1])
        angle = angle_calculate(pt1, pt2)        
        
        centre = tuple(centre)  
        cv2.circle(img,centre,1,(0,0,255),8)             #marking the centre of aruco
        cv2.circle(img,centre,int(line_dist),(255,0,0),8)#outer circle
        
        cv2.line(img,centre,orient_centre,(0,0,255),4)  #angle pointing dial
        cv2.line(img,centre,xAxis_end,(255,0,0),4)      #reference axis
        
        #Show angle at the top
        cv2.putText(img, str(angle), (int(centre[0]-20), int(centre[1]-10)-int(line_dist)), font, 1, (0,0,255), 2, cv2.LINE_AA)
        
        # displaying the id no
        #cv2.putText(img, str(key), (int(centre[0] + 20), int(centre[1])), font, 1, (0,0,255), 2, cv2.LINE_AA)
        
        robot_state[key] = (int(centre[0]), int(centre[1]), angle)#HOWEVER IF YOU ARE SCALING IMAGE AND ALL...THEN BETTER INVERT X AND Y...COZ THEN ONLY THE RATIO BECOMES SAME
        
    return img,robot_state


if __name__ == '__main__':
    print("custom_aruco library compiled")
