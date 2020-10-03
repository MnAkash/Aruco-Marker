# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:50:43 2020

@author: akash
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:12:45 2020

@author: akash
"""

import cv2
from cv2 import aruco
import yaml
import numpy as np
import math

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

def drawAngle(image, cameraMatrix, distCoeffs,  rvec,  tvec,  angle, centre, dial_end, xAxis_end):
    l=0.0375
    half_l = l / 2.0

    ## project cube points
    points3d = np.array([half_l, half_l, l])
    points3d = np.vstack((points3d, np.array([half_l, -half_l, l])))
    points3d = np.vstack((points3d, np.array([-half_l, -half_l, l])))
    points3d = np.vstack((points3d, np.array([-half_l, half_l, l])))
    '''
    axisPoints = np.vstack((axisPoints, np.array([half_l, half_l, 0])))
    axisPoints = np.vstack((axisPoints, np.array([half_l, -half_l, 0])))
    axisPoints = np.vstack((axisPoints, np.array([-half_l, -half_l, 0])))
    axisPoints = np.vstack((axisPoints, np.array([-half_l, half_l, 0])))
    '''


    #cv.ProjectPoints2(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, dpdrot=None, dpdt=None, dpdf=None, dpdc=None, dpddist=None) → None 
    imagePoints,jacobian = cv2.projectPoints(points3d, rvec, tvec, cameraMatrix, distCoeffs)
    #print(type(imagePoints))

    
    #cv2.line(image, tuple(imagePoints[0][0].astype(int)), tuple(imagePoints[1][0].astype(int)), (52,235, 158), 2)
    #cv2.line(image, tuple(imagePoints[0][0].astype(int)), tuple(imagePoints[2][0].astype(int)), (52,235, 158), 2)
    #cv2.line(image, centre, dial_end, (52,235, 158), 2)
    cv2.circle(img,tuple(imagePoints[0][0].astype(int)),int(line_dist),(255,0,0),8)#outer circle

    
            
            
def angle_calculate(corner,trigger = 0):  # function which returns rotation angle in the range of 0-359
    
    pt1 , pt2 = corner[0], corner[1]
    
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

aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
arucoParams = aruco.DetectorParameters_create()

camera = cv2.VideoCapture(0)
ret, img = camera.read()

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

markerLength = 0.0375           #camera calibrated with 3.75cm marker

ret, img = camera.read()
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
h,  w = img_gray.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

pose_r, pose_t = [], []
while True:
    ret, img = camera.read()
    t1 = cv2.getTickCount()
    
    img_aruco = img
    im_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    h,  w = im_gray.shape[:2]
    dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict, parameters=arucoParams)
    
    if corners == None:
        print ("pass")
    else:
        # estimatePoseSingleMarkers(markerCorners, markerLemgth, cameraMatrix, distCoeffs, rvecs=None, tvecs=None)
        rvec, tvec,_objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, newcameramtx, dist) 
        #print ("Rotation ", rvec)
        #print("Translation", tvec)
        count = 0
        if rvec is not None:
            for corner in corners:
                #img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                #img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec[count], tvec[count], markerLength)    # axis length 100 can be changed according to your requirement
                corner = corner[0] #getting first element of numpy array
                centre = corner[0] + corner[1] + corner[2] + corner[3]#so being numpy array, addition is not list addition
                centre[:] = [int(x / 4) for x in centre]    #finding the centre
                
                dial_end = tuple((corner[0]+corner[1])/2)  # mid point of 1st and 2nd corner point        
                line_dist = math.sqrt((centre[1]-dial_end[1])**2 + (centre[0]-dial_end[0])**2)
                xAxis_end = (int(centre[0]+line_dist), int(centre[1]))  # mid point of 2nd and 3rd point of rectangle for drawing x axis
                centre = tuple(centre)
                angle= angle_calculate(corner)
                #print(angle)
                #drawing cube using custom function
                drawAngle(img_aruco, newcameramtx, dist, rvec[count], tvec[count], angle, centre, dial_end, xAxis_end)
                count +=1
            
    cv2.putText(img_aruco,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.imshow("World co-ordinate frame axes", img_aruco)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
          
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    
camera.release()
cv2.destroyAllWindows()


'''
imagePoints,jacobian = cv2.projectPoints(points3d, rvec[0], tvec[0], newcameramtx, dist)
'''