# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:36:23 2020

@author: akash
"""
'''we are using 3.75cm markerLenght'''
import cv2
from cv2 import aruco
import yaml
import numpy as np



aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
arucoParams = aruco.DetectorParameters_create()

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
mtx = loadeddict.get('camera_matrix')
dist = loadeddict.get('dist_coeff')
mtx = np.array(mtx)
dist = np.array(dist)

camera = cv2.VideoCapture(0)
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
    #cv2.imshow("original", img_gray)
    if corners == None:
        print ("pass")
    else:
        # estimatePoseSingleMarkers(markerCorners, markerLemgth, cameraMatrix, distCoeffs, rvecs=None, tvecs=None)
        rvec, tvec,_objPoints = aruco.estimatePoseSingleMarkers(corners, 0.0375, newcameramtx, dist) 
        print ("Rotation ", rvec)
        print("Translation", tvec)
        count = 0
        if rvec is not None:
            for index in rvec:
                img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                img_aruco = cv2.drawFrameAxes(img_aruco, newcameramtx, dist, rvec[count], tvec[count], 0.05, 2)    # axis length 100 can be changed according to your requirement
                count +=1
    
    cv2.putText(img_aruco,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
    cv2.imshow("World co-ordinate frame axes", img_aruco)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
          
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc = 1/time1
    #print(frame_rate_calc)  
    
    
    

    
camera.release()
cv2.destroyAllWindows()
