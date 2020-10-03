# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:12:45 2020

@author: akash
"""

import cv2
from cv2 import aruco
import yaml
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

def drawCubeWireframe(image, cameraMatrix, distCoeffs,  rvec,  tvec, l):
    
    half_l = l / 2.0

    ## project cube points
    axisPoints = np.array([half_l, half_l, l])
    axisPoints = np.vstack((axisPoints, np.array([half_l, -half_l, l])))
    axisPoints = np.vstack((axisPoints, np.array([-half_l, -half_l, l])))
    axisPoints = np.vstack((axisPoints, np.array([-half_l, half_l, l])))
    axisPoints = np.vstack((axisPoints, np.array([half_l, half_l, 0])))
    axisPoints = np.vstack((axisPoints, np.array([half_l, -half_l, 0])))
    axisPoints = np.vstack((axisPoints, np.array([-half_l, -half_l, 0])))
    axisPoints = np.vstack((axisPoints, np.array([-half_l, half_l, 0])))

    #imagePoints = np.zeros((8,2),dtype=np.int)
    #cv.ProjectPoints2(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints, dpdrot=None, dpdt=None, dpdf=None, dpdc=None, dpddist=None) → None 
    imagePoints,jacobian = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
    #print(type(imagePoints))
           
    # draw cube edges lines
    cubeCornerRelation = {
        0 : [1,3,4],
        1 : [2,5],
        2 : [3,6],
        3 : [7],
        4 : [5,7],
        5 : [6],
        6 : [7]
    }
    
    for key in cubeCornerRelation:
        for value in cubeCornerRelation[key]:
            start_point = tuple(imagePoints[key][0].astype(int))
            end_point = tuple(imagePoints[value][0].astype(int))
            cv2.line(image, start_point, end_point, (52,235, 158), 2)
    

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
    #cv2.imshow("original", img_gray)
    if corners == None:
        print ("pass")
    else:
        # estimatePoseSingleMarkers(markerCorners, markerLemgth, cameraMatrix, distCoeffs, rvecs=None, tvecs=None)
        rvec, tvec,_objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, newcameramtx, dist) 
        #print ("Rotation ", rvec)
        #print("Translation", tvec)
        count = 0
        if rvec is not None:
            for index in range(len(ids)):
                #img_aruco = aruco.drawDetectedMarkers(img, corners, ids, (0,255,0))
                #img_aruco = aruco.drawAxis(img_aruco, newcameramtx, dist, rvec[count], tvec[count], markerLength)    # axis length 100 can be changed according to your requirement
                
                #drawing cube using custom function
                drawCubeWireframe(img_aruco, newcameramtx, dist, rvec[count], tvec[count], markerLength)
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

import cv2
from cv2 import aruco
import yaml
import numpy as np
src = np.ones((6, 3))
src[:,1] = 2
src[:,2] = range(6) # source points
rvec = np.array([0,0,0], np.float) # rotation vector
tvec = np.array([0,0,0], np.float) # translation vector
fx = fy = 1.0
cx = cy = 0.0
cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
result,jacobian = cv2.projectPoints(src, rvec, tvec, cameraMatrix, None)
for n in range(len(src)):
    print(src[n], '==>', result[n])
'''