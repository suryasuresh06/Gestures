import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import math
# reading image
img = cv2.imread('suryagesture.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
shape = img.shape
print shape
y = shape[0]
x = shape[1]

cv2.rectangle(img,((2*x)/3,(2*y)/3),(x/3,y/3),0)
# cropping the image
crop_img = img[250:500,500:770]
grey = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
# Gaussian Blur
blur_img = cv2.GaussianBlur(grey,(45,45),0)
# binarization
ret, thresh = cv2.threshold(blur_img, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# finding contours
image, contours, hierarchy = cv2.findContours(thresh.copy(), \
                               cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# the important contour

cnt = max(contours, key = lambda x: cv2.contourArea(x))
# to make the contours visible

thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
cnt1 = contours[0]
# drawing contours
cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
hull = cv2.convexHull(cnt,returnPoints=False)
# finding convexity defects
defects = cv2.convexityDefects(cnt,hull)
count=0
count1 = 0
count2 = 0
# tolerance limits for angles
tol = 15
tol1 = 30
for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(thresh,start,end,[0,0,255],4)
        
        # measuring distances to implement cosine rule
        l1 = sqrt(((start[0] - far[0])**2) + ((start[1] - far[1])**2))
        l2 = sqrt(((far[0] - end[0])**2) + ((far[1] - end[1])**2))
        l3 = sqrt(((start[0] - end[0])**2) + ((start[1] - end[1])**2))
        # cosine rule
        cos_thetha = ((l2**2) + (l1**2) - (l3**2))/(2*l1*l2)
        thetha = math.acos(cos_thetha)
        # angle in degrees
        thetha = math.degrees(thetha)
        
        # gap between fingers
        if(thetha<90):
             count = count+1
            
        
        # in right and left gestures part of the hand is flat while a part forms a right angle   
        if(thetha<(180+tol) and thetha>(180-tol)):
              count2 = count2+1
        if(thetha<(90+tol1) and thetha>(90-tol1)):
              count1 = count1 + 1     
                         
             
        
#cv2.imshow('thresh',thresh)
if(count==4):
             print " hi gesture"
             res = np.vstack((crop_img,thresh))
             cv2.imshow('hi gesture',res)
elif(count==1):
             print " victory"
             res = np.vstack((crop_img,thresh))
             cv2.imshow('victory',res)
elif(count2>=1 and count1==1):
        if(thresh[240,250].all() == 0):
             print " left"
             res = np.vstack((crop_img,thresh))
             cv2.imshow('left',res)
        else:
             print "right"
             res = np.vstack((crop_img,thresh))
             cv2.imshow('right',res)

res = np.vstack((crop_img,thresh))
#cv2.imshow('result',res)

cv2.waitKey(0)

