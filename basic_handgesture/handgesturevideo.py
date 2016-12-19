import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
while(True):
    ret,frame = cap.read()
    shape = frame.shape
    y1 = shape[0]/3
    y2 = 2*shape[0]/3
    x1,x2 = shape[1]/3,2*shape[1]/3 
    cv2.rectangle(frame,(x2,y2),(x1,y1),(0,255,0),0)
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    new_frame = frame_hsv[y1:y2, x1:x2]
    roihist = cv2.calcHist([new_frame],[0,1], None, [180,256], [0, 180,0,256] )
    cv2.normalize(roihist,roihist,0,180,cv2.NORM_MINMAX)
    bproject = cv2.calcBackProject([frame_hsv],[0,1],roihist,[0,180,0,256],3)
    cv2.GaussianBlur(bproject,(9,9),0)
    cv2.GaussianBlur(bproject,(9,9),0)
    cv2.GaussianBlur(bproject,(9,9),0)
    cv2.GaussianBlur(bproject,(9,9),0)
    ret,thresh1 = cv2.threshold(bproject,250,255,0)
    thresh1 = cv2.merge((thresh1,thresh1,thresh1))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    cv2.filter2D(thresh1,-1,disc,thresh1)
    final = cv2.bitwise_and(frame,thresh1)
    cv2.imshow('final',final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
