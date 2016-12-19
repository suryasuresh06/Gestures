# importing modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
# capturing video through webcam
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(True):
    ret,frame = cap.read()
    shape = frame.shape
    y1 = shape[0]/3
    y2 = 2*shape[0]/3
    x1,x2 = shape[1]/3,2*shape[1]/3 
    cv2.rectangle(frame,(x2,y2),(x1,y1),(0,255,0),0)
    # converting to HSV
    frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # cropping original image to roi
    new_frame = frame_hsv[y1:y2, x1:x2]
    # Histogram of roi
    roihist = cv2.calcHist([new_frame],[0,1], None, [180,256], [0, 180,0,256] )
    cv2.normalize(roihist,roihist,0,180,cv2.NORM_MINMAX)
    # finding back projection
    bproject = cv2.calcBackProject([frame_hsv],[0,1],roihist,[0,180,0,256],3)
    # removing noise 
    cv2.GaussianBlur(bproject,(9,9),0)
    cv2.GaussianBlur(bproject,(9,9),0)
    cv2.GaussianBlur(bproject,(9,9),0)
    cv2.GaussianBlur(bproject,(9,9),0)
    # thresholding
    ret,thresh1 = cv2.threshold(bproject,250,255,0)
    thresh1 = cv2.merge((thresh1,thresh1,thresh1))
    # convolution
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    cv2.filter2D(thresh1,-1,disc,thresh1)
    final = cv2.bitwise_and(frame,thresh1)
    out.write(final)
    cv2.imshow('final',final)
    # escpae key
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
