# OpenCV program to perform Edge detection in real time 
# import libraries
import cv2
cap = cv2.VideoCapture("motion.mp4")         # capture frames from a video
while(1):                         # loop runs if capturing has been initialized
    ret, frame = cap.read()       # reads frames from a camera
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # converting BGR to HSV
    res, thresh = cv2.threshold(hsv[:,:,0], 25, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Original',frame)         # Display an original image
    edges = cv2.Canny(frame,80,180)
    cv2.imshow('Video_with_Edges',edges)      # Display edges in a frame
    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyAllWindows()
        break
