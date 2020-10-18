# Real-time Face, Eyes and Nose Detection

# Import necessary libraries
import cv2
import numpy as np

# VideoCapture() is used to initialize the video capture object.
# Read() to capture frame by frame
# imshow() to display it
# cv2.waitkey()
# release() to release the capture

# Load the cascade
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
noise_detect = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Initializing video capturing object
capture = cv2.VideoCapture(0)

# Initialize the VideoWriter
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

# Read frame by frame and convert BGR to GRAY
while True:
    # Start capturing frames
    ret, capturing = capture.read()

    # Convert RGB to gray using cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(capturing, cv2.COLOR_BGR2GRAY)
    face_detection = face_detect.detectMultiScale(gray, 1.3, 5)
    # Rectangles are drawn around the face image using cv2.rectangle function
    for (x, y, w, h) in face_detection:
        cv2.rectangle(capturing, (x, y), (x + w, y + h), (32,32, 255), 2)
    
    # Find the Region Of Interest (ROI) in color image and grayscale image
    gray_roi = gray[y:y + h, x:x + w]
    color_roi = capturing[y:y + h, x:x + w]

    # Apply eye detector on the grayscale Region Of Interest (ROI)
    eye_detector = eyes_detect.detectMultiScale(gray_roi)

    # Rectangles are drawn around the color eyes
    for (eye_x, eye_y, eye_w, eye_h) in eye_detector:
        cv2.rectangle(color_roi, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (255, 0, 0), 5)

    # Apply nose detector in the grayscale ROI
    nose_detector = noise_detect.detectMultiScale(gray_roi, 1.3, 5)

    # Rectangles are drawn around noise
    for (nose_x, nose_y, nose_w, nose_h) in nose_detector:
        cv2.rectangle(color_roi, (nose_x, nose_y), (nose_x + nose_w, nose_y + nose_h), (0, 255, 0), 5)


    # Saving the frame
    if ret==True:
        out.write(capturing)

    # Display the it using imshow built-in function
    cv2.imshow("Real-time Detection", capturing)

    # Check if the user has pressed Esc key
    c = cv2.waitKey(1)
    if c == 27:
        break

# Close the capturing device
capture.release()
# Close all windows
cv2.destroyAllWindows()
