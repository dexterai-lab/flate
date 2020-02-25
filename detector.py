import cv2
import sys
import numpy as np 

cascPath = "C:\\Users\\User\\Anaconda3\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

#id=input('Enter User ID:')
sampleNum=0

video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not video_capture.isOpened():
    raise IOError("Cannot open webcam")


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=2.5,
        minNeighbors=1,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        sampleNum=sampleNum+1
        #cv2.imwrite("C:/Users/User/Documents/Image processing/flate/dataset/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.waitKey(100)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    #cv2.waitKey(10)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    #if sampleNum>50:
     #   break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()