import cv2, time

first_frame=None  #not undefined
#0, 1...indicate camera in laptop
#filename
video=cv2.VideoCapture(0)

while True:
    #check is boolean
    #frame is numpy array, the very first frame
    check, frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21,21), 0) #remove noise

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    #over 30, assign 255, less than 30, assign 0
    #[0] for other thresh methods, will return a recommended threshold
    thresh_frame=cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=1)

    (_,cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for countour in cnts:
        if cv2.contourArea(countour) < 2000:
            continue
        (x,y,w,h)=cv2.boundingRect(countour)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow("Motion Detector", frame)
    #cv2.imshow("Gray",gray)
    #cv2.imshow("Delta",delta_frame)
    #cv2.imshow("Thresh",thresh_frame)

    key=cv2.waitKey(10)
    if(key == ord('q')):
        break;

video.release() #release the camera, stop record
cv2.destroyAllWindows()
