import cv2

face_cacsade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video=cv2.VideoCapture(0)
while True:
    dummy,frame= video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= face_cacsade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    cv2.imshow("myface",frame)
    cv2.imwrite("facedetection.png",frame)
    if cv2.waitKey(25) == 32:
        break

video.release()
cv2.destroyAllWindows()



