import cv2

image=cv2.imread("4people.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face_cacsade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces= face_cacsade.detectMultiScale(gray,1.1,5)
print(faces)
print(len(faces))

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(125,100,80),2)


    #active region
    region=image[y:y+h,x:x+w]
    cv2.imwrite("facedetection.png",region)

cv2.imshow("fecedetection",image)
cv2.waitKey(0)