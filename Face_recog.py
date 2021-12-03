import cv2
import pickle
import numpy as np
face_cascade = cv2.CascadeClassifier("D:/haarcascade_frontalface_alt.xml")
#eye_cascade = cv2.CascadeClassifier("D:\haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
lables = {}
with open("lables.pickle","rb") as f:
    og_lables = pickle.load(f)
    lables = {v:k for k, v in og_lables.items()}

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45: #and conf<=85:
            #print(id_)
            #print(lables[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = lables[id_]
            cv2.putText(img, name, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item,roi_color)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        #eyes = eye_cascade.detectMultiScale(roi_gray)
        #for (ex, ey, ew, eh) in eyes:
            #cv2.rectangle(roi_color,(ex, ey), (ex+ew,ey+eh),(0, 255,0),2)

    cv2.imshow('img',img)
    k = cv2.waitKey(20)&0xff
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()