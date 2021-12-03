import cv2
import numpy as np

cap = cv2.VideoCapture(0)
path = "D:/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(path)
X = []
padding = 10
name = input("Enter your name:")
while True:
    ret, image = cap.read()
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(image)
    print(faces)
    for face in faces:
        x, y, w, h = face
        face_section = image[y-padding:y+h+padding, x-padding:x+w+padding]
        face_section = cv2.resize(face_section, (100, 100))
    cv2.imshow("Camera", face_section)
    X.append(face_section.reshape(1,-1))
    print(len(X),X[-1].shape)
    key_pressed = cv2.waitKey(25)
    if key_pressed == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
x = np.asarray(X)
np.save("C:/Users/NIkhil/Desktop/ML-face-recog-data"+'/'+name+'.npy',X)