import cv2
import numpy as np
from PIL import Image
import os
import pickle
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,"images")

face_cascade = cv2.CascadeClassifier("D:/haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
lable_ids = {}
y_lables = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            lable = os.path.basename(root).replace(" ", "-").lower()
            #print(lable, path)
            if not lable in lable_ids:
                lable_ids[lable] = current_id
                current_id+=1
            id_ = lable_ids[lable]
            #print(lable_ids)
            #y_lables.append(lable)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_lables.append(id_)

#print(y_lables)
#print(x_train)
with open("lables.pickle","wb") as f:
    pickle.dump(lable_ids, f)

recognizer.train(x_train, np.array(y_lables))
recognizer.save("trainner.yml")