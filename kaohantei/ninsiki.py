from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
import cv2
import sys

dir = "face.jpeg"



img = cv2.imread(sys.argv[1])
filepath = "sys.argv[1]"

cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

for rect in facerect:
    x = rect[0]
    y = rect[1]
    width = rect[2]
    height = rect[3]
    dst = img[y:y+height, x:x+width]


image = cv2.resize(dst, (100, 100))
image = image.transpose(2, 0, 1)


model = load_model('gazou.h5')

opt = Adam(0.0001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

result = model.predict_classes(np.array([image/255.]))

if result[0] == 0:
    print("he may be man1")
elif result[0] == 1:
    print("he may be man2")
    
