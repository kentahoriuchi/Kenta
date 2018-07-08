from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D,  Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
import cv2
import os
import shutil
import random

#リスト作成
image_list = []
label_list = []

#訓練用画像データの読み込み
for dir in os.listdir("date/train"):

    dir1 = "date/train/" + dir
    label = 0

    dir2 = dir1 + "/" + "face"
    if os.path.isdir(dir2):
        shutil.rmtree(dir2)
    os.mkdir(dir2)

    if dir == "man1":
        label = 0
    elif dir == "man2":
        label = 1

    i = 0

    for file in os.listdir(dir1):

        filepath = dir1 + "/" +file

        path  = os.path.splitext(filepath)

        if path[1] == ".jpeg":
            img = cv2.imread(filepath)

            #顔認証　追加部分
            cascade_path = "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)

            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            if len(facerect) > 0:
                label_list.append(label)


                for rect in facerect:
                    x = rect[0]
                    y = rect[1]
                    width = rect[2]
                    height = rect[3]
                    dst = img[y:y+height, x:x+width]
                    new_image_path = dir2 + '/' + str(i) + path[1]
                    cv2.imwrite(new_image_path, dst)
                    i += 1

                image_new = cv2.imread(new_image_path)
                #ここまで

                image = cv2.resize(image_new, (100, 100))
                image_tran = image.transpose(2, 0, 1)

                image_list.append(image_tran/255.)

                for i in range (3):
                    theta = random.randint(-45, 45)
                    label_list.append(label)
                    center = tuple([50, 50])
                    angle = theta
                    scale = 1.0
                    rotation_matorix = cv2.getRotationMatrix2D(center, angle, scale)
                    image_rot = cv2.warpAffine(image, rotation_matorix, tuple([100, 100]), flags=cv2.INTER_CUBIC)

                    image_2 = image_rot.transpose(2, 0, 1)

                    image_list.append(image_2/255.)

image_list = np.array(image_list)

Y = to_categorical(label_list)

#学習部分
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 100, 100)))
model.add(Activation("relu"))
model.add(Convolution2D(32, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode="same"))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(200))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))

opt = Adam(0.0001)

#コンパイル、実行
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])



model.fit(image_list, Y, nb_epoch=100, batch_size=25, validation_split=0.2)

#結果表示
total = 0.
ok_count = 0.

#テスト用画像の読み込み
for dir in os.listdir("date/test"):

    dir1 = "date/test/" + dir
    label = 0

    dir2 = dir1 + "/" + "face"
    if os.path.isdir(dir2):
        shutil.rmtree(dir2)
    os.mkdir(dir2)

    if dir == "man1":
        label = 0
    elif dir == "man2":
        label = 1

    i = 0

    for file in os.listdir(dir1):

        filepath = dir1 + "/" +file

        path  = os.path.splitext(filepath)

        if path[1] == ".jpeg":
            img = cv2.imread(filepath)

            #顔認証　追加部分
            cascade_path = "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)

            image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))

            if len(facerect) > 0:
                label_list.append(label)

                for rect in facerect:
                    x = rect[0]
                    y = rect[1]
                    width = rect[2]
                    height = rect[3]
                    dst = img[y:y+height, x:x+width]
                    new_image_path = dir2 + '/' + str(i) + path[1]
                    cv2.imwrite(new_image_path, dst)
                    i += 1

                image_new = cv2.imread(new_image_path)
                #ここまで

                image = cv2.resize(image_new, (100, 100))
                image = image.transpose(2, 0, 1)



                result = model.predict_classes(np.array([image/255.]))
                print("label:", label, "result:", result[0])

                total += 1.

                if label == result[0]:
                   ok_count +=1

print("seikai: ", ok_count / total * 100, "%")

#save
model.save('gazou.h5')
