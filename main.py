import cv2
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from tqdm import tqdm
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import pandas as pd
import numpy as np

data_dir = 'images'

def createdataframe(dir):
    img_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            img_paths.append(os.path.join(dir,label, imagename))
            labels.append(label)
    return img_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale= True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

def cnn_network(x_train, y_train, x_test, y_test):
    model = Sequential() # Mạng cnn Sequential

    #Tầng 1
    model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(48,48,1)))

    #Tầng 2
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    #Tầng 3
    model.add(Conv2D(256, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    #Tầng 4
    model.add(Conv2D(512, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))
    
    #Tầng 5
    model.add(Conv2D(512, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    #Tầng 6
    model.add(Flatten())

    # Tầng 7
    model.add(Dense(512))
    model.add(Dropout(0.4))
    # Tầng 8
    model.add(Dense(256))
    model.add(Dropout(0.3))

    #Tầng 9
    model.add(Dense(7, activation='softmax'))



    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    model.fit(x = x_train, y = y_train, batch_size=128, epochs=100, validation_data= (x_test, y_test))

    model.save("emotion_detection.h5")


def extract_feature(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

def start():
    model = load_model("emotion_detection.h5")
    haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade=cv2.CascadeClassifier(haar_file)
    cv2.namedWindow('Nhan dien khuon mat va cam xuc')
    webcam=cv2.VideoCapture(0)
    # class_labels = ['Binh thuong', 'Buon', 'Ghe tom', 'Lo so', 'Ngac nhien', 'Tuc gian', 'Vui ve']
    class_labels  = ['Tuc gian', 'Ghe tom', 'Lo so', 'Vui ve', 'Buon', 'Ngac nhien', 'Binh thuong']


    while True:
        i,im=webcam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(im,1.3,5)
        try: 
            for (p,q,r,s) in faces:
                image = gray[q:q+s,p:p+r]
                cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
                image = cv2.resize(image,(48,48))
                img = extract_feature(image)
                pred = model.predict(img)
                prediction_label = class_labels[pred.argmax()]
                
                cv2.putText(im, '% s' %(prediction_label), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))
            cv2.imshow("Output",im)
            cv2.waitKey(27)
        except cv2.error:
            pass

def training_data():
    train = pd.DataFrame()
    train['imgs'], train['label'] = createdataframe(data_dir)
    createdataframe(data_dir)
    train_features = extract_features(train['imgs'])
    x_train = train_features / 255.0
    x_test = train_features / 255.0
    le = LabelEncoder()
    le.fit(train['label'])

    y_train = le.transform(train['label'])
    y_test = le.transform(train['label'])

    y_train = to_categorical(y_train, num_classes = 7)
    y_test = to_categorical(y_test, num_classes = 7)

    cnn_network(x_train, y_train, x_test, y_test)

def main():
    # training_data()
    start()


main()