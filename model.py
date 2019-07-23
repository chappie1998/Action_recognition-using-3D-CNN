
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Art By Ankit<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

#importing libraries
import os
import numpy as np
import cv2
import glob
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model

#load the path of data's directory
path = "action_recognition_dataset/data"
print("loading the path of data's directory:-  " , path)

#define image col, rows and depth
img_rows = 16
img_cols = 16
img_depth = 15

#dataset loading and preparing for training
labels = []
target = []
for label in os.listdir(path):
    labels.append(label)
    for filename in glob.glob(os.path.join(path,label, '*.avi')):
        frames = []
        cap = cv2.VideoCapture(filename)
        for i in range(15):
            ret, frame = cap.read()
            frame = cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            # cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        input = np.array(frames)
        # print("video shape before ratation:- ", gray.shape)
        ipt = np.rollaxis(np.rollaxis(input,2,0),2,0)
        # print("video shape after ratation:- ",ipt.shape)
        target.append(ipt)


#printing some needfull things
print("length of clasese of dataset:- ", labels)
print("length of target dataset:- ", len(target))
print("size of target dataset:- ", np.array(target).shape)

#deining the model's parameters
batch_size = 32
nb_classes = 6
nb_epoch = 500
num_samples = len(target)

#resize the x_train data for model
x_train = []
for tar in target:
    x_train.append(tar.reshape(16, 16, 15, 1))

# Pre-processing of the training data
x_train = np.array(x_train)
x_train = x_train.astype('float32')
x_train -= np.mean(x_train)
x_train /=np.max(x_train)


# labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
#preparing Data for the clasese
label=np.ones((num_samples,),dtype = int)
label[0:100]= 0
label[100:199] = 1
label[199:299] = 2
label[299:399] = 3
label[399:499]= 4
label[499:] = 5

#categorical the label data
Y_train = to_categorical(label, nb_classes)


#Model architature
model = Sequential()
model.add(Convolution3D(32, (5, 5, 5), input_shape=(img_rows, img_cols, img_depth, 1), activation='relu'))
model.add(MaxPooling3D((3, 3, 3)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, kernel_initializer="normal", activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(nb_classes,init='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])

#showing model summary and saving model in .png file
model.summary()
plot_model(model, to_file='model.png')

#traning the model
mk1 = model.fit(x_train, Y_train, batch_size=batch_size, epochs = nb_epoch)

#saving trained model
model.save("mk1.h5")
print("trained model saved")
