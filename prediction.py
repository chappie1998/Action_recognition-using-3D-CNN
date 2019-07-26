
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Art By Ankit<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

#importing libraries
import os
import numpy as np
import cv2
from keras.models import load_model


#load the path of data's directory
path = "action_recognition_dataset/person15_boxing_d1_uncomp.avi"
print("loading the path of data's directory:-  " , path)

#labels for target
labels = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

# load the model we saved and compile  the model
model = load_model('mk1.h5')
model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics=["accuracy"])

#font of text
font = cv2.FONT_HERSHEY_SIMPLEX

#define image col, rows and depth
img_rows = 16
img_cols = 16
img_depth = 15

#preprossing the video for the model
cap = cv2.VideoCapture(0)
while (1):
    ret, video = cap.read()
    frame = video.copy()
    frames = []
    for i in range(15):
        frame = cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)


    input = np.array(frames)
    # print("video shape before ratation:- ", input.shape)
    input = np.rollaxis(np.rollaxis(input,2,0),2,0)
    # print("video shape after ratation:- ",ipt.shape)

    # print("size of target dataset:- ", np.array(input).shape)

    #reshape the input as model required
    input = input.reshape(1, 16, 16, 15, 1)

    #predicting the model output
    out = model.predict(input, verbose=0)
    # print(out)

    #converting output in label's format
    for i in range(len(labels)):
        if out[0][i] == 1:
            action = labels[i]

    #putting the predicted action on the frame
    cv2.putText(video,action,(10,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('frame',video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
