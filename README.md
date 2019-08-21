# Action_recognition-using-3D-CNN

A 3D CNN(Convolution neural network) implementation in Keras for action recognition.This model trained on videos. This model architecture achieved 96% accuracy after some hours of training on my GPU(RTX 2080TI). 

This model architecture and th prediction scripts are such like that, It works on a video and in real time also. Now free to play around the model, change numbers of layers, size, data.

### The basic concept of action recognition using 3D CNN:-

![alt text](https://github.com/ankitgc1/Action_recognition-using-3D-CNN/blob/master/Action_recognition_using_3D_CNN.jpg)

### Setup
#### [Datasets download](http://www.nada.kth.se/cvap/actions/)
   Create a directory "action_recognition_dataset/data" and put all your downloaded in the data folder. And put all your test data in "action_recognition_dataset" directory. 
#### Dependencies
1. keras
2. glob
3. opencv
4. numpy

### Model architecture
![alt text](https://github.com/ankitgc1/action_recognition-using-3D-CNN/blob/master/model_architecture.png)
