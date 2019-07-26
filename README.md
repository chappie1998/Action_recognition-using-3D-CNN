# Action_recognition-using-3D-CNN

A 3D CNN(Convolution neural network) implementation in Keras for action recognition.This model trained on videos. This model architecture achieved  96% accuracy after some hours of training on my GPU. 

The prediction also takes input a video. So, This model can work in real-time. Now free to play around the model,  change numbers of layers, size, data.


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
