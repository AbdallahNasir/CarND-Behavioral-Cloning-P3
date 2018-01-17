# Behavioral Cloning
## Introduction
This project is about training a model to drive a car in a simulator using a data from a manual run. 

## Data Gathering
The data being used is collected using the simulater. The car was driven two laps, with some recovery and some hard parts like the bridge were driven by more than once in a lap. The data included clockwise and anti-clockwise drive to generalize the model as much as possible.

Below is a sample of the data being used in the training.

![alt text](/resources/center_2017_12_28_12_03_39_164.jpg "A center image while steering in a corner")

## Data Preprocessing.
Right and left images were used with a correction of 0.25, which was chosen based on expiremntal trials. Each image was flipped horizontally to remove bias towards left or right.

## Building the model
### LeNet
Inspired by LeNet, I started develpment with two convnets layer model, with maxpooling and activation and dropouts after the convnets. Dropouts and maxpooling layers were used to reduce overfitting.

The results were not good, at driving, as the model barely could drive half of the track. 

### Final Solution
I increased the convnets layers by one, making the netweork deeper, but not as deep as the NVidea netweork, as the problem was not that compicated.
## Model Layers
The model consists of the following layers.

![alt text](/resources/model.png "Model Architecture")

Initially, the image got cropped, from top, to remove sky and mountains, and from bottom to remove the hood. The following image is a sample output from cropping.

![alt text](/resources/center_2018_01_04_16_54_29_409_CROPPED.jpg "Cropped image")

The image got normalized, centered around zero with a little deviation.

A total of three convnets were used, each followed by maxpooling and dropouts layer to reduce overfitting, and the activation used is relu.

After convnets, a two hidden fully connected layers are used. Two is enough, as going deeper will increase the possibility of over fitting.

## Training the model
The model was trained in two epochs only. This number came after several trials, of sometimes 10 epochs, and was noticed that the validation loss does not change usually after the second epoch. Adam optimizer was used, and the loss was computed using mean squared error.

## Struggles
The first trials were no use at all. The car kept bending to the left and was out of track quikly. It took some time to realize that the images in training are read as BGR by opencv, while the images in the driving is read as RGB.

Data collection was not easy, driving the car was harder than coming up with the best solution. 

Figuring which part of the project causes the car not to move well was not easy at the beginning. I could not identify if there is a problem with the data, or it is the model. Looking into the training and validation loss resolved this issue. Since they were low, then it must be a data issue. If the training loss was low, while the validation loss was high, that mean an over fitting issue, and the model is to deep, or a dropouts or regularization is needed.

## Improvements
1. Driving for track 2 should be considered.
2. The car keep bouncing between left and right side of the drivable area, which suggests to renew the data with something better. For example the following image is a center image, and the car is very close to the left side of the drivable area, which leads to that bouncing.

![alt text](/resources/center_2017_12_28_12_03_39_164.jpg "A center image while steering in a corner")

3. More data augmentation techniques can be used.
