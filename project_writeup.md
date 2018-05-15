# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2018_05_08_14_03_24_315.jpg "Center Image driving"
[image3]: ./examples/center_2018_05_08_14_23_12_809.jpg "Recovery Image"
[image4]: ./examples/center_2018_05_08_14_23_14_251.jpg "Recovery Image"
[image5]: ./examples/center_2018_05_08_14_23_15_544.jpg "Recovery Image"
[image6]: ./examples/normal.png "Normal vs. Flipped Image"
[image7]: ./examples/flipped.png "Normal vs. Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model (data_dir and model_dir need to be changed to retrain the model)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network, more specifically the Nvidia Behavioral Cloning model (5x5x24, pool, 5x5x36, 5x5x48, 3x3x64, 3x3x64, fully connected layers of depth 100, 50, 10).  (model.py line 122:129)

The model includes RELU layers to introduce nonlinearity (model.py line 122:129), and the data is normalized in the model using a Keras lambda layer (model.py line 141). 

#### 2. Attempts to reduce overfitting in the model

Dropout layers attempted to be added in order to reduce overfitting (model.py lines 132, 134, 136). However, the performance of the vehicle actually got worse with dropout.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 83-85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 154).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 2 laps of center lane driving, 1 lap of recovering from the left and right sides of the road, and 1 lap navigating the turns forward and backwards.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take the example, then add small improvements to see which resulted in an overall improvement.

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because it was a basic convolutional neural network that worked well for small inputs.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the dataset to include a more diverse dataset, then expanded the model as well as the convolution sizes to balance the subsequent increase in training error. This made for an overall lower mean-squared error on the validation data than the previous dataset.

Then I tried applying several preprocessing methods, as well as applying dropout. These included grayscaling and adding the left and right images. However, all of these images resulted in a model that performed worse overall.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially on the curves. To improve the driving behavior in these cases, I made sure extra data were recorded on the curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

* 5x5x24

* pool

* 5x5x36

* pool

* 5x5x48

* pool

* 3x3x64

* 3x3x64

* fully connected 100

* fully connected 50

* fully connected 10

* fully connected 1
  (model.py line 122:129)



[//]: # (TODO: add a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric))

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from drifting off the side of the road. These images show what a recovery looks like starting from the left :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would balance any bias the data had toward one turning direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 8574 data points. I then preprocessed this data by cropping the image so that only the road was in view. Additionally, I tried steps such as adding grayscale and adding extra images, but these didn’t seem to help overall performance.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 40000 as evidenced by the sudden flattening of the loss curve after 30 or 35 thousand. I used an adam optimizer so that manually training the learning rate wasn't necessary.
