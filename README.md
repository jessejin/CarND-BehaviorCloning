#**Behavioral Cloning** 
---
**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_image.png "Center Image"
[image2]: ./examples/training_steering_hist.png "Training data set steering angles histogram"
[image3]: ./examples/crop_resize.png "Cropped unneccessary area and resized to 64x64"
[image4]: ./examples/Flip.png "Flipped image"
[image5]: ./examples/random_brightness.png "Random brightness applied"
[image6]: ./examples/architecture.png "Random brightness applied"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* readme.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy
####1. Final Model Architecture
![alt text][image6]




The model contains dropout layers in order to reduce overfitting (model.py line 110 and line 113). 
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 131).

####2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded more images when the car is turning since the car is driving straight most of the time. The steering histogram shows this.
![alt text][image2]

I first tried to use these original images to train the model with many tries, but the car would not even past the first turn. 

With mentor's direction, I started looking at augment my data with:
1. Cropping unneccessary area from the image to reduce noise
2. Flipping the image to create more data set since the training track only has one right turn.
3. Adding random brightness to the image to reduce over fitting.

![alt text][image3]
![alt text][image4]
![alt text][image5]

After the collection process, I had 6417 of data points. 
I randomly shuffled the data set and put 10% of the data into a validation set. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 


###Reflections
I spent countless hours trying to training the deep neural network correctly. After many tries and debugs, I was able to find the tips to figure out where the problem resides.
Most of the time a good designed training model would work. By testing the model on the simulator and find out where the car fails will indicates the weakness of the training data.
By augmenting the data where the car fails, it helped significantly improving the training.

