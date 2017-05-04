## Behavioral Cloning

---
[//]: # (Image References)

[image1]: pics/nvidia_model.png "Model Visualization"
[image2]: pics/3.jpg "Recovery Image"
[image3]: pics/1.jpg "Normal Image"
[image4]: pics/2.jpg "Flipped Image"

---

#### 1. This repo includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture

#### 1. An appropriate model architecture has been employed

The cnn architecture implemented is one developped by NVidia for self driving cars. Ref. https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
It consists of a normalization step, 3 5x5 convolutions with a 2x2 stride, 2 3x3 convolusions with a single stride and then a fully connected networks with layers of 1164, 100, 50 and 10 neurons.
The model includes RELU layers to introduce nonlinearity. 

![alt text][image1]

#### 2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Training Strategy

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer away from the curb. This image show what a recovery looks like:

![alt text][image2]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increase the dataset at no cost. For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]


After the collection process, I had more than 60'000 samples. I then preprocessed this data by normalizing in the range of [-0.5 , 0.5]


I finally randomly shuffled the data set and put 15% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 5 and 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Training file and performance example

A pre-trained keras file can be downloaded from [this link](https://www.dropbox.com/s/kxodvoysk30qx1k/model.h5.tar.gz?dl=1)

The following low-res video shows an example of the performance of the network driving a car around a track in Udacity's self driving car simulator.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=9VUBIP2Q7iA
" target="_blank"><img src="http://img.youtube.com/vi/9VUBIP2Q7iA/0.jpg" 
alt="Self driving car" width="560" height="315" border="10" /></a> 
