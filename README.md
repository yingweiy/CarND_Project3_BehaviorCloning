# **Project III: Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./model.png "Model Visualization"
[accuracy]: ./TrainingAccuracy.png "accuracy"
[accuracy-kb]: ./nvidia100.png "kb_accuracy"
[cl]: ./center_lane.jpg
[cl_f]: ./center_lane_flipped.jpg
[left_rc1]: ./left_cover1.jpg
[left_rc2]: ./left_cover2.jpg
[left_rc3]: ./left_cover3.jpg


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode - no modifications
* model.h5 containing a trained convolution neural network 
* writeup_report.md or README.md summarizing the results - this report
* video.mp4 recording the simulation results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the default nvidia architect, which consists of five convolutional layers with 5x5 or 3x3 filter sizes 
and depths between 24 and 64 (model.py lines 61-65) 

    model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64,(3,3), activation='relu'))
    model.add(Convolution2D(64,(3,3), activation='relu'))

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a
 Keras lambda layer (code line 59). 

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    
#### 2. Attempts to reduce overfitting in the model

I first tried using the dropout layer approach, but the results are not good. Somehow, the vehicle cannot stay in the 
track after adding this layer. So I do not including this layer in the final model.

The model was trained and validated on different data sets to ensure that the model was not overfitting 
(code line 34-35). The model was tested by running it through the simulator and ensuring that the vehicle 
could stay on the track.

* The dataset I that is generated with keyboard control:

    csvfn = '/home/yingweiy/SDC/udacity_code/Project3_Data/driving_log.csv'

* The dataset II that is generated with mouse control (much smoother):

    csvfn = './training_data/driving_log.csv'

In addition, I found my initial model was over-complex, 1024 neurons in the first fully-connected layer. This 
also causes the over-fitting problem. I reduced it to 100 neurons to reduce the overfitting problem.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72).

    model.compile(loss='mse', optimizer='adam')

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving,
 recovering from the left and right sides of the road, and dring clock-wise and counter-clock-wise. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to achieve a high accuracy of controlling the steering.

My first step was to use a convolution neural network model similar to the nvidia's default network, but I used 1024 
neurons in the first fully connected layer, instead of default 100 neurons.  
I thought this model might be appropriate because it has the basic feature extraction layers (e.g., the convolutional layers)
as well as the inference layers, i.e., four fully-connected layers. The 1024 neurons will be large enough 
to memorize the variaty of features extracted from lower levels.

In order to gauge how well the model was working, I split my image and steering angle data into a training and
 validation set. I found that my first model had a low mean squared error on the training set but a high mean squared 
 error on the validation set. This implied that the model was overfitting. I realized that this is due to the training dataset 
 that I am using. The first training dataset is generated using keyboard, and the problem with it is that the steering 
 control value is very unstable, because when the key is up, the value suddenly comes back to 0. This introduces a lot of 
 overfitting problem. 
 
As shown in the figure below, the training accuracy can reach a very low value, 0.004, but the validation set has a 
very high value around 0.035.
   
![alt text][accuracy-kb]

To combat the overfitting, I did the follows:

* Reduced the fully connected layer neurons from 1024 to 100 neurons only.
* Instead of using keyboard, using a mouse to control the car to get a much smoother training dataset.
* Reduced the training epoches from 30 to 10. 

After these fixes, the overfitting problem is reduced. The accuracy of the validation set gets down to 0.014.

![alt text][accuracy]

The final step was to run the simulator to see how well the car was driving around track one. There were a few 
spots where the vehicle fell off the track such as at the turns before and after the bridge.
 
To improve the driving behavior in these cases, I added more training data by recording the drivings around those 
segments. In addition, I also drived a lap clockwise to generate some right turn data points. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
Click below to play the recorded video on YouTube:

[![run result](https://img.youtube.com/vi/e286o46tYuk/0.jpg)](https://youtu.be/e286o46tYuk)

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-70) consisted of a convolution neural network with the 
following layers and layer sizes:

    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,(5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(36,(5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(48,(5,5), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64,(3,3), activation='relu'))
    model.add(Convolution2D(64,(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. 
Here is an example image of center lane driving:

![alt text][cl]

I then recorded the vehicle recovering from the left side and right sides of the road back to center 
so that the vehicle would learn to recover the position from sides. 
These images show as an example of what a recovery looks like starting from the left side:

![alt text][left_rc1]

![alt text][left_rc2]

![alt text][left_rc3]

I also used the images from the left and right cameras as suggested by the video instruction, with corrections 
of 0.2 (positive/negative corresponding to their side):

    loadImage(log, 'LeftImage', correction=0.2)
    loadImage(log, 'RightImage', correction=-0.2)


To augment the dataset, I also flipped images and angles thinking that this would enrich the dataset for training.
 For example, here is an image that has then been flipped:

![alt text][cl]
![alt text][cl_f]

After the collection process, I had 33141*4=132,546 number of data points. I then preprocessed this data by cropping the image
as shown in the course video (see line 60 in model.py):

    model.add(Cropping2D(cropping=((70,25), (0,0)))) 

I finally randomly shuffled the data set and put 10% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was 
over or under fitting. The ideal number of epochs was 6 as evidenced by the minimum validation error as shown in the 
accuracy figure. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.
