# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/hist_train.png "Visualization-train"
[image2]: ./examples/hist_test.png "Visualization-test"
[image3]: ./examples/hist_valid.png "Visualization-valid"
[image4]: ./test_images/17.jpg "Traffic Sign 1"
[image5]: ./test_images/14.jpg "Traffic Sign 2"
[image6]: ./test_images/25.jpg "Traffic Sign 3"
[image7]: ./test_images/02.jpg "Traffic Sign 4"
[image8]: ./test_images/22.jpg "Traffic Sign 5"
[image9]: ./examples/sample_images.PNG "Visualization-sample images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/thayumaanavan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.  As per below charts, the amount of examples per class in every dataset show similar shapes and regardless from not having an uniform distributions, the amount of examples per class in the training set is proportional to the same class in validation and test datasets per class.

Training                   |  Validation               |  Testing
:-------------------------:|:-------------------------:|:-------------------------:
![][image1]                |  ![][image2]              |  ![][image3]

The images come in RGB format and are already contained in numpy arrays and the labels come as a list of integers.

*Image data shape = (32, 32, 3)
*Number of classes = 43

Every image has a corresponding label and these labels correspond to a category of traffic signal. These categories can be seen in the file signnames.csv.

![][image9] 


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The pre-processing pipeline consists in three different steps. 

* Grayscale: The RGB channels disappear and only one channel corresponding to intensities remain.This allows to reduce the numbers of channels in the input of the network without decreasing the performance. In fact, as Pierre Sermanet and Yann LeCun mentioned in their paper ["Traffic Sign Recognition with Multi-Scale Convolutional Networks"](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), using color channels did not seem to improve the classification accuracy.
```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
* Histogram equalization: The contrast of the image is enhanced obtaining a more uniform histogram.
```
gray_equalized = cv2.equalizeHist(gray)
```
* Normalization: The values of the intensity image no longer go from 0 to 255, but they range now from -1 to 1 in floating point format. Still using Matplotlib visualization the image looks identical to equalized one.
```
norm_image = (gray_equalized - 128.0)/ 128.0
```

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  | outputs 400 |
| Fully connected		| Input = 400 and Output = 120        							|
| RELU					|												|
| Fully connected		| Input = 120 and Output = 84        							|
| RELU					|												|
| Fully connected		| Input = 84 and Output = 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used below parameter values.
* Optimizer: AdamOptimizer
* batch size : 128
* epochs : 50
* Learning Rate : 0.0009

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.946 (94.6%)
* test set accuracy of 0.931(93.1%)

I used the same [LeNet architecture](http://yann.lecun.com/exdb/lenet/) and preprocessing codes used in the LeNet lab class. But not able to get beyond 89%.
Then I improved the preprocessing steps followed in the paper
["Traffic Sign Recognition with Multi-Scale Convolutional Networks"](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and also added histogram equalization.
Also, I have tweaked the learning rate(to 0.0009) and epoch (to 50).
After this change, I was able to improve the accuracy of the model as mentioned above.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

Test 1                     |  Test 2                   |  Test 3   
:-------------------------:|:-------------------------:|:-------------------------:
![][image4]               |  ![][image5]             |  ![][image6]              

Test 4                     |  Test 5                   
:-------------------------:|:-------------------------:
![][image7]               |  ![][image8]             


Below are the qualities of test images which might be difficult to classify:
The images might be difficult to classify because they are in different size, with water mark and clear shape with no shadow.
* Test 1 & 4 have watermarks.
* Test 3 is tilted slightly which may cause difficulty for the model.
* Test 5 has leaves in some part of the shape and also shadow falling on it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 17      		        | 17  									        | 
| 14     			    | 14							    |
| 25					| 25				|
| 2	      		    | 1		 				        |
| 22			        | 25      							            |

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

These are the top 5 probabilities
```[[  1.00000000e+00   1.19873390e-13   7.82299354e-16   1.72693960e-18
    1.31664106e-18]
 [  9.98637855e-01   8.75854457e-04   3.48322385e-04   4.75172637e-05
    4.59054681e-05]
 [  1.00000000e+00   3.83663325e-18   1.29936029e-21   9.04603366e-25
    2.77326640e-25]
 [  9.99998093e-01   1.87811895e-06   6.92120250e-11   2.86236353e-14
    4.05694689e-15]
 [  9.99978423e-01   1.95940811e-05   1.22328186e-06   8.53118195e-07
    5.78888670e-09]]
 ```
 
 These are corresponding classes of probabilities
 ```
 [[17 26  0 13 40]
 [14 38  3  8 26]
 [25 22 36  5 38]
 [ 1  6  0 10  5]
 [25 29 22 31 18]]
 ```

* Correct class: 17, 5 most probable classes: [17 26  0 13 40]
* Correct class: 14, 5 most probable classes: [14 38  3  8 26]
* Correct class: 25, 5 most probable classes: [25 22 36  5 38]
* Correct class: 2, 5 most probable classes: [ 1  6  0 10  5]
* Correct class: 22, 5 most probable classes: [25 29 22 31 18]

For class 2 image, it's value is not found in the top 5 probable class itself. For class 22, it's value is found in 3rd place.
