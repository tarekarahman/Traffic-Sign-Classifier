# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images/aheadonly35.jpg "aheadonly35"
[image2]: ./test_images/bicycle29.jpg "bicycle29"
[image3]: ./test_images/noentry17.jpg "noentry17"
[image4]: ./test_images/pedestrian22.jpg "pedestrian22"
[image5]: ./test_images/wildanimals31.jpg "wildanimals31"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./histogram/hist.png "Histogram"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data. (hist.png)

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

1) I shuffled the data as a part of preprocessing just to make sure that the order in which the data comes does not matters to CNN

2)  I decided to convert the images to grayscale so CNN can focus on shape to maximize discriminity between classes

3)  I normalized the image data to be between 0 and 1

4) I applied dropout after the first fully connected layer for regularization



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Input	32x32x1 grayscale image
Convolution	5x5 filter with 1x1 stride, valid padding, outputs 28x28x6
RELU	
Convolution	5x5 filter with 2x2 stride, valid padding, outputs 14x14x10
RELU	
Convolution	5x5 filter with 1x1 stride, valid padding, outputs 8x8x16
RELU	
Max Pooling	2x2 ksize with 2x2 stride, valid padding, outputs 4x4x16
Flatten	outputs 256
Fully Connected	Input 256 and outputs 120
RELU	
Dropout	keep_prob=0.5
Fully Connected	Inputs 120 and outputs 100
RELU	
Fully Connected	Inputs 100 and outputs 84
RELU	
Fully Connected	Inputs 84 and outputs 43
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model using the following hyperparameters-

Epochs - 25

Batch Size - 64

Learning Rate - 0.001

Optimizer- AdamOptimizer

mu - 0

sigma - 0.1

dropout keep probability - 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.963 
* test set accuracy of 0.939

First, I chose to implement using LeNet Architecture and got around 89% validation accuracy. I needed to adjust a few things to reach 93%.

Then,  I decided to convert the images to grayscale so CNN can focus on shape to maximize discriminity between classes

Third,  I normalized the image data to be between 0 and 1 and accuracy reached around 92.9%. I still needed to do more to reach validation accuracy more than 93%.

Fourth, I increased the epochs from 10 to 25 to train model more efficiently and run the model through the entire data more.

By then, I passed the requirement and I have validation accuracy more than 93%. I then thought to increase perfromance by applying regularization.

 I applied dropout after the first fully connected layer for regularization with 0.5 probability to avoid overfitting. Then my validation accuracy jumped to 96.3% 

My final model consisted of the following layers:

Input	32x32x1 grayscale image
Convolution	5x5 filter with 1x1 stride, valid padding, outputs 28x28x6
RELU	
Convolution	5x5 filter with 2x2 stride, valid padding, outputs 14x14x10
RELU	
Convolution	5x5 filter with 1x1 stride, valid padding, outputs 8x8x16
RELU	
Max Pooling	2x2 ksize with 2x2 stride, valid padding, outputs 4x4x16
Flatten	outputs 256
Fully Connected	Input 256 and outputs 120
RELU	
Dropout	keep_prob=0.5
Fully Connected	Inputs 120 and outputs 100
RELU	
Fully Connected	Inputs 100 and outputs 84
RELU	
Fully Connected	Inputs 84 and outputs 43

I used AdamOptimizer to learn the paramters of my model.


Finalvalidation accuracy reached 96.3%
and test accuracy reached ~94%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. 

Here are five German traffic signs that I found on the web: (if not rendered properly, they can be found in test_images folder)

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

I resized the new test image to 32x32 to adapt to the LeNET skeleton architecture and centered the images. I also applied the same preprocessing steps to the new test images as I did in training (e.g. data shuffling, normalization and converted the images to grayscale).Gererally, to improve classification performance, training images can be augmented by images with various translations, rotations, brightness, contrast ...etc. 
Blur and Resolution affected classification performance in one of the images. To avoid that in the future, images with different levels of resolution can be added during training.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead Only      		| Ahead Only   									| 
| Bicycle Only     			| Bicycle Only 										|
|No Entry					| No Entry											|
| Pedestrians	      		| Pedestrians					 				|
| Wild Animal Crossing			| Road work      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image Bicycles crossing, it was predicted correctly with high probability ~1

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|   9.99901056e-01        			| Bicycles crossing   									| 
| 9.89751425e-05    				| Dangerous curve to the right 										|
| 1.20684851e-09					| Ahead only											|
|  4.47077320e-10	      			| Bumpy road					 				|
| 3.53324731e-10				    | Slippery road     							|




For the second image Pedestrians, it was predicted correctly with high probability ~1

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  9.99603093e-01        			| Pedestrians   									| 
| 3.96950229e-04    				| Right-of-way at the next intersection 										|
| 4.39166342e-10					| Road narrows on the right											|
|  6.18663784e-14	      			| General caution					 				|
| 5.61674906e-16				    | Children crossing      							|


For the third image Ahead only, it was predicted correctly with high probability ~1

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.00000000e+00        			| Ahead only   									| 
|3.16898526e-12    				| Right-of-way at the next intersection 										|
| 1.41629248e-13					| Road narrows on the right											|
|  6.02215099e-15	      			| General caution					 				|
| 6.07433333e-17				    | Children crossing      							|

For the fourth image No entry, it was predicted correctly with high probability ~1

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  1.00000000e+00        			| No entry   									| 
| 1.96009389e-23    				| Traffic signals 										|
|1.36372619e-26					| Stop											|
|  1.44439796e-2	      			| Turn left ahead					 				|
| 2.02138140e-30				    | Children crossing      							|

For the fifth image (Wild animals crossing), the model is relatively sure that this is a Road work even though it is Wild animals crossing. It guessed the second softmax probabilty correctly though.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99998331e-01         			| Road work   									| 
| 1.64936466e-06     				| Wild animals crossing 										|
|  2.74958042e-11					| Double curve											|
| 2.94524469e-13	      			| Bumpy Road					 				|
| 3.47560234e-15				    | Bicycles crossing     							|


INFO:tensorflow:Restoring parameters from ./lenet
[[  5.86026426e-37   0.00000000e+00   2.20188465e-28   8.91764045e-32
    0.00000000e+00   6.49125239e-29   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   2.23494849e-27   2.40038715e-18
    1.66906245e-31   5.54874194e-31   2.66014923e-34   0.00000000e+00
    0.00000000e+00   0.00000000e+00   1.56385164e-18   1.94114567e-16
    5.01946993e-17   2.74958042e-11   2.94524469e-13   2.59830427e-15
    4.58616746e-19   9.99998331e-01   8.11653611e-25   3.06777776e-25
    1.50050654e-36   3.47560234e-15   2.13830754e-17   1.64936466e-06
    0.00000000e+00   7.33309813e-25   5.30543096e-28   4.97413586e-27
    2.33027962e-36   5.14961394e-23   2.32380146e-27   4.00124641e-31
    0.00000000e+00   0.00000000e+00   0.00000000e+00]
 [  2.58911080e-34   5.12652835e-32   2.05452955e-37   2.51348444e-35
    0.00000000e+00   3.21746670e-37   0.00000000e+00   0.00000000e+00
    1.63337322e-35   1.31736663e-37   0.00000000e+00   3.96950229e-04
    9.85244609e-22   0.00000000e+00   0.00000000e+00   0.00000000e+00
    6.30002926e-35   0.00000000e+00   6.18663784e-14   5.21503283e-24
    3.17897408e-23   3.42824682e-20   5.89318154e-37   5.61597967e-23
    4.39166342e-10   2.11418839e-25   8.39476808e-23   9.99603093e-01
    5.61674906e-16   1.22588480e-22   9.05724720e-19   4.75003966e-26
    7.30611516e-38   2.68382668e-37   0.00000000e+00   4.55188778e-35
    3.13762896e-36   2.11842596e-33   0.00000000e+00   0.00000000e+00
    1.55041629e-20   9.26128744e-34   2.82732335e-35]
 [  4.59695503e-23   1.91953372e-34   5.28662979e-36   2.64109654e-17
    0.00000000e+00   3.56538383e-28   0.00000000e+00   1.14833673e-35
    9.32375847e-32   1.41629248e-13   9.66347450e-21   1.83826768e-34
    1.26289645e-19   7.43741154e-23   8.68107231e-21   5.79226762e-21
    1.18301040e-26   2.41111046e-25   1.27122308e-36   1.87358491e-22
    6.97203913e-19   2.73856656e-38   5.36053465e-19   4.09442042e-24
    3.85268516e-37   7.87541456e-18   1.90968976e-25   0.00000000e+00
    9.42343541e-20   3.16898526e-12   7.97152310e-35   6.68608094e-38
    5.06064526e-23   1.47470794e-19   6.07433333e-17   1.00000000e+00
    6.02215099e-15   8.06094443e-33   1.91249119e-23   2.17918386e-36
    4.02620524e-26   2.65515888e-24   2.78605503e-32]
 [  2.02138140e-30   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   1.36560378e-36   0.00000000e+00   0.00000000e+00
    4.38794133e-35   4.30628480e-37   1.36372619e-26   0.00000000e+00
    0.00000000e+00   1.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
    0.00000000e+00   0.00000000e+00   1.96009389e-23   0.00000000e+00
    0.00000000e+00   0.00000000e+00   7.26269657e-35   0.00000000e+00
    1.01239128e-35   1.70543869e-35   1.44439796e-27   0.00000000e+00
    2.55304717e-35   0.00000000e+00   1.55935238e-31   0.00000000e+00
    1.91861966e-33   0.00000000e+00   0.00000000e+00]
 [  3.64576801e-24   7.67946992e-32   8.42647464e-30   2.28027876e-14
    0.00000000e+00   1.24049034e-17   2.91992687e-33   1.84604812e-33
    4.16357711e-34   1.32785918e-18   1.04542569e-19   2.66860255e-25
    1.03323904e-25   6.39891655e-23   1.44585147e-30   5.33650755e-24
    6.80170086e-28   0.00000000e+00   3.11496415e-33   2.06478144e-17
    9.89751425e-05   4.85527785e-31   4.47077320e-10   3.53324731e-10
    3.15580922e-18   5.06219795e-13   3.54909731e-30   2.88203037e-37
    1.94544623e-11   9.99901056e-01   5.16462416e-19   4.88990273e-25
    2.19813622e-29   1.84944137e-30   6.77354277e-13   1.20684851e-09
    1.08930996e-12   2.07442794e-33   2.47400351e-16   1.02046298e-37
    6.03744982e-37   1.36209154e-22   2.23098541e-25]]
TopKV2(values=array([[  9.99998331e-01,   1.64936466e-06,   2.74958042e-11,
          2.94524469e-13,   3.47560234e-15],
       [  9.99603093e-01,   3.96950229e-04,   4.39166342e-10,
          6.18663784e-14,   5.61674906e-16],
       [  1.00000000e+00,   3.16898526e-12,   1.41629248e-13,
          6.02215099e-15,   6.07433333e-17],
       [  1.00000000e+00,   1.96009389e-23,   1.36372619e-26,
          1.44439796e-27,   2.02138140e-30],
       [  9.99901056e-01,   9.89751425e-05,   1.20684851e-09,
          4.47077320e-10,   3.53324731e-10]], dtype=float32), indices=array([[25, 31, 21, 22, 29],
       [27, 11, 24, 18, 28],
       [35, 29,  9, 36, 34],
       [17, 26, 14, 34,  0],
       [29, 20, 35, 22, 23]], dtype=int32))