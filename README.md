# Guess-the-fruit

Part A: Learning with prototypes

At first, we just used HOG feature extraction technique to make prototypes but for some 
asymmetrical fruits (different orientations have different symmetries) like Banana Lady finger,
Chestnut, Carambula, Corn, Kohlrabi, Hazelnut, Mango Red, Pear 2 and Pear Stone , the accuracy
was below 35%. Hence to overcome this we also made prototypes for the RGB values of the fruits.
We compared both kinds of prototypes with the test images and then appended our predictions in our list "y_pred". Doing this increased the accuracy of our model. The number of features selected for making the feature vector by HOG technique was decided by hit and trial.
Finally our metrics for learning with prototypes model came out as follows:

Accuracy =  58.93 %
Precision =  58.93 %
Recall =  59.189 %
F1 score =  59.06 %

Part B: K nearest - neighbors

We extracted HOG feature i.e. histogram of oriented gradients to increase the accuracy. Since the HOG is providing us the (edge features+edge direction) and then creating histograms of it to define the image better. Hence its more efficient and classifies the image more accurately. We even calculated the RGB value of fruits and calculated the accuracy using that but that accuracy was coming a bit low. So to increase that we extracted HOG feature. To calculate the metrics we stored the predicted classes and true classes in separate lists and then compared using sklearn.metrics. Since running 13100 test images takes approx 1hr, so we took some 70 images as test set and tested on it. The accuracy, precision, recall and F1 score metrics for extracting the HOG feature came as:  

Accuracy = 84.285%
Precision = 84.285%
Recall = 72.619%
F1 score = 78.018%

There were some fruits for which accuracy came as low as 33.33% like Potato Sweet, Apple Braeburn and Watermelon. And for some fruits it came 100% like Apple Pink Lady, Mulberry, Avocado, Blueberry, Mango, Onion Red, etc.


Part C: Making Artificial Neural-Network for classification of the fruits

Firstly, we created lists named images and classNo which contained images of all fruits and
class names respectively. We then splitted the data into training and testing by using sklearn
library. At first we used the images without resizing them thus it took a long time to process
them. We therefore resized them from 100 x 100 to 32 x 32. But it was still taking them a long 
time to get processed , thus we changed all of them to grayscale. Doing this not only increased
the processing speed but it also increased the accuracy of our model. We used libraries like
keras and tensorflow to make the input, hidden and output layers for the artificial neural network
model. To predict anything we had to go through all 12 epochs hence we decided to save our model, which is now named as shweta_model2.h5 . To check it again, we loaded our model and again checked the accuracy of the model on the testing dataset. 
Finally our metrics for ann model came out as follows:

Accuracy on training set : 96.35 %
Loss on training set : 0.1117
Accuracy on testing set : 98.15 %
Loss on testing set : 0.0625

The Precision, recall and F! scores of each class are shown in the Neural_Networks.ipynb file.




