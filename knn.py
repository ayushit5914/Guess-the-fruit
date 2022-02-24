import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import precision_score, recall_score, accuracy_score

train='fruits-360/Training'

labels=os.listdir(train)
# converting image to rgb array=flattening
def to_flatten(image,size=(32,32)):
  im = cv2.resize(image,size)
  return im.flatten()
#calculating euclidean distance between two vectors 
def distance(a,b):
  a=np.array(a)
  b=np.array(b)
  dist = np.linalg.norm(b-a)
  return dist
#calculating the final class of the data point after calculating its k-nearest neighbors
def k_nearest(training_data, test_data, k):
  dis = list()
  nearest_neighbors=list()
  for b in range(len(training_data)):
    d=distance(test_data, training_data[b][1])
    dis.append((training_data[b][0],d))
  sorted_dis=sorted(dis, key=lambda x: x[1])
  for c in range(k):
    nearest_neighbors.append(sorted_dis[c])
  output_values=[row[0] for row in nearest_neighbors]
  result=max(set(output_values), key=output_values.count)
  return result         

training_image_rgb=list() # storing tuples of each class and its rgb array in list 
training_image_hog=list() # storing tuples of each class and its hog vector array in list
# training each dataset
for i in labels:
  knn=np.zeros((1100))
  count=0
  rgb_mean=np.zeros((3072))
  for img in os.listdir(f'{train}/{i}'):
    image=cv2.imread(f'{train}/{i}/{img}')
    hog_array=hog(image, orientations=11, pixels_per_cell=(15,15), cells_per_block=(2,2), channel_axis=-1)#calculating hog vector
    training_image_hog.append((i,hog_array))
    rgb_array=to_flatten(np.array(image),size=(32,32))#calculating rgb vector
    training_image_rgb.append((i,rgb_array))
print("All classes done")

test='fruits-360/Test1'
urls=os.listdir(test)
predicted_class=[]# storing predicted class of each point in a list
true_class=[]# storing true class of each pint in a list
#testing each dataset
for i in urls:
  pred=[]# list storing the predicted class of each test fruit for printing the accuracy of each fruit
  true=[]# list storing the actual class of each test fruit for printing the accuracy of each fruit 
  for img in os.listdir(f'{test}/{i}'):
    test_image=cv2.imread(f'{test}/{i}/{img}')
    k=int(np.sqrt(len(labels)))
    test_hog=hog(test_image,orientations=11,pixels_per_cell=(15,15), cells_per_block=(2,2), channel_axis=-1)# calculating hog vector
    class_hog_fruit=k_nearest(training_image_hog, test_hog, k)
    test_rgb_array=to_flatten(np.array(test_image),size=(32,32))#calculating rgb vector
    class_rgb_fruit=k_nearest(training_image_rgb, test_rgb_array, k)
    true_class.append(i)
    true.append(i)
    #checking if the predicted class is equal to the true class
    if(class_hog_fruit==i):
      predicted_class.append(class_hog_fruit)
      pred.append(class_hog_fruit)
    elif(class_rgb_fruit==i):
      predicted_class.append(class_rgb_fruit)
      pred.append(class_rgb_fruit)
    else:
      predicted_class.append(class_rgb_fruit)
      pred.append(class_rgb_fruit)
  # printing accuracy of each fruit
  accuracy=accuracy_score(true, pred)
  print(i +" "+ str(accuracy))
print("Test cases done")

# final metrics(accuracy, precision, recall) of the whole dataset
accuracy=accuracy_score(true_class, predicted_class)
precision=precision_score(true_class, predicted_class, average='micro')
recall=recall_score(true_class, predicted_class, average='macro')
F1_score=2*(precision*recall)/(precision+recall)
print("Accuracy = ", accuracy)
print("Precision = ", precision)
print("Recall = ", recall)
print("F1 Score = ", F1_score)
