#!/usr/bin/env python
# coding: utf-8

# # Making Prototypes

# In[1]:


import os
import numpy as np
import pandas as pd
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog

path = "C:\\Users\\shwet\\Desktop\\archive\\fruits-360_dataset\\fruits-360\\Training"
all_Classes = os.listdir(path) 
prototypes = {}
prototypes_rgb = {}
cl_count = 0
for cl in all_Classes:
    proto = np.zeros((1100))
    count = 0
    red=0
    blue=0
    green=0
    for img in os.listdir(f'{path}\\{cl}'):
        fruit=imread(f'{path}\\{cl}\\{img}')
        vector = hog(fruit, orientations=11, pixels_per_cell = (15,15), cells_per_block = (2,2), multichannel=True)
        proto = proto+vector
        count+=1
        r=0
        g=0
        b=0
        ctr=0
        for x in range (0,100,1):
            for y in range(0,100,1):
                color = fruit[y,x]
                r=r+int(color[0])
                g=g+int(color[1])
                b=b+int(color[2])
                ctr=ctr+1
        red += r/ctr
        green += g/ctr
        blue += b/ctr
    prototypes_rgb[cl] = [(red/count),(green/count),(blue/count)]            
    proto = proto/count
    prototypes[cl] = proto
    cl_count+=1
    print("class: ",cl_count ,": ", cl," : done")


# # Defining Euclidean Distance function

# In[2]:


def Euclidean(v1,v2):
    import numpy as np
    # calculating Euclidean distance using linalg.norm()
    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.linalg.norm(v1 - v2)
    return (dist)


# In[3]:


prototypes_rgb


# # Loading testing data and predicting classes

# In[7]:


path2 = "C:\\Users\\shwet\\Desktop\\archive\\fruits-360_dataset\\fruits-360\\Test"
classes = os.listdir(path2) 
y_pred =[]    #list for classes predicted by the model
y_true =[]    #list for actual classes
for cla in classes:
    for img in os.listdir(f'{path2}\\{cla}'):
        fruit=imread(f'{path2}\\{cla}\\{img}')
        vector = hog(fruit, orientations=11, pixels_per_cell = (15,15), cells_per_block = (2,2), multichannel=True) #HOG vector for the test image
        list11=[]  # this list contains distances of the vector from all 131 prototypes
        list22=[]  # this list contains all 131 classes
        for cl in prototypes.keys():
            dis = Euclidean(prototypes[cl],vector)
            list11.append(dis)
            list22.append(cl)
        mini = min(list11)
        mini_ind = list11.index(mini)
        classObj = list22[mini_ind]    #variable storing the predicted class
        ############################# Testing from rgb prototypes of all classes ##########################3
        r=0
        g=0
        b=0
        ctr=0
        for x in range (0,100,1):
            for y in range(0,100,1):
                color = fruit[y,x]
                r=r+int(color[0])
                g=g+int(color[1])
                b=b+int(color[2])
                ctr=ctr+1
        r = r/ctr
        g = g/ctr
        b = b/ctr
        vector_rgb=[r,g,b]
        listRGB1 = []     # this list contains distances of the vector from all 131 prototypes
        listRGB2 = []     # this list contains all 131 classes
        for cl in prototypes_rgb.keys():
            distance = Euclidean(prototypes_rgb[cl],vector_rgb)
            listRGB1.append(distance)
            listRGB2.append(cl)
        minim = min(listRGB1)
        minim_ind = listRGB1.index(minim)
        classObjRGB = listRGB2[minim_ind]
        y_true.append(cla)
        if classObj == cla :
            y_pred.append(classObj)
        elif classObjRGB == cla:
            y_pred.append(classObjRGB)
        else:
            y_pred.append(classObjRGB)
    print("Class ",cla, "complete" )        


# # Reporting accuracy, precision, recall and F1 score of our model

# In[8]:


from sklearn.metrics import precision_score, recall_score, accuracy_score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='micro')
recall = recall_score(y_true, y_pred, average='macro')
F1 = 2 * (precision * recall) / (precision + recall)
print("Accuracy = ", accuracy*100 , "%")
print("Precision = ", precision*100 , "%")
print("Recall = ", recall)
print("F1 score = ", F1)


# In[ ]:




