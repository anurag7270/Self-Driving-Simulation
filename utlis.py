# We will be saving all our functions here in utlis
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg    # Reason to use matplot is that it gives an RGB image
from imgaug import augmenters as iaa
import cv2
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Flatten,Dense
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
############################################## Importing Data ##########################################################

def importDataInfo(path):
    coloums = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=coloums)
    # print(data.head())
    """
    # The path that we have given here is just referring to the data. After the data, we have the name of the
    # file. So, we have to join the name with myData folder. So, we are going to use OS for that.
    # Then, we will say that "os.path.join" and we want to join the path and we also want to add driving log.
    # Then we are going to say that name of our columns are basically coloums. Then we can print this out
    # So, it will show us the initial information.
    # So, after loading the csv file; if we were to only get the first coloum and the "center" information
    """
    # print(data['Center'][0])
    # print(getName(data['Center'][0]))
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())      # Now we have filename with the extension.
    print('Total Images Imported:', data.shape[0])  # Now we can print out the complete no. of images that we have.
    return data

############################################ Getting File Path ########################################################

# So, the the output---> (C:\Users\DELL\Desktop\self_project\IMG\center_2021_01_09_01_23_10_668.jpg) that we will get
# when we run above: print(data['center'][0]), in order to import the output we only need the name at the end
# with .jpg. So, we will remove all of these folders from our path. So, we can create a function that can split
# output based on this slash \ and then we can get the last element of it.



def getName(filePath):
    return filePath.split('\\')[-1]
    # We only need the last part: [-1]


# So, this is the actual path that we have--->(C:\Users\DELL\Desktop\self_project\IMG\center_2021_01_09_01_23_10_668.jpg)
# And after the split, we are only getting the name of our file--->(center_2021_01_09_01_23_10_668.jpg) which is our image.
# Now, we need to apply this to all out images in our data file.

############################################ Balancing Data ###########################################################
def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 2000  # Cut off value
    hist, bins = np.histogram(data['Steering'], nBins)
    # print(bins)
    """ When we will print bins, we will get output from -1 to 1; but there will not be any 0
    because we are expecting to drive straight most of the time so we should have a bin
    for zero as well. To fix this what we can do is we can create a centre.
    """
    if display:
        center = (bins[:-1] + bins[
                              1:]) * 0.5  # center-->we will create two matrices and we will do element wise addition
        # print(center)
        plt.bar(center, hist, width=0.05)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()
        """
        Here we are going to write our function so we are going to say balance data and then we want the
        data as input and what we will do we will create a flag that will allow us to turn the display
        on and off because once we are done with this stage we don't want to see the graph again and again. Now the
        first thing we want to define is our number of pins so we will put it as 31. Now this has to be an odd number
        because we want the 0 to be at the center and then we have positive side and the negative side
        then we have samples per bin. Then we have to find the histogram values so how much data we have of each class
        so to do that we are going to use numpy dot histogram.
        """

    # Removing the redundant data, we will need is the list of the data we want to remove so or remove index list
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]  # Our samplePerBin will be the maximum number that we will obtain.
        removeIndexList.extend(binDataList)
    print('Removed Images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images: ', len(data))
    """
    So, what's this doing is that it's going to each of these bins and then it's checking 
    its value so it is checking that if let's say this is 0.25 and the next one is let's say 0.1,
    It will check all the values that are in between this and it will append the index of these
    values in our bin data list. So we will get all the values that range in that certain bin appended 
    in our bin data list so then we will have all the values but the problem is that they will
    be in order and we don't want to keep all the values in order. We want to shuffle them 
    before we remove. If we remove within the order then it will remove all the small values and then
    it will keep all the big values. So we don't want that; so will shuffle it. 
    """
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.05)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()
        """our data on the right hand side should be almost equal to the data on the left;
         if not; then recollect the data """
    return data
##################################################### Loading Data ####################################################
def loadData(path,data):
    imagesPath = []
    steering = []
    """ We need two list we are not importing the images at this time we are just preparing the
    paths so that whenever we want to import we can just refer to this list or an array and
    then we can simply import and then we will create steering list
    """
    for i in range (len(data)):
        indexedData = data.iloc[i]
        #print(indexedData)
        imagesPath.append(os.path.join(path,'IMG',indexedData[0]))
        #print(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering
"""
When we will run this, we will see the very first elements of our entry is our Center so if we want
to access Center we will just refer to the element zero and you can see that our staining
is element number 3 so 0,1,2,3. So we will access that using 3. so here we are going to write
that imagesPath.append() so as you can see here we have the center value now what we want
is we don't want just the name of this we cannot simply import with just the name we want the name of
the folder it's in and the the parent folder as well so the parent folder is my data and then we have
image and then we have this image name so we have to add these as well so what we can do is we can 
write here that OS dot dot dot join and then we can write our path that's why we needed this path which 
is basically my data and then we want to add to that image so we will write here IMG and then we 
also need the image name. so once we are done with the for-loop what we can do is we can change these
lists to numpy arrays. so we can say that images path is equals to numpy dot as array."""

############################################# Data Augmentation ########################################################
"""
   The reason we need steering is because we will also use flip so if we have an image which
   had left curve and if we flip it horizontally the curve will become right so we will have 
   to change the sign for it so that's why we are getting the staining angle as well.
"""
def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)
    #print(np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand(),)
    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    # ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.1,1.5))
        img = brightness.augment_image(img)

    # FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = -steering

    return img, steering

# imgRe, st = augmentImage('test.jpg',0)
# plt.imshow(imgRe)
# plt.show()

############################################ Pre-processing od Data ####################################################
def preProcessing(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img

# imgRe = preProcessing(mpimg.imread('test.jpg'))
# plt.imshow(imgRe)
# plt.show()

################################################### Batch Generalization ##############################################

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0,len(imagesPath)-1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index],steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch),np.asarray(steeringBatch))

"""
we are going to convert  imgBatch=[] and steeringBatch=[] so these
two again they are list we are going to convert it into number (numpy arrays)
"""
################################################### Creating Model#####################################################

def creatModel():
    model = Sequential()

    model.add(Convolution2D(24, (5, 5), (2, 2),input_shape = (66,200,3),activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.001),loss='mse')
    return model