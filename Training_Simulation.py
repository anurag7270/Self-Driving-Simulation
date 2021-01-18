###################################################################
########################## Anurag Tripathi ########################
###################################################################

# We will get error if we don't import files from utilities. So, let's import everything.
print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from utlis import *
from sklearn.model_selection import train_test_split


##################################### Step-1 is to import the data ##################################################

# We will create a function
path = 'myData'
data= importDataInfo(path)

# Now in utlis, we will import other libraries.

#################################### Step-2 is to Visualizating the data ############################################

""" 
The second very important part is basically visualization of your data. Now this is important because if
you have a lot of angles off for example the left curve and you have very little angle of the right curve
your model will journalize to go mostly on the left-hand side so we want to balance the data so that we
have equal amounts of information for each class. So in this case we we have a regression problem.
Regression problem---> we have a continuous value which can range from -1 to 1. So we can have infinite
number of values in between. Now the thing is that in order to visualize it better what we can do is
we can split it into bins and then we can plot a bar graph. So, we will go to utils and make a function for balancing.
"""

data = balanceData(data, display=False)

####################################### Step-3 Preapring for processing #############################################

imagesPath, steerings = loadData(path,data)
#print(imagesPath[0], steering[0])

################################## Step-4 Splitting of the data in Training & Validation###############################

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steerings, test_size=0.2,random_state=5)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

########################################## Step-5 Data Augmentation #################################################

# We will it during the training

########################################## Step-6 Pre-processing ###################################################




########################################## Step-7 Batch Normalization #################################################
""""
The idea here is that we do not send all of the images together to our training model what we do is we send the 
images in batches so this helps in generalization and it gives us more freedom to how we can create our images
and how we can send them to our model so what we can do is before actually sending it to the model we should augment
our image and we should also pre process our image so what we will do is we will create a function that will take
in the images path that we defined earlier so it will take in the images path and the steerings values so then we can 
for example define that we need a batch of hundred images and from these images path and steerings list it will
take out hundred random images and then it will augment it and pre process it and send it to our model """
########################################################################################################################



########################################## Step-8 Creating the Model #################################################

model = creatModel()
model.summary()

########################################## Step-9 Training the Model #################################################
history = model.fit(batchGen(xTrain,yTrain,100,1),steps_per_epoch=30,epochs=20,
                    validation_data=batchGen(xVal,yVal,100,0),validation_steps=200)

########################################## Step-10 Saving the Model #################################################
"""
Saving the weights and architecture of the model
"""
model.save('model.h5')
print('Model Saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training','Validation'])
#plt.ylim([0,1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()