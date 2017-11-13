
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
%matplotlib inline
from multiprocessing import Queue
# Load pickled data
import pickle
import cv2
import csv

print("Loading package is a sucess!")

samples  = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(csvfile)
    for line in reader:
        #print(line)
        samples.append(line)
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)        


print(len(train_samples))
samples_per_epoch = len(train_samples)/32
val_size = int(samples_per_epoch/10.0)

# samples contains all the lines in the csv file
def trainningdata_gen(samples, batch_size = 32):
    num_samples = len(samples)

    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []
            for line in batch_samples:
                # firstly store the center image
                source_path_center = line[0]
                filename_center = source_path_center.split('/')[-1]
                current_path_center = ('IMG/')+filename_center
                image_center = cv2.imread(current_path_center)
                image_center = cv2.cvtColor(image_center, cv2.COLOR_BGR2RGB)
                images.append(image_center)
                measurement = float(line[3])
                measurements.append(measurement)
                images.append(cv2.flip(image_center,1))
                measurements.append(measurement*(-1.0))
                                    
                # Add random britness
                image_center_t = cv2.cvtColor(image_center, cv2.COLOR_RGB2HSV)
                image_center_t[:,:,2] = image_center_t[:,:,2]*random.uniform(0.3,1.0)
                image_center_t = cv2.cvtColor(image_center_t, cv2.COLOR_HSV2RGB)
                images.append(image_center_t)
                measurements.append(measurement)
                                    
                """
            
                # Then store the left image
                source_path_left = line[1]
                filename_left = source_path_left.split('/')[-1]
                current_path_left = ('data/IMG/')+filename_left
                image_left = cv2.imread(current_path_left)
                image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
                images.append(image_left)
                measurements.append(measurement+0.2)
                images.append(cv2.flip(image_left,1))
                measurements.append((measurement+0.2)*-1.0)
                
                image_left_t = cv2.cvtColor(image_left, cv2.COLOR_RGB2HSV)
                image_left_t[:,:,2] = image_left_t[:,:,2]*random.uniform(0.3,1.0)
                image_left_t = cv2.cvtColor(image_left_t, cv2.COLOR_HSV2RGB)
                images.append(image_left_t)
                measurements.append(measurement)
                
                
            
                # Then store the right image
                source_path_right = line[2]
                filename_right = source_path_right.split('/')[-1]
                current_path_right = ('data/IMG/')+filename_right
                image_right = cv2.imread(current_path_right)
                image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)                    
                images.append(image_right)
                measurements.append(measurement-0.2)
                images.append(cv2.flip(image_right,1))
                measurements.append((measurement-0.2)*-1.0)
                
                
                image_right_t = cv2.cvtColor(image_right, cv2.COLOR_RGB2HSV)
                image_right_t[:,:,2] = image_right_t[:,:,2]*random.uniform(0.3,1.0)
                image_right_t = cv2.cvtColor(image_right_t, cv2.COLOR_HSV2RGB)
                images.append(image_right_t)
                measurements.append(measurement)
               """
            
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            X_train, y_train = shuffle(X_train, y_train)
            yield X_train,y_train
train_generator = trainningdata_gen(train_samples, batch_size=32)

validation_generator = trainningdata_gen(validation_samples, batch_size=32)
# Import the Keras package
from keras.models import Sequential
from keras.layers import Flatten,Dropout,Dense,ELU, Lambda, Activation
from keras.layers import SpatialDropout2D,Cropping2D,Convolution2D,MaxPooling2D
from sklearn.utils import shuffle
print("Loading keras is a sucess!")
model = Sequential()

# normolize the image around the center
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(160, 320, 3),
        output_shape=(160, 320, 3)))
# crop the image
model.add(Cropping2D(cropping=((70, 25), (0, 0)),
                     dim_ordering='tf', # default
                     input_shape=(160, 320, 3)))
# First layer of CNN
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation = "relu"))
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation = "relu"))
model.add(Convolution2D(48,5,5,subsample=(2, 2),activation = "relu"))
model.add(Convolution2D(64,3,3,activation = "relu"))
model.add(Convolution2D(128,3,3,activation = "relu"))


model.add(Flatten())

model.add(Dense(400))
model.add(Dropout(0.5))

model.add(Dense(200))
model.add(Dropout(0.5))

model.add(Dense(100))

model.add(Dense(75))

model.add(Dense(50))

model.add(Dense(25))

model.add(Dense(20))

model.add(Dense(1))


model.summary()

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="./tmp/comma-4c.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)

model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples)*4, validation_data=validation_generator,
            nb_val_samples=len(validation_samples)*4, nb_epoch=4)
model.save('model_center.h5')
print('Work is finished')
exit()