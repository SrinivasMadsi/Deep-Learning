# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:13:01 2019

@author: srinivas.madsi
"""
# Convolutional Neural Network
# Image classification of Cat and Dog
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense # Here Dense used for fully connected layer

# Initializing the CNN
classifier = Sequential()

# Convolution->Max Pooling->Flattening->Full connection

# Step1 - Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu')) # 32 feature detectors and 3*3 Matrix and input_shape is colour images here

# Step2 -Pooling {It is used to reduce size of the Feature map to reduce no.of nodes}
classifier.add(MaxPooling2D(pool_size = (2,2)))

## Adding a  second convolution layer to tune the model
classifier.add(Convolution2D(32,3,3,activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step3 - Flattening { All the future maps to put in a Single Vector}
classifier.add(Flatten())

#Step4 - Full Connection
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

# Compiling CNN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

# Fitting the CNN to the Images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
test_set  = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
'''
# The below code will take time, so reducing values and epochs to test the output
classifier.fit_generator(training_set,
                                steps_per_epoch=8000,
                                epochs=25,
                                validation_data=test_set,
                                validation_steps=2000)
'''

classifier.fit_generator(training_set,
                                steps_per_epoch=80,
                                epochs=3,
                                validation_data=test_set,
                                validation_steps=500)

# Making new predictions with single image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size = (64,64))
test_image = image.img_to_array(test_image) # Now dimensions will be 64,64,3
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image) # Here result will be binary
training_set.class_indices # To understand the binary outcome we should execute this

