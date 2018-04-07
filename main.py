# -*- coding: utf-8 -*-
"""
Keras and CNN for MNIST Dataset classification

Refernce:
https://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/
@author: jkuo
"""

import numpy as np
np.random.seed(123)  # for reproducibility



from keras.models import Sequential # linear stack of NN layers
from keras.layers import Dense, Dropout, Activation, Flatten # core layers from Keras
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils #utils used to transform data
from keras import backend as K
K.set_image_dim_ordering('th') #using the theano ordering where the color channel comes first
#keras.backend.backed()# to check backend: tensorflow

from keras.datasets import mnist #mnist dataset

from matplotlib import pyplot as plt 

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print(X_train.shape) #(60000, 28, 28) 28x28 pixel images

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

mode=1 # 0 for MLP, 1 for CNN

## MLP approach
if mode == 0:
    # For a multi-layer perceptron model we must reduce the images down into a 
    # vector of pixels. In this case the 28×28 sized images will be 784 pixel input values.
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    
    # one hot encode outputs (integer (0~k-1) represents class)
    y_train = np_utils.to_categorical(y_train) # converts to a binary matrix where col are the class and row are the data
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # define baseline model, one hidden layer MLP
    def baseline_model():
    	# create model
    	model = Sequential()
    	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    	# Compile model
    	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    	return model
    
    # build the model
    model = baseline_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100)) #1.72

## CNN Approach
if mode == 1:
    def cnn_model():
    	# create model
    	model = Sequential()
    	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    	model.add(MaxPooling2D(pool_size=(2, 2)))
    	model.add(Dropout(0.2))
    	model.add(Flatten())
    	model.add(Dense(128, activation='relu'))
    	model.add(Dense(num_classes, activation='softmax'))
    	# Compile model
    	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    	return model
    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshape to be [samples][pixels][width][height] in theano ordering
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    
    # build the model
    model = cnn_model()
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100)) #1.19%