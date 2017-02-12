import pandas

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda

from keras.optimizers import Adam
from keras.models import load_model, model_from_json
import os
import json

#read the data file for center image and steering
driving_log = 'driving_log.csv'
header_names = ['center_image', 'left_image','right_image','steering','throttle','brake','speed']
driving_log_csv = pandas.read_csv(driving_log,names=header_names)

X_train = driving_log_csv['center_image']
Y_train = driving_log_csv['steering']

#split train and validation data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
X_train = X_train.as_matrix()
X_val   = X_val.as_matrix()
Y_train = Y_train.as_matrix()
Y_val   = Y_val.as_matrix()
Y_train = Y_train.astype(np.float32)
Y_val   = Y_val.astype(np.float32)



def crop_image(image):
    #crop the image
    image = image[60:136, 0:320,:]
    image = cv2.resize(image,(32,32),cv2.INTER_AREA)
    return image

def random_shift_crop(image,steering,shift_pixels_x=60,shift_pixels_y=10):
    # Randomly crop subsections of the image
    col_start,col_end =shift_pixels_x,320-shift_pixels_x
    tx= np.random.randint(-shift_pixels_x,shift_pixels_x+1)
    ty= np.random.randint(-shift_pixels_y,shift_pixels_y+1)
    #crop and resize to 64x64
    image = image[60+ty:136+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(image,(32,32),cv2.INTER_AREA)
    #adjust the steering to reflect the random chop
    if shift_pixels_x != 0:
        steering += -tx/90.0
   
    return image,steering

def random_brightness(image):   
    #50% chance to change brightness
    if np.random.randint(2) == 0: 
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)
        image[:,:,2] = image[:,:,2]*random_bright
        image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def random_flip(image,steering):
    #50% chance to flip image
    if np.random.randint(2) == 0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering
        

def get_random_image_steering(X_train,Y_train,show_figure=False):    
    i = np.random.randint(0,len(Y_train))
    image,steering =  plt.imread(X_train[i]), Y_train[i]   
    #image,steering = random_shear(image,steering,shear_range=100)
    image,steering = random_shift_crop(image,steering)   
    image,steering = random_flip(image,steering)    
    image = random_brightness(image)
    return image,steering

def training_sample_generator(X_train,Y_train,batch_size):
    images = np.zeros((batch_size, 32, 32, 3))
    steerings = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            image,steering = get_random_image_steering(X_train,Y_train)
            images[i],steerings[i] = image,steering

        yield images, steerings

def get_validation_set(X_val,Y_val):
    size = len(Y_val)
    images = np.zeros((size,32,32,3))
    steerings = np.zeros(size)
    for i in range(size):
        image,steering = random_flip(plt.imread(X_val[i]), Y_val[i])
        images[i],steerings[i] = crop_image(image),steering
    return images,steerings
    
def get_training_model():
    model = Sequential()
    #normalize the image
    model.add(Lambda(lambda x: (x / 127.5) - 1,input_shape=(32,32,3)))
    model.add(Convolution2D(32, 5,5 ,border_mode='same', subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3,3 ,border_mode='same',subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(1))
    model.summary()    
    return model

model_json = 'model.json'
model_weights = 'model.h5'

model = get_training_model()
restart=True
if os.path.isfile(model_json) and restart:
    with open(model_json) as json_file:
        model = model_from_json(json.load(json_file))
        model.load_weights(model_weights)    
        print('Loaded trained model ...')

#use Adam optimizer and Mean squared error loss function
model.compile(optimizer='adam', loss='mse')

generator = training_sample_generator(X_train,Y_train,400)
X_val, Y_val = get_validation_set(X_val,Y_val)
history = model.fit_generator(generator, samples_per_epoch=20000,
                              validation_data=(X_val,Y_val), nb_epoch=10)


model.save_weights(model_weights)
json_string = model.to_json()
with open(model_json, 'w') as json_file:
    json.dump(json_string, json_file)



