# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:52 2018

MAIN CNN TRAIN SCRIPT

@author: jmak
"""

from dataprocess import *
from model_utils import *
import pickle
import numpy as np

np.random.seed(66)

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, Merge, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

##### PARAMS #####
window_size = 512

##### DATA LOAD AND PROCESS #####
dataset = load_data(df)
spectro_data = normalise_spectro(dataset)
spectro_data = concatenate_data(spectro_data)


##### MAKE CNN MODEL #####
def make_model(dropout = 0.25, lr = 0.0001):
    """
    Main CNN model body
    """
    batch_shape = (129, window_size, 1)
    
    x_input = Input(batch_shape)
        
    X = Conv2D(filters = 32, kernel_size = (10, 10), strides = (2, 2), padding = 'valid')(x_input)
    X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = 32, kernel_size = (5, 5), strides = (2, 2), padding = 'valid')(X)
    X = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid')(X)
    X = Activation('relu')(X)
    
    X = Flatten()(X)
    X = Dropout(dropout)(X)
    X = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = x_input, outputs = X)
    
    opt = Adam(lr = lr)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return(model)
    
 
def train_final_model(model, spectro_data, train_step = 100, epochs = 30):
    full_gen = make_generator(spectro_data['full_dog'], spectro_data['full_cat'], window_size)
    fit = model.fit_generator(generator = full_gen, 
                                steps_per_epoch = train_step,
                                epochs = epochs, 
                                verbose = True)

    return(model, fit)


def grid_search(learning_rate = [0.001, 0.0001, 0.00005], dropouts = [0.2, 0.35, 0.5]):
    """
    Loop through ranges of hyper parameters to tune model
    """
    grid_results = {}
    for lr in learning_rate:
        grid_results[lr] = {}
        for dropout in dropouts:
            print('training: dropout = ' + str(dropout) + ' lr = ' + str(lr))
            model = make_model(dropout, lr)
            train_gen = make_generator(spectro_data['train_dog'], spectro_data['train_cat'], window_size)
            valid_gen = make_generator(spectro_data['valid_dog'], spectro_data['valid_cat'], window_size)
            fit = model.fit_generator(generator = train_gen,
                    steps_per_epoch = 10,
                    validation_data = valid_gen, 
                    validation_steps = 25,
                    epochs = 50, 
                    verbose = True)
            grid_results[lr][dropout] = fit.history  
    
    pickle.dump(grid_results, open('fit/results.p', 'wb'))   
    plot_grid_search(grid_results)
    return(grid_results)

""" 
FOR GRID SEARCH
"""
#grid_restuls = grid_search()

"""
FOR FINAL MODEL
"""
model = make_model(dropout = 0.35, lr = 0.00005)
print(model.summary())
model, fit_hist = train_final_model(model, spectro_data, epochs = 10)
model.save('model.h5')