# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:51:52 2018

@author: jmak
"""

from dataprocess import *
from model_utils import *
import pickle

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPooling2D, Dropout, Merge, Concatenate
from keras.models import Model
from keras.optimizers import Adam

##### PARAMS #####
window_size = 256

##### DATA LOAD AND PROCESS #####
dataset = load_data(df)
concat_data = concatenate_data(dataset)
norms = calc_norm(concat_data)
norm_data = normalise_dataset(concat_data, norms)
spectro_data = {x: convert_spectro(y) for x, y in norm_data.items()}

##### MAKE CNN MODEL #####
def make_model(dropout = 0.25, lr = 0.000075):
    batch_shape = (129, window_size, 1)
    
    x_input = Input(batch_shape)
    
    X = Conv2D(filters = 8, kernel_size = (129, 8), strides = (129, 4), padding = 'same')(x_input)
    X = MaxPooling2D(pool_size = (1, 3), strides = (1, 2), padding = 'same')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(filters = 10, kernel_size = (129, 8), strides = (129, 4), padding = 'same')(X)
    X = MaxPooling2D(pool_size = (1, 3), strides = (1, 2), padding = 'same')(X)
    X = Activation('relu')(X)
    
    d1 = BatchNormalization()(X)
    d1 = Flatten()(d1)
    d1 = Dropout(dropout)(d1)
    d1 = Dense(16, activation = 'relu')(d1)
    
    X = Conv2D(filters = 12, kernel_size = (129, 8), strides = (129, 4), padding = 'same')(X)
    X = MaxPooling2D(pool_size = (1, 3), strides = (1, 2), padding = 'same')(X)
    X = Activation('relu')(X)
    
    d2 = BatchNormalization()(X)
    d2 = Flatten()(d2)
    d2 = Dropout(dropout)(d2)
    d2 = Dense(16, activation = 'relu')(d2)
    
    X = Conv2D(filters = 14, kernel_size = (129, 8), strides = (129, 4), padding = 'same')(X)
    X = MaxPooling2D(pool_size = (1, 3), strides = (1, 2), padding = 'same')(X)
    X = Activation('relu')(X)
    
    d3 = BatchNormalization()(X)
    d3 = Flatten()(d3)
    d3 = Dropout(dropout)(d3)
    d3 = Dense(16, activation = 'relu')(d3)
    
    X = Concatenate()([d1, d2, d3])
    X = Dropout(dropout)(X)
    X = Dense(1, activation = 'sigmoid')(X)
    
    model = Model(inputs = x_input, outputs = X)
    
    opt = Adam(lr = lr)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

    return(model)
    
def train_model(model, spectro_data, train_step = 50, val_step = 50, epochs = 50, verbose = False):
    train_gen = make_generator(spectro_data['train_dog'], spectro_data['train_cat'], window_size)
    valid_gen = make_generator(spectro_data['test_dog'], spectro_data['test_cat'], window_size)
    
    fit = model.fit_generator(generator = train_gen,
                        steps_per_epoch = train_step,
                        validation_data = valid_gen, 
                        validation_steps = val_step,
                        epochs = epochs, 
                        verbose = verbose)
    
    return(model, fit)

def grid_search(learning_rate = [0.001, 0.0005, 0.0001, 0.000075, 0.00005], dropouts = [0.125, 0.25, 0.375, 0.5]):
    grid_results = {}
    for lr in learning_rate:
        grid_results[lr] = {}
        for dropout in dropouts:
            print('training: dropout = ' + str(dropout) + ' lr = ' + str(lr))
            model = make_model(dropout, lr)
            model, fit = train_model(model, spectro_data)
            grid_results[lr][dropout] = fit.history  
    
    pickle.dump(grid_results, open('fit/results.p', 'wb'))   
    plot_grid_search(grid_results)
    return(grid_results)
    
model = make_model()
#grid_restuls = grid_search([0.001, 0.0001, 0.00005], [0.25, 0.5])
model, fit_hist = train_model(model, spectro_data, verbose = True)
model.save('model.h5')
