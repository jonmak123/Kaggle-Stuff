# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 23:56:09 2018

@author: Jonathan Mak
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
import scipy.io.wavfile as sci_wav
from keras.utils import plot_model

## MAKE GENERATOR FOR TRAINING AND VALIDATION
def make_generator(dog, cat, window_size, batch_size = 20):
    """
    Returns a generator object which randomly extracts segments of 
    fixed window_size from train data for training
    """
    while True:
        X = []
        y = []
        for i in range(batch_size):
            dog_or_cat = random.choice([0, 1])
            data = dog if dog_or_cat == 1 else cat
            s = int(data.shape[1])
            idx_range = np.arange(s - window_size)
            start_ = random.choice(idx_range)
            end_ = start_ + window_size 
            X.append(data[:, start_: end_])
            y.append(dog_or_cat)
        X = np.stack(X).reshape((batch_size, 129, window_size, 1))
        y = np.array(y)
        yield((X, y))
        
        
## PLOT GRID SEARCH RESULTS
def plot_grid_search(results):
    """
    Plot loss after grid search
    """
    plt.figure(figsize = (50, 30))
    g = 1
    for lr_ in results.keys():
        for dropout_ in results[lr_].keys():
            dat = results[lr_][dropout_]
            ax = plt.subplot(3, 3, g)
            train_, = ax.plot(dat['loss'], label = 'train')
            valid_, = ax.plot(dat['val_loss'], label = 'valid')
            ax.set_title('lr = ' + str(lr_) + '; dropout = ' + str(dropout_))
            ax.set_ylim([0, 1])
            g += 1
    plt.tight_layout()


## PLOT MODEL SCHEMA
def plot_cnn(model):
    """
    Return model visualisation
    """
    plot_model(model)


## PAD SPECTROGRAM DATA WITH ZEROS WHEN IT IS TOO SHORT
def pad_s(s, window_size):
    """
    Pad with trailing zero array(s) if the sample is too short
    """
    if s.shape[1] < window_size:
        s = np.hstack((s, np.zeros((129, window_size - s.shape[1])) + -1))
    return(s)


## DIVIDE SPECTROGRAM DATA INTO CHUCKS OF SELECTED WINDOW SIZE OF EVAL
def make_slice(sample, window_size):
    """
    Divide the clip data into segments of same window_size for final testing
    """
    length = sample.shape[1]
    seg_n = int(length // window_size)
    idx = [x * window_size for x in range(seg_n)]
    if length % (seg_n * window_size) != 0:
        idx.append(length - window_size)
    return(idx)