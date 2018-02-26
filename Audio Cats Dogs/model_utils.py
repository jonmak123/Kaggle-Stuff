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


## MAKE GENERATOR FOR TRAINING AND VALIDATION
def make_generator(dog, cat, window_size, batch_size = 20):
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
    plt.figure(figsize = (30, 30))
    g = 1
    for lr_ in results.keys():
        for dropout_ in results[lr_].keys():
            dat = results[lr_][dropout_]
            ax = plt.subplot(5, 5, g)
            train_, = ax.plot(dat['acc'], label = 'train')
            valid_, = ax.plot(dat['val_acc'], label = 'valid')
            ax.set_title('lr = ' + str(lr_) + '; dropout = ' + str(dropout_))
            ax.set_ylim([0, 1])
            g += 1
    plt.tight_layout()
    

## PAD SPECTROGRAM DATA WITH ZEROS WHEN IT IS TOO SHORT
def pad_s(s, window_size):
    if s.shape[1] < window_size:
        s = np.hstack((s, np.zeros((129, window_size - s.shape[1])) + -1))
    return(s)


## DIVIDE SPECTROGRAM DATA INTO CHUCKS OF SELECTED WINDOW SIZE OF EVAL
def make_slice(sample, window_size):
    length = sample.shape[1]
    seg_n = int(length // window_size)
    idx = [x * window_size for x in range(seg_n)]
    if length % (seg_n * window_size) != 0:
        idx.append(length - window_size)
    return(idx)