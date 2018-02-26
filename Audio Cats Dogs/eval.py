# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:32:13 2018

@author: jmak
"""

import numpy as np
from dataprocess import *
from model_utils import *
import pickle

from keras.models import load_model
##### PARAMS #####
window_size = 256
norms = (1.1438465118408203, 4144.41064453125)

##### DATA LOAD AND PROCESS #####
dataset = load_data(df)
dogs = dataset['test_dog']
cats = dataset['test_cat']

x = dogs + cats
x = [(s - norms[0]) / norms[1] for s in x]
x = [convert_spectro(s) for s in x]
x = [pad_s(s, window_size) for s in x]
x = [[s[:, idx:idx+window_size] for idx in make_slice(s, window_size)] for s in x]
x_cnn = [np.expand_dims(np.array(s), 3) for s in x]
x_svm = [np.max(np.array(s), axis = 2) for s in x]

y = [1] * len(dogs) + [0] * len(cats)
y = np.array(y)

def avg_pred(model, x, y):
    pred_log = [np.mean(model.predict(s)) for s in x]
    pred = np.array(pred_log) > 0.5
    return(pred)

cnn = load_model('model.h5')
svm = pickle.load(open('svm.p', 'rb'))

cnn_pred = avg_pred(cnn, x_cnn, y)
print('cnn: ', str(np.mean(cnn_pred * 1 == y)))

svm_pred = avg_pred(svm, x_svm, y)
print('svm: ', str(np.mean(svm_pred * 1 == y)))
