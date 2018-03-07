# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:32:13 2018

@author: jmak
"""

import numpy as np
from dataprocess import *
from model_utils import *
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model
import os 

def avg_pred(model, x, y):
    """
    Calcuate mean probability of segement predictions and 
    apply threshold to output final prediction
    """
    pred_log = [np.mean(model.predict(s)) for s in x]
    pred = np.array(pred_log) > 0.5
    return(pred)

def eval_cnn(model, df = df):
    """
    Load data and convert with FFT as usual,
    concatenate test dog and test cat data, 
    pad with zeros for clips that are too short (<512),
    extract segments of fixed window_size for each clip, 
    feed into trained CNN model and output performance metrics
    """
    ##### PARAMS #####
    window_size = 512
    
    ##### DATA LOAD AND PROCESS #####
    dataset = load_data(df)
    spectro_data = normalise_spectro(dataset)
    
    dogs = spectro_data['test_dog']
    cats = spectro_data['test_cat']
    
    x = dogs + cats
    x = [pad_s(s, window_size) for s in x]
    x = [[s[:, idx:idx+window_size] for idx in make_slice(s, window_size)] for s in x]
    x_cnn = [np.expand_dims(np.array(s), 3) for s in x]
    
    y = [1] * len(dogs) + [0] * len(cats)
    y = np.array(y)
    
    cnn_pred = avg_pred(cnn, x_cnn, y)
    score = np.mean(cnn_pred * 1 == y)
#    print('cnn: ', str(np.mean(cnn_pred * 1 == y)))
    print(confusion_matrix(y, cnn_pred * 1))
    print(classification_report(y, cnn_pred * 1))
    
    return(score, cnn_pred)

def eval_ext(model, dir_):
    ##### PARAMS #####
    window_size = 512
    
    sampling_rate, test = sci_wav.read(dir_)
    if test.ndim == 2:
        test = (test[:, 0] + test[:, 1])/2
#        test = test[:, 0]
    test1 = convert_spectro(test, sampling_rate)
    
    normaliser = StandardScaler()
    test2 = normaliser.fit_transform(test1)
    
    x = pad_s(test2, window_size)
    x = [x[:, idx:idx+window_size] for idx in make_slice(x, window_size)]
    x_cnn = np.expand_dims(np.array(x), 3)
    
    pred_log = model.predict(x_cnn)
    
    return(pred_log)
    
cnn = load_model('model.h5')

""" TEST ON TEST SET """
#score, cnn_pred = eval_cnn(cnn)

""" TEST ON EXTERNAL """
dirs_ = os.listdir('external_test')
y_ = [1 if 'dog' in d else 0 for d in dirs_]
pred = [eval_ext(cnn, 'external_test/' + d) for d in dirs_]
pred_l = [(np.mean(x) > 0.5) * 1 for x in pred]
np.mean(pred_l == np.array(y_))
