# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:53:50 2018

@author: jmak
"""
import numpy as np
import pandas as pd

import scipy.io.wavfile as sci_wav
from scipy import signal

ROOT_DIR = 'input/cats_dogs/'
CSV_PATH = 'input/train_test_split.csv'
df = pd.read_csv(CSV_PATH)

def read_wav(dir_):
    return(sci_wav.read(dir_)[1])

def load_data(df):
    dataset = {}
    for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        file_names = list(df[k].dropna())
        v = [read_wav(ROOT_DIR + f) for f in file_names]
        dataset[k] = v
        
    return(dataset)

def concatenate_data(dataset):
    concat_data = {j : np.concatenate(dataset[j]).astype('float32') for j in dataset.keys()}
    
    return(concat_data)

def calc_norm(concat_data):
    dog_mean = np.mean(concat_data['train_dog'])
    dog_std = np.std(concat_data['train_dog'])
    cat_mean = np.mean(concat_data['train_cat'])
    cat_std = np.std(concat_data['train_cat'])
    
    mean = (dog_mean + cat_mean) / 2
    std = (dog_std + cat_std) / 2
    
#    return((dog_mean, dog_std, cat_mean, cat_std))
    return((mean, std))
    
def normalise_dataset(concat_data, norms):
#    dog_mean, dog_std, cat_mean, cat_std = norms
    mean, std = norms
    norm_data = {}
    for k in concat_data.keys():
        norm_data[k] = (concat_data[k] - mean) / std
#        if 'dog' in k:
#            norm_data[k] = (concat_data[k] - dog_mean) / dog_std
#        if 'cat' in k:
#            norm_data[k] = (concat_data[k] - cat_mean) / cat_std

    return(norm_data)    
        
def convert_spectro(data, sample_rate = 16000):
    f, t, s = signal.spectrogram(data, sample_rate)
    return(s)


