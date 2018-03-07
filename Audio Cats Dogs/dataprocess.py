# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:53:50 2018

@author: jmak
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.io.wavfile as sci_wav
from scipy import signal

ROOT_DIR = 'input/cats_dogs/'
CSV_PATH = 'input/train_test_split.csv'
df = pd.read_csv(CSV_PATH)

def read_wav(dir_):
    """
    read .wav files into array
    """
    return(sci_wav.read(dir_)[1])

def load_data(df):
    """
    Read loaded csv and output a dict with data
    """
    dataset = {}
    for k in ['train_cat', 'train_dog', 'valid_cat', 'valid_dog', 'test_cat', 'test_dog', 'full_cat', 'full_dog']:
        file_names = list(df[k].dropna())
        v = [read_wav(ROOT_DIR + f) for f in file_names]
        dataset[k] = v  
    return(dataset)

def convert_spectro(data, sample_rate = 16000):
    """
    Convert to fft data
    """
    f, t, s = signal.spectrogram(data, sample_rate)
    return(s)
    
def normalise_spectro(dataset):
    """
    Apply standard scaler to data for each clip as data preprocessing
    """
    normaliser = StandardScaler()
    spectro_data = {i : [normaliser.fit_transform(convert_spectro(j)) for j in dataset[i]] for i in dataset.keys()}
    return(spectro_data)

def concatenate_data(spectro_data):
    """
    Concatenate all train arrays etc into one for training
    """
    concat_data = {j : np.hstack(spectro_data[j]).astype('float32') for j in spectro_data.keys()}    
    return(concat_data)

    


