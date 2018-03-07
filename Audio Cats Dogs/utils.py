import numpy as np  # linear algebra
import pandas as pd  # CSV file
import scipy.io.wavfile as sci_wav  # Open wav files
import matplotlib.pyplot as plt
import numpy as np
import random


ROOT_DIR = 'input/cats_dogs/'
CSV_PATH = 'input/train_test_split.csv'


def read_wav_files(wav_files):
    '''Returns a list of audio waves
    Params:
        wav_files: List of .wav paths

    Returns:
        List of audio signals
    '''
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]


def load_dataset(dataframe):
    '''Load the dataset in a dictionary.
    From the dataframe, it reads the [train_cat, train_dog, test_cat, test_dog]
    columns and loads their corresponding arrays into the <dataset> dictionary

    Params:
        dataframe: a pandas dataframe with 4 columns [train_cat, train_dog, 
        test_cat, test_dog]. In each columns, many WAV names (eg. ['cat_1.wav',
        'cat_2.wav']) which are going to be read and append into a list

    Return:
        dataset = {
            'train_cat': [[0,2,3,6,1,4,8,...],[2,5,4,6,8,7,4,5,...],...]
            'train_dog': [[sound 1],[sound 2],...]
            'test_cat': [[sound 1],[sound 2],...]
            'test_dog': [[sound 1],[sound 2],...]
        }
    '''
    df = dataframe

    dataset = {}
    for k in ['train_cat', 'train_dog', 'valid_cat', 'valid_dog', 'test_cat', 'test_dog', 'full_cat', 'full_dog']:
        v = list(df[k].dropna())
        v = read_wav_files(v)
        v = np.concatenate(v).astype('float32')

        # Compute mean and variance
        if k == 'train_cat':
            dog_std = dog_mean = 0
            cat_std, cat_mean = v.std(), v.mean()
        elif k == 'train_dog':
            dog_std, dog_mean = v.std(), v.mean()

        # Mean and variance suppression
        std, mean = (cat_std, cat_mean) if 'cat' in k else (dog_std, dog_mean)
        v = (v - mean) / std
        dataset[k] = v

        print('loaded {} with {} sec of audio'.format(k, len(v) / 16000))

    return dataset