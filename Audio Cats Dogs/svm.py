# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:12:50 2018

@author: Jonathan Mak
"""

import numpy as np
from dataprocess import *
from model_utils import *
import pickle

from sklearn.svm import SVC


##### PARAMETERS #####
dataset = load_data(df)
concat_data = concatenate_data(dataset)
norms = calc_norm(concat_data)
norm_data = normalise_dataset(concat_data, norms)
spectro_data = {x: convert_spectro(y) for x, y in norm_data.items()}

window_size = 256

##### TRAIN SVM #####
train_gen = make_generator(spectro_data['train_dog'], spectro_data['train_cat'], window_size)
scores = []

xy_train = [(next(train_gen)) for x in range(400)]
x_train = np.vstack([j[0] for j in xy_train])
x_train = np.max(x_train, axis = 2)
x_train = np.squeeze(x_train, axis = 2)
y_train = np.concatenate([k[1] for k in xy_train])
svm = SVC()
svm = svm.fit(x_train, y_train)
print(svm.score(x_train, y_train))
pickle.dump(svm, open('svm.p', 'wb'))