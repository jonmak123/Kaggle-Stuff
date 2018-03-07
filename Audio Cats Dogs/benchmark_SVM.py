# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:12:50 2018

Train SVM Benchmark model

@author: Jonathan Mak
"""

import numpy as np
from dataprocess import *
from model_utils import *
import pickle
import warnings
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=DeprecationWarning)

##### MAKE TRAIN DATA #####
dataset = load_data(df)

spectro_data = normalise_spectro(dataset)
spectro_data = {i : [np.max(j, axis = 1) for j in spectro_data[i]] for i in spectro_data.keys()}

x_train = spectro_data['full_dog'] + spectro_data['full_cat']        
y_train = [1] * len(spectro_data['full_dog']) + [0] * len(spectro_data['full_cat'])
x_test = spectro_data['test_dog'] + spectro_data['test_cat']        
y_test = [1] * len(spectro_data['test_dog']) + [0] * len(spectro_data['test_cat'])

##### PLOT DISTRIBUTION #####
#DOG = np.mean(np.array(x_train)[np.array(y_train) == 1], axis = 0)
#CAT = np.mean(np.array(x_train)[np.array(y_train) == 0], axis = 0)
#plt.hist(DOG, alpha = 0.5, bins = 30)
#plt.hist(CAT, alpha = 0.5, bins = 30)

##### TRAIN AND EVAL
svm = SVC()
svm = svm.fit(x_train, y_train)
pred = svm.predict(x_test)
score = svm.score(x_test, y_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
