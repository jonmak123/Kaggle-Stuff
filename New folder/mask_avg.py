# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:44:05 2017

@author: jmak
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image


#meta = pd.read_csv('input/metadata.csv')
#uniq_cars = list(set([x.split('_')[0] for x in mask_dir]))


##### AUX FUNC TO OPEN IMG TO ARRAY #####
def open_img(file_dir, folder = './train_masks/'):
    img = Image.open(folder + file_dir).convert('L')
    img = np.array(img, dtype = 'uint8')
    return(img)
    

##### READ ALL IMG AND CONCAT TO ARRAY THEN TRANSPOSE #####
def read_all_img(mask_dir):    
    all_img = [open_img(x) for x in mask_dir]
    all_img = np.concatenate(all_img)
    all_img = np.reshape(all_img, (318, 16, 1280, 1918))
    all_img = all_img.transpose((1, 0, 2, 3))
    return(all_img)


##### AVERAGE MASKS OVER VIEWS AND SAVE JPEG #####
def avg_view(all_img):
    avg_array = np.mean(all_img, axis = 1)
    for i in range(avg_array.shape[0]):
        plt.imsave('./avg_img/view_' + str(i + 1) + '.jpg', 
                   avg_array[i], 
                   cmap = 'gray', 
                   format = 'jpg')
    return(avg_array)


##### USE AVERAGED MASK ON A TRAIN IMAGE #####
def use_mask(avg_array, prob, view_index = 0, example_index = 0):
    mask = avg_array[view_index, :, :]
    mask = mask / mask.max()
    
    mask = (mask > prob) * 1 
    train_img = open_img(train_dir[0], 'input/train/')
    
    masked_img = train_img * mask
    plt.imshow(masked_img, cmap = 'gray')
    

##### EXAMPLE SCRIPT RUN #####
if __name__ == '__main__':
    mask_dir = os.listdir('input/train_masks/')
    train_dir = os.listdir('input/train/')
#    all_img = read_all_img(mask_dir)
#    avg_array = avg_view(all_img)
#    use_mask(avg_array, 0.4)
