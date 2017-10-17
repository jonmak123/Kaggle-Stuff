# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 23:36:26 2017

@author: Jonathan Mak
"""

import numpy as np
import tensorflow as tf
import os
from PIL import Image
from skimage.transform import rotate, SimilarityTransform, warp, resize
from math import ceil

def _get_var(_name, weights, _type, _trainable):
    if _type == 'w':
        ind = 0
        name = 'filter_' + _name
    elif _type == 'b':
        ind = 1
        name = 'bias_' + _name
    init = tf.constant_initializer(value = weights[_name][ind], dtype = tf.float32)
    init_shape = weights[_name][ind].shape
    var = tf.get_variable(name = name, initializer = init, shape = init_shape, trainable = _trainable)
    return(var)

def _get_bilinear(shape, _name):
    width = shape[0]
    heigh = shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([shape[0], shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(shape)
    for i in range(shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    var = tf.get_variable(name = 'filter_' + _name, initializer = init, shape = weights.shape)
    return(var)

def xavier_init(_name, shape):
    xavier = tf.get_variable(name = _name, 
                             shape = shape, 
                             initializer = tf.contrib.layers.xavier_initializer())
    return(xavier)
    
def _conv_layer(_input, _name, weights, _trainable = False):
    with tf.variable_scope(_name):
        w = _get_var(_name, weights, 'w', _trainable)
        b = _get_var(_name, weights, 'b', _trainable)
        conv = tf.nn.conv2d(_input, w, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu(bias)
    return(relu)
    
def _fc_layer(_input, _name, shape, weights, relu = True):
    with tf.variable_scope(_name):
        w = weights[_name][0]
        w = tf.reshape(w, shape)
        b = weights[_name][1]
        conv = tf.nn.conv2d(_input, w, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, b)
        if relu:
            relu = tf.nn.relu(bias)
        else:
            relu = bias
    return(relu)
    
def _fc_layer_train(_input, _name, shape, relu = True):
    with tf.variable_scope(_name):
#        w = tf.Variable(tf.truncated_normal(shape), 0.1)
        w = xavier_init('filter_' + _name, shape)
        b = tf.Variable(tf.constant(0.1, shape = [shape[3]]))
        conv = tf.nn.conv2d(_input, w, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, b)
        if relu:
            relu = tf.nn.relu(bias)
        else:
            relu = bias
    return(relu)
    
def open_img(file_dir, resize_to = (224, 224)):
    img = Image.open(file_dir).convert('RGB')
    img = img.resize(resize_to, Image.ANTIALIAS)
    img = np.array(img)
    return(img)     

def conv_score(_input, num_class, _name):
    shape = _input.get_shape()
#    w = tf.Variable(tf.truncated_normal([1, 1, shape[3].value, num_class]), 0.1)
    w = xavier_init('skip_' + _name, [1, 1, shape[3].value, num_class])
    b = tf.Variable(tf.constant(0.1, shape = [num_class]))
    conv = tf.nn.conv2d(_input, w, [1, 1, 1, 1], padding = 'SAME')
    bias = tf.nn.bias_add(conv, b)
    return(bias)

def upsampling(_input, target, k_size, stride, num_class, _name):
    with tf.variable_scope(_name):
        shape = _input.get_shape()
        output = tf.shape(target)
#        w = tf.Variable(tf.truncated_normal([k_size, k_size, num_class, shape[3].value]), 0.1)
        w = _get_bilinear([k_size, k_size, num_class, shape[3].value], _name)
        upsample = tf.nn.conv2d_transpose(_input, w, tf.stack([output[0], output[1], output[2], num_class]), strides = [1, stride, stride, 1], padding='SAME')
    return(upsample)

def softmax_loss(logits, annotation):
    softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                                 labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                                 name="entropy"))
    return(softmax_loss)

def dice_loss(y_true, y_pred):
    smooth = 1
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    return(2 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice(logits, annotation):
    softmax = tf.nn.softmax(logits)[:, :, :, 1]
    y_ = tf.cast(annotation, tf.float32)
    loss = tf.add(softmax_loss(logits, annotation), (1 - dice_loss(y_, softmax)))
    return(loss)

def image_rotate(img, angle):
    img_trans = rotate(img, angle, preserve_range=True)
    img_trans = img_trans.astype('uint8')
    return(img_trans)

def image_trans_scale(img, translation, scale):
    tform = SimilarityTransform(scale = scale, translation = translation)
    img_trans = warp(img, tform, preserve_range = True)
    img_trans = img_trans.astype('uint8')
    return(img_trans)

class batchreader_floyd:
    
    def __init__(self, train_list, annotation_list, img_folder, img_size):
        print('Initialising floyd batchreader...')
        self.img_size = img_size
        self.current = 0
        self.epoch = 0
        self.list_len = len(train_list)
        self.rand = np.arange(self.list_len)
        np.random.shuffle(self.rand)
        self.train_list = [img_folder + 'train/' + train_list[k] for k in self.rand]
        self.annotation_list = [img_folder + 'train_masks/' + annotation_list[k] for k in self.rand]
        
    def next_batch(self, batch_size):
        start = self.current
        self.current += batch_size
        end = self.current
        if end > self.list_len:
            self.epoch += 1
            print('************Finished epoch:' + str(self.epoch))
            self.current = 0
            start = self.current
            self.current += batch_size
            end = self.current
            np.random.shuffle(self.rand)
            self.train_list = [self.train_list[k] for k in self.rand]
            self.annotation_list = [self.annotation_list[k] for k in self.rand]
            print('...', start)
        ## LOAD IMAGE
        train_imgs = [np.array(Image.open(j).convert('RGB').resize((self.img_size, self.img_size), Image.ANTIALIAS)) for j in self.train_list[start:end]]
        annotation_imgs = [np.array(Image.open(j).convert('L').resize((self.img_size, self.img_size), Image.ANTIALIAS)) for j in self.annotation_list[start:end]] 
        ## ROTATE IMAGE
        angle = np.random.randint(0, 10)
        train_imgs = [image_rotate(x, angle) for x in train_imgs]
        annotation_imgs = [image_rotate(x, angle) for x in annotation_imgs]
        ## SCALE AND TRANSLATE
        scale = np.random.randint(85, 100)/100
        trans_x = np.random.randint(-15, 15)
        trans_y = np.random.randint(-15, 15)
        train_imgs = [image_trans_scale(x, (trans_x, trans_y), scale) for x in train_imgs]
        annotation_imgs = [image_trans_scale(x, (trans_x, trans_y), scale) for x in annotation_imgs]        
        ## HORIZONTAL FLIP
        random = np.random.randint(0, 1)
        if random == 1:
            train_imgs = [x[:, ::-1] for x in train_imgs]
            annotation_imgs = [x[:, ::-1] for x in annotation_imgs]
        ## NORMALISE ANNOTATION 
        annotation_imgs = [x / 255.0 for x in annotation_imgs]
        ## EXPAND DIMS
        annotation_imgs = [np.expand_dims(i, axis = 2) for i in annotation_imgs]
        ## RESHAPE AND STACK
        img_array = np.stack(train_imgs, axis = 0)
        annotation_array = np.stack(annotation_imgs, axis = 0)
        return(img_array, annotation_array)