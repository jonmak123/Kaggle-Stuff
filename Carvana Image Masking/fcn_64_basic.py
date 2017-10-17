# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:44:38 2017

@author: jmak
"""

import tensorflow as tf
import numpy as np
import os 
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


##### READ AND SHUFFLE #####
mask_dir = os.listdir('input/train_masks/')
train_dir = os.listdir('input/train/')
rand = np.arange(len(train_dir))
np.random.shuffle(rand)
mask_dir = [mask_dir[j] for j in rand]
train_dir = [train_dir[j] for j in rand]


##### READ IMAGE #####
def open_img(file_dir, type_, mode, resize_to = (192, 108)):
    if type_ == 'train':
        folder = 'input/train/'
    if type_ == 'mask':
        folder = 'input/train_masks/'
    img = Image.open(folder + file_dir).convert(mode)
    img = img.resize(resize_to, Image.ANTIALIAS)
    img = np.array(img)
    return(img)

ex_train = open_img(train_dir[0], 'train', 'L')
#plt.imshow(ex_train, cmap = 'gray')
#ex_mask = open_img(mask_dir[0], 'mask', 'L')
#plt.imshow(ex_mask, cmap = 'gray')


##### RESET GRAPH TO CLEAR PREVIOUS RUNS #####
tf.reset_default_graph()


##### GLOBAL PARAMETERS #####
batch_size = 1
n_height = ex_train.shape[0]
n_width = ex_train.shape[1]
n_chnl = 1


##### DECLARE PLACEHOLDERS #####
x = tf.placeholder(tf.float32, shape = [None, n_height, n_width, n_chnl])
y = tf.placeholder(tf.float32, shape = [None, n_height, n_width])


##### SUMMARY OPS WRAPPER FUNC FOR TENSORBOARD ##### 
def variable_summaries(var):
    ## TENSORBOARD SUMMARY ONLY WORKS WITH CPU IT SEEMS
    with tf.device('/cpu:0'):
        ## THESE ARE FOR WEIGHTS
        with tf.name_scope('summaries'):
            tf.summary.scalar('mean', tf.reduce_mean(var))
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


##### MAIN MODEL BODY #####
def CNN_model(x):
    
    with tf.device('/cpu:0'):
    
        with tf.name_scope('Conv_L1'):
            
            ## CONV LAYER 1
            conv1_f_dim = 5
            conv1_chnl = 6
            conv1_stride = 1
            f_conv1 = tf.Variable(tf.truncated_normal([conv1_f_dim, conv1_f_dim, n_chnl, conv1_chnl], 0.1), name = 'conv1_filter')
            b_conv1 = tf.Variable(tf.constant(0.1, shape = [conv1_chnl]), name = 'conv1_bias')
            conv1 = tf.nn.conv2d(x, f_conv1, [1, conv1_stride, conv1_stride, 1], 'SAME', name = 'conv1_main')
            conv1 = tf.nn.relu(tf.add(conv1, b_conv1), name = 'conv1_actn')
#            variable_summaries(f_conv1)
            
            ## MAXPOOL 1
            mp1_dim = 2
            mp1_stride = 2
            pool1 = tf.nn.max_pool(conv1, [1, mp1_dim, mp1_dim, 1], [1, mp1_stride, mp1_stride, 1], 'SAME', name = 'max_pool_1')
    
        with tf.name_scope('Conv_L2'):    
            
            ## CONV LAYER 2
            conv2_f_dim = 5
            conv2_chnl = 12
            conv2_stride = 1
            f_conv2 = tf.Variable(tf.truncated_normal([conv2_f_dim, conv2_f_dim, conv1_chnl, conv2_chnl], 0.1), name = 'conv2_filter')
            b_conv2 = tf.Variable(tf.constant(0.1, shape = [conv2_chnl]), name = 'conv2_bias')
            conv2 = tf.nn.conv2d(pool1, f_conv2, [1, conv2_stride, conv2_stride, 1], 'SAME', name = 'conv2_main')
            conv2 = tf.nn.relu(tf.add(conv2, b_conv2), name = 'conv2_actn')
            variable_summaries(f_conv2)
            
            ## MAXPOOL 1
            mp2_dim = 2
            mp2_stride = 2
            pool2 = tf.nn.max_pool(conv2, [1, mp2_dim, mp2_dim, 1], [1, mp2_stride, mp2_stride, 1], 'SAME', name = 'max_pool_2')
        
        with tf.name_scope('Conv_L3'):
            
            ## CONV LAYER 3
            conv3_f_dim = 5
            conv3_chnl = 24
            conv3_stride = 1
            f_conv3 = tf.Variable(tf.truncated_normal([conv3_f_dim, conv3_f_dim, conv2_chnl, conv3_chnl], 0.1), name = 'conv3_filter')
            b_conv3 = tf.Variable(tf.constant(0.1, shape = [conv3_chnl]), name = 'conv3_bias')
            conv3 = tf.nn.conv2d(pool2, f_conv3, [1, conv3_stride, conv3_stride, 1], 'SAME', name = 'conv3_main')
            conv3 = tf.nn.relu(tf.add(conv3, b_conv3), name = 'conv3_actn')
            variable_summaries(f_conv3)
            
            # MAXPOOL 3
            mp3_dim = 2
            mp3_stride = 2 
            pool3 = tf.nn.max_pool(conv3, [1, mp3_dim, mp3_dim, 1], [1, mp3_stride, mp3_stride, 1], 'SAME', name = 'max_pool_3')
            pool2_shape = pool3.get_shape()
            pool3 = tf.reshape(pool3, [-1, int(pool2_shape[1]) * int(pool2_shape[2]) * int(pool2_shape[3])])

        with tf.name_scope('FC1'):
             
            ## FULLY CONNECTED LAYER 1
            full_out = n_height * n_width
            w_fc1 = tf.Variable(tf.truncated_normal([int(pool2_shape[1]) * int(pool2_shape[2]) * int(pool2_shape[3]), full_out], 0.1), name = 'full_weight')
            b_fc1 = tf.Variable(tf.constant(0.1, shape = [full_out]), name = 'full_bias')
            fc1 = tf.matmul(pool3, w_fc1, name = 'fully_connected_main')
            fc1 = tf.nn.relu(tf.add(fc1, b_fc1), name = 'fully_connected_actn')
            fc1 = tf.reshape(fc1, [-1, n_height, n_width])
            variable_summaries(w_fc1)
        
#        with tf.name_scope('FC2'):
#             
#            ## FULLY CONNECTED LAYER 2
#            c2_shape = pool3.get_shape()
#            full_out = n_height * n_width
#            full_pre = tf.reshape(pool3, [-1, int(c2_shape[1]) * int(c2_shape[2]) * int(c2_shape[3])])
#            w_full = tf.Variable(tf.truncated_normal([int(c2_shape[1]) * int(c2_shape[2]) * int(c2_shape[3]), full_out], 0.1), name = 'full_weight')
#            b_full = tf.Variable(tf.constant(0.1, shape = [full_out]), name = 'full_bias')
#            fullc = tf.matmul(full_pre, w_full, name = 'fully_connected_main')
#            fullc = tf.nn.relu(tf.add(fullc, b_full), name = 'fully_connected_actn')
#            fullc = tf.reshape(fullc, [-1, n_height, n_width])
#            variable_summaries(w_full)
        
    return(fc1)


pred = CNN_model(x)  


with tf.name_scope('Accuracy'):    
    cost = tf.reduce_mean(tf.square(tf.subtract(pred, y)))
    tf.summary.scalar('accuracy', cost)


with tf.name_scope('Optimiser'):
    optimise = tf.train.AdamOptimizer().minimize(cost)
    
merged = tf.summary.merge_all()

def train_model(epoch):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        with tf.device('/cpu:0'):  
            writer = tf.summary.FileWriter('./model/events/2')
            writer.add_graph(sess.graph)
        
        for ep in range(epoch):
            for i in range(len(train_dir)):
                batch_x = open_img(train_dir[i], 'train', 'L')
                batch_x = batch_x.reshape((batch_size, n_height, n_width, n_chnl))
                batch_y = open_img(mask_dir[i], 'mask', 'L')
                batch_y = batch_y.reshape((batch_size, n_height, n_width))
                _, mse = sess.run([optimise, cost], feed_dict = ({x : batch_x, y: batch_y}))
#                mse = sess.run([cost], feed_dict = ({x : batch_x, y: batch_y}))
                if i % 10  == 0:
                    summ = sess.run(merged, feed_dict = ({x : batch_x, y: batch_y}))
                    writer.add_summary(summ, ep * len(train_dir) + i)
                print('batch', str(i), 'mse:', mse)
                    
        with tf.name_scope('Save_model'):
            saver = tf.train.Saver()
            saver.save(sess, './model/mymodel')
            
    return(sess)


def use_model(train_index):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './model/mymodel')
        x_img = open_img(train_dir[train_index], 'train', 'L')
        out_mask = sess.run(pred, feed_dict={x : x_img.reshape((batch_size, n_height, n_width, n_chnl))})
        
        out_mask = out_mask[0]
        out_mask = (out_mask > 0) * 1 
        x_masked = out_mask * x_img
        plt.imshow(x_masked, cmap = 'gray')


#if __name__ == '__main__':
#    sess = train_model(1)         
#    use_model(0)
#            


    