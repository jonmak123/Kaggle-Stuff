# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:34:03 2017

@author: jmak

CMD:
    
    floyd run --tensorboard --data jonmak/datasets/vgg16/7:vgg_weights --data jonmak/datasets/carvana/5:input --gpu "python fcn_8.py"
"""
import numpy as np
import tensorflow as tf
import os
import Utils
import pandas as pd

tf.reset_default_graph()

cars = pd.read_csv('meta.csv')

img_size = 256
chnl = 3
num_class = 2
NUM_ITER = 50000
batch_size = 2
valid_batch_size = 3
valid_size = 50

img_folder = '/input/'
vgg_folder = '/vgg_weights/'
logs_dir = '/output/log/'

def model(x, keep_prob):
    ## NEED TO DOWNLOAD VGG NPY FILE ##
    weights = np.load(vgg_folder + 'vgg16.npy', encoding='latin1').item()
    
    norm = [123.68, 116.779, 103.939]
    x_norm = tf.subtract(x, norm)
    
    with tf.variable_scope('conv_1'):
        conv1_1 = Utils._conv_layer(x_norm, 'conv1_1', weights)
        conv1_2 = Utils._conv_layer(conv1_1, 'conv1_2', weights)
        pool1 = tf.nn.max_pool(conv1_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    
    with tf.variable_scope('conv_2'):
        conv2_1 = Utils._conv_layer(pool1, 'conv2_1', weights)
        conv2_2 = Utils._conv_layer(conv2_1, 'conv2_2', weights)
        pool2 = tf.nn.max_pool(conv2_2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope('conv_3'):    
        conv3_1 = Utils._conv_layer(pool2, 'conv3_1', weights)
        conv3_2 = Utils._conv_layer(conv3_1, 'conv3_2', weights)
        conv3_3 = Utils._conv_layer(conv3_2, 'conv3_3', weights)
        pool3 = tf.nn.max_pool(conv3_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope('conv_4'):    
        conv4_1 = Utils._conv_layer(pool3, 'conv4_1', weights)
        conv4_2 = Utils._conv_layer(conv4_1, 'conv4_2', weights)
        conv4_3 = Utils._conv_layer(conv4_2, 'conv4_3', weights)
        pool4 = tf.nn.max_pool(conv4_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope('conv_5'):    
        conv5_1 = Utils._conv_layer(pool4, 'conv5_1', weights, _trainable = True)
        conv5_2 = Utils._conv_layer(conv5_1, 'conv5_2', weights, _trainable = True)
        conv5_3 = Utils._conv_layer(conv5_2, 'conv5_3', weights, _trainable = True)
        pool5 = tf.nn.max_pool(conv5_3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.variable_scope('fc'):    
#        fc6 = Utils._fc_layer_train(pool5, 'fc6', [7, 7, 512, num_class])
#        dropout6 = tf.nn.dropout(fc6, keep_prob = keep_prob)       
        
        fc6 = Utils._fc_layer_train(pool5, 'fc6', [7, 7, 512, 200])   
        dropout6 = tf.nn.dropout(fc6, keep_prob = keep_prob) 
        fc7 = Utils._fc_layer_train(dropout6, 'fc7', [1, 1, 200, num_class])
        
#    ## FOR VGG TESTING ##
#    fc8 = Utils._fc_layer(dropout7, 'fc8', [1, 1, 4096, 1000] , weights, relu = False)
#    pred = tf.argmax(fc8, 3)
#    return(pred)
#    #####################
    
    with tf.variable_scope('deconv1'):     
        conv4_score = Utils.conv_score(pool4, num_class, 'skip4')
        upsample_1 = Utils.upsampling(dropout6, conv4_score, 4, 2, num_class, 'deconv1')
        fuse1 = tf.add(upsample_1, conv4_score)

    with tf.variable_scope('deconv2'):     
        conv3_score = Utils.conv_score(pool3, num_class, 'skip3')
        upsample_2 = Utils.upsampling(fuse1, conv3_score, 4, 2, num_class, 'deconv2')    
        fuse2 = tf.add(upsample_2, conv3_score)

    with tf.variable_scope('deconv3'): 
        upsample_3 = Utils.upsampling(fuse2, x, 16, 8, num_class, 'deconv3')
        pred = tf.multiply(tf.argmax(upsample_3, 3), 255)
    
    return(upsample_3, pred)

def main():    
    x = tf.placeholder(tf.float32, [None, img_size, img_size, chnl])
    annotation = tf.placeholder(tf.int32, shape=[None, img_size, img_size, 1])
    keep_prob = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    print("Setting up image reader...")
    all_train = cars['train'].tolist()
    all_annotation = cars['mask'].tolist()
    split_dummy = np.arange(len(all_train))
    np.random.shuffle(split_dummy)
    train_x = [all_train[i] for i in split_dummy[:-valid_size]]
    train_y = [all_annotation[i] for i in split_dummy[:-valid_size]]
    valid_x = [all_train[i] for i in split_dummy[-valid_size:]]
    valid_y = [all_annotation[i] for i in split_dummy[-valid_size:]]
    train_reader = Utils.batchreader_floyd(train_x, train_y, img_folder, img_size)
    valid_reader = Utils.batchreader_floyd(valid_x, valid_y, img_folder, img_size)
    
    print('Setting up inference model and loss...')
    logits, pred_annotation = model(x, keep_prob)
#    loss = Utils.softmax_loss(logits, annotation)
    loss = Utils.bce_dice(logits, annotation)
    learning_rate = tf.train.exponential_decay(1e-3, global_step, 1000, 0.90, staircase=True)
    optimiser = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
    print('Starting TF session...')
    sess = tf.Session()
    print('Initialising variables...')
    sess.run(tf.global_variables_initializer())
    num_train_var = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Number of trainable parameters:', num_train_var)
    
    print("Setting up Saver...")
    saver = tf.train.Saver()
    print('Setting up Tensorboard Writer...')
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    
    print('Setting TF Summary ops...')
    train_summ_op = tf.summary.merge([tf.summary.scalar("train_loss", loss)])
    
    valid_summ_op = tf.summary.merge([tf.summary.image("input_image", x),
                                      tf.summary.image("ground_truth", tf.cast(annotation, tf.float32)),
                                      tf.summary.image("pred_annotation", 
                                                       tf.cast(tf.reshape(pred_annotation, [valid_batch_size, img_size, img_size, 1]), tf.uint8)),
                                      tf.summary.scalar("valid_loss", loss)])
    
    print('Start Training!')
    for itr in range(NUM_ITER):
        itr = itr + 1
        train_images, train_annotations = train_reader.next_batch(batch_size)
        feed_dict = {x: train_images, annotation: train_annotations, keep_prob: 0.9}
        _ = sess.run(optimiser, feed_dict = feed_dict)
        
        if itr % 1000 == 0:
            print('Running itr:', str(itr))
            # RECORD TRAIN LOSS
            train_loss, summary_str = sess.run([loss, train_summ_op], feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, itr)
            # EVAL AND RECORD VALIDATION LOSS
            valid_images, valid_annotations = valid_reader.next_batch(valid_batch_size) 
            feed_dict = {x: valid_images, annotation: valid_annotations, keep_prob: 1.0}
            valid_loss, summary_str = sess.run([loss, valid_summ_op], feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, itr)
            print('Iteration:', str(itr),  '---',
                  'Global Step:', tf.train.global_step(sess, global_step),  '---',
                  'Learning Rate:', sess.run(learning_rate), '---',
                  'Train Loss:', str(round(train_loss, 5)) , '---', 
                  'Valid Loss:', str(round(valid_loss, 5)))
            
    saver.save(sess, logs_dir + 'carvana')
    
#def vgg_test():
#    synset = [l.strip() for l in open('synset.txt').readlines()]
#    
#    dir_ = os.listdir('test')
#    img = [Utils.open_img('test/' + x) for x in dir_]
#    img = [x.reshape((1, 224, 224, 3)) for x in img]
#    
#    x = tf.placeholder(tf.float32, [None, img_size, img_size, chnl])
#    keep_prob = tf.placeholder(tf.float32)
#    
#    pred = model(x, keep_prob)
#    
#    print('Starting TF session...')
#    sess = tf.Session()
#    print('Initialising variables...')
#    sess.run(tf.global_variables_initializer())
#    
##    image = np.stack(img)
#    image = img[1]
#    feed_dict = {x: image, keep_prob: 1}
#    prediction = sess.run(pred, feed_dict = feed_dict)
#    
#    out_index = np.argmax(np.bincount(prediction.flatten()))
#    print('Prediction is:', synset[out_index])
#    return(prediction)
        
        
if __name__ == '__main__':
    main()
#    with tf.device('/cpu:0'):
#        main()